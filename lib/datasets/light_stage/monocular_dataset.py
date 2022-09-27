import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
from lib.utils import snapshot_data_utils as snapshot_dutils


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        """ Dataset __init__() Function
        Args:
            data_root - str, the root data directory, eg. 'data/people_snapshot/female-3-casual'
            human     - str, which specific human's data we're gonna use, eg. 'female-3-casual'
            ann_file  - str, the path for 'params.npy' which stores each frame's smpl parameters, including
                        beta, pose and tran, eg. 'data/people_snapshot/female-3-casual/params.npy'
            split     - str, ['train', 'test'], indicating which kind of dataset it is
        """
        super(Dataset, self).__init__()

        self.data_root = data_root      # eg. 'data/people_snapshot/female-3-casual'
        self.human = human              # eg. 'female-3-casual'
        self.split = split              # ['train', 'test']

        camera_path = os.path.join(self.data_root, 'camera.pkl')        # eg. 'data/people_snapshot/female-3-casual/female-3-casual'
        self.cam = snapshot_dutils.get_camera(camera_path)              # {'K': K, 'R': R, 'T': T, 'D': D}, R 是 (3,3) 单位矩阵, T 是 (3,) 零矩阵
        self.num_train_frame = cfg.num_train_frame                      # eg. cfg.num_train_frame: 230

        params_path = ann_file          # eg. 'data/people_snapshot/female-3-casual/params.npy'
        self.params = np.load(params_path, allow_pickle=True).item()    # {'beta': (10,), 'pose': (648, 72), 'trans': (648, 3)}

        self.nrays = cfg.N_rand         # rays sampled from one image at once, eg. 1024

    def prepare_input(self, i):
        """ Transform 6890 Vertices of Current Frame's SMPL from World to SMPL
        Args:
            i - data index that this sampler is gonna use 
        Returns:
            coord      - (6890, 3) of int32, voxel indice for each vertice of this smpl, in smpl coordinates
            out_sh     - (3,) of int32, voxel indices(multiplication of 32) bound for the whole human
            can_bounds - (2, 3) of float32, human vertices bound in world coordinates
            bounds     - (2, 3) of float32, human vertices bound in smpl coordinates
            Rh         - (3,) of float32, rotation axis + rotation angle, before Rodrigues Transformation, smpl2world
            Th         - (3,) of float32, translation matrix smpl2world
        """
        # read 6890 vertices' world coordinates from the npy file
        vertices_path = os.path.join(self.data_root, 'vertices', '{}.npy'.format(i))
        xyz = np.load(vertices_path).astype(np.float32)     # (6890, 3), world coordinates
        nxyz = np.zeros_like(xyz).astype(np.float32)

        # obtain the original bounds(world coordinates) for point sampling
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[1] -= 0.1                                   # (3,), [x_min, y_min-0.1, z_min]
        max_xyz[1] += 0.1                                   # (3,), [x_max, y_max+0.1, z_max]
        can_bounds = np.stack([min_xyz, max_xyz], axis=0)   # (2, 3), bounds in world

        # transform smpl from the world coordinate to the smpl coordinate
        Rh = self.params['pose'][i][:3]
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)         # root joint rotation (3, 3), smpl to world
        Th = self.params['trans'][i].astype(np.float32)     # root joint translation (3,), smpl to world
        # (xyz-Th) @ R == (R.inv @ (xyz-Th).T).T: R.T==R.inv
        xyz = np.dot(xyz - Th, R)                           # (6890, 3), vertices in smpl coordinates

        # obtain the bounds(smpl coordinates or canonical) for coord construction
        min_xyz = np.min(xyz, axis=0)
        max_xyz = np.max(xyz, axis=0)
        min_xyz[1] -= 0.1                                   # (3,), [x_min, y_min-0.1, z_min]
        max_xyz[1] += 0.1                                   # (3,), [x_max, y_max+0.1, z_max]
        bounds = np.stack([min_xyz, max_xyz], axis=0)       # (2, 3), bounds in smpl

        # construct the coordinate(voxel coordinates with unit voxel 5x5x5 mm)
        dhw = xyz[:, [2, 1, 0]]                             #? 干嘛换到 [z, y, x], (6890, 3), each point is [z, y, x]
        min_dhw = min_xyz[[2, 1, 0]]                        #? 干嘛换到 [z, y, x]
        max_dhw = max_xyz[[2, 1, 0]]                        #? 干嘛换到 [z, y, x]
        voxel_size = np.array(cfg.voxel_size)               # (3,): [0.005, 0.005, 0.005], 5mm
        # compute the voxel indice for each vertice of this smpl, in smpl coordinates
        coord = np.round((dhw - min_dhw) / voxel_size).astype(np.int32) # (6890, 3), [z, y, x]

        # construct the output shape, 这边的操作是把整体 voxel size 变成 32 的倍数
        out_sh = np.ceil((max_dhw - min_dhw) / voxel_size).astype(np.int32)
        x = 32
        out_sh = (out_sh | (x - 1)) + 1                     # (3,), [z, y, x]

        return coord, out_sh, can_bounds, bounds, Rh, Th

    def __getitem__(self, index):
        """ Custom __getitem__() Function used by torch.utils.data.DataLoader
        Args:
            index - data index that this sampler is gonna use
        Retunrs:
            ret - a dict who has all the processed data
        """
        # read i-th original image from file
        img_path = os.path.join(self.data_root, 'image', '{}.jpg'.format(index))
        img = imageio.imread(img_path).astype(np.float32) / 255.    # (H, W)
        # read i-th masked image(only human segmented) from file
        msk_path = os.path.join(self.data_root, 'mask', '{}.png'.format(index))
        msk = imageio.imread(msk_path)                              # (H, W)

        frame_index = index
        latent_index = index
        # undistort the original image and masked image
        K = self.cam['K']
        D = self.cam['D']
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)
        # get the extrinsic matrix(identity matrix)
        R = self.cam['R']                                           # (3, 3), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        T = self.cam['T'][:, None]                                  # (3, 1), [[0], [0], [0]]
        RT = np.concatenate([R, T], axis=1).astype(np.float32)      # (3, 4), world2camera

        # transform 6890 vertices of current frame's smpl from world to smpl
        coord, out_sh, can_bounds, bounds, Rh, Th = self.prepare_input(frame_index)

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        # mask the background / white the background if specified
        if cfg.mask_bkgd:
            img[msk == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1
        K = K.copy().astype(np.float32)
        K[:2] = K[:2] * cfg.ratio

        # sample nrays rays from current image if train, or all rays intersects with 3d bounding box if test
        rgb, ray_o, ray_d, near, far, coord_, mask_at_box = if_nerf_dutils.sample_ray(
            img, msk, K, R, T, can_bounds, self.nrays, self.split)

        ret = {
            'coord': coord,             # (6890, 3) of int32, voxel indice for each vertice of this smpl, in smpl coordinates
            'out_sh': out_sh,           # (3,) of int32, voxel indices(multiplication of 32) bound for the whole human
            'rgb': rgb,                 # (nrays, 3), rgb color of each corresponding ray
            'ray_o': ray_o,             # (nrays, 3), rays origin in world coordinates
            'ray_d': ray_d,             # (nrays, 3), rays direction in world coordinates
            'near': near,               # (nrays,), near distance of each ray
            'far': far,                 # (nrays,), far distance of each ray
            'mask_at_box': mask_at_box, #?(nrays,), 都是 1 的一个 array...
            'msk': msk                  # (H, W), image mask after resolution reduction by ratio
        }

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {
            'bounds': bounds,               # (2, 3) of float32, human vertices bound in smpl coordinates
            'R': R,                         # (3, 3), rotation matrix of current smpl smpl2world
            'Th': Th,                       # (3,), translation matrix of current smpl smpl2world
            'latent_index': latent_index,   # int, current frame index
            'frame_index': frame_index,     # int, current frame index
            'view_index': 0                 # int, default 0
        }
        ret.update(meta)

        Rh0 = self.params['pose'][index][:3]                        #? prepare_input 不是返回过了吗
        R0 = cv2.Rodrigues(Rh0)[0].astype(np.float32)               #? prepare_input 不是返回过了吗
        Th0 = self.params['trans'][index].astype(np.float32)        #? prepare_input 不是返回过了吗
        meta = {'R0_snap': R0, 'Th0_snap': Th0, 'K': K, 'RT': RT}
        ret.update(meta)

        return ret

    def __len__(self):
        return self.num_train_frame
