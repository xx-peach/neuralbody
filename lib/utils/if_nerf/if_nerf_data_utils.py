import numpy as np
from lib.utils import base_utils
import cv2
from lib.config import cfg
import trimesh


def get_rays(H, W, K, R, T):
    """ Function for Ray Generation, Matrix Multiplicaion
    Args:
        H, W - int, the height and width of the input image
        K    - (3, 3) of float32 the camera intrinsic matrix
        R    - (3, 3) of float32, rotation matrix from world to camera
        T    - (3,) of float32, translation matrix from world to camera
    Returns:
        rays_o - (H, W, 3), duplication of (3, ) camera origin in world coordinate
        rays_d - (H, W, 3), camera directions in world coordinate, same as NeRF
    """
    # calculate the camera origin, RX + T = 0 -> X = R.inv @ -T
    rays_o = -np.dot(R.T, T).ravel()                    # (3,)
    
    # calculate the world coodinates of pixels using `np.meshgrid()`
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)     # (H, W, 3)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)      # (H, W, 3), same as K.inv @ xy1.T
    pixel_world = np.dot(pixel_camera - T.ravel(), R)   # (H, W, 3)
    
    #! calculate the ray direction, 下面减 rays_o 是因为上面是计算了 pixel_world 而不是只是乘了 R
    rays_d = pixel_world - rays_o[None, None]           # (H, W, 3)
    rays_o = np.broadcast_to(rays_o, rays_d.shape)      # (H, W, 3)
    return rays_o, rays_d


def get_bound_corners(bounds):
    """ Generate 3d Bounding Box's 8 Vertices Coordinate(World)
    Args:
        bounds - (2, 3) of float32, human vertices bound in world coordinates
    Returns:
        corners_3d - (8, 3) of float32, 3d bounding box coordinates of 6890 smpl vertices
    """
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    """ Generate 2d Image Mask Projected from 3d Vertices Bouning Box
    Args:
        bounds - (2, 3) of float32, human vertices bound in world coordinates
        K      - (3, 3) of float32, camera's intrinsic matrix
        pose   - (3, 4) of float32, project matrix from world to camera
        H, W   - int, the height and width of the input image
    Returns:
        mask - (H, W) of 0/1, 2d human mask projected from 3d smpl vertices bounding box
    """
    corners_3d = get_bound_corners(bounds)                  # (8, 3), 3d bounding box
    corners_2d = base_utils.project(corners_3d, K, pose)    # (8, 3), project 3d bounding box to image
    corners_2d = np.round(corners_2d).astype(int)           # (8, 3), rounded to int
    # use `cv2.fillPoly()` to draw the mask of 2d bounding box projected from 3d world bounding box
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)    # verticel left plane
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)    # verticle right plane
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)    # verticle back plane
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)    # verticle front plane
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)    # bottom plane
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)    # top plane
    return mask


def get_near_far(bounds, ray_o, ray_d):
    """ Calculate Intersections with 3D Bounding Box Using 3D Slabs Principle
        https://blog.csdn.net/weixin_40301728/article/details/114239266
    Args:
        bounds - (2, 3) of float32, human vertices bound in world coordinates
        ray_o  - (nbody+nrand, 3), duplication of (3, ) camera origin in world coordinate
        ray_d  - (nbody+nrand, 3), 这个还不知道到底是啥东西现在
    Returns:
        near        - (n,) of float32, near distance for those rays intersecting with the 3d bounding box
        far         - (n,) of float32, far distance for those rays intersecting with the 3d bounding box
        mask_at_box - (nbody+nrand,) of int32, 1 if that ray intersects with the 3d bounding box
    """
    # normalize all H*W input rays' direction, get H*W normalized view directions
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)  # (nbody+nrand, 1)
    viewdir = ray_d / norm_d                                # (nbody+nrand, 3)
    # truncate numerical small to 1e-5 and -1e-5
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5   # (nbody+nrand, 3)
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5  # (nbody+nrand, 3)

    # compute x_near, x_far, y_near, y_far, z_near, z_far for H*W rays respectively
    tmin = (bounds[:1]  - ray_o[:1]) / viewdir  # (nbody+nrand, 3), 这边减掉 ray_o 是因为生成 ray_d 时减了 ray_o
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir  # (nbody+nrand, 3), 这边减掉 ray_o 是因为生成 ray_d 时减了 ray_o

    # compute max(t_xnear, t_ynear, t_znear) and min(t_xfar, t_yfar, t_zfar)
    t1 = np.minimum(tmin, tmax)     # (nbody+nrand, 3), t_xnear, t_ynear, t_znear
    t2 = np.maximum(tmin, tmax)     # (nbody+nrand, 3), t_xfar, t_yfar, t_zfar
    near = np.max(t1, axis=-1)      # (nbody+nrand,), max(t_xnear, t_ynear, t_znear)
    far  = np.min(t2, axis=-1)      # (nbody+nrand,), min(t_xfar, t_yfar, t_zfar)
    
    # if max(t_xnear, t_ynear, t_znear) < min(t_xfar, t_yfar, t_zfar) -> intersect
    mask_at_box = near < far                            # (nbody+nrand)
    near = near[mask_at_box] / norm_d[mask_at_box, 0]   # (n,)
    far  =  far[mask_at_box] / norm_d[mask_at_box, 0]   # (n,)
    return near, far, mask_at_box


def sample_ray(img, msk, K, R, T, bounds, nrays, split):
    """ Sample Rays from Current Image(['train', 'test'])
    Args:
        img    - (H, W, 3) of float32, original image of this batch
        msk    - (H, W, 3) of float32, masked image of the batch
        K      - (3, 3) of float32, camera's intrinsic matrix
        R      - (3, 3) of float32, rotation matrix from world to camera
        T      - (3,) of float32, translation matrix from world to camera
        bounds - (2, 3) of float32, human vertices bound in world coordinates
        nrays  - int, namely choose how many rays for this batch
        split  - ['train', 'test']
    Returns:
        rgb         - (nrays, 3), rgb color of each corresponding ray
        ray_o       - (nrays, 3), rays origin in world coordinates
        ray_d       - (nrays, 3), rays direction in world coordinates
        near        - (nrays,), near distance of each ray
        far         - (nrays,), far distance of each ray
        coord       - (nrays, 2), correspoding image indices (u, v) of n rays
        mask_at_box - #? (batch_size,), 都是 1 的一个 array...
    """
    # get H*W rays' origin and direction
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)          # (H, W, 3)

    # generate 2d human mask which is projected from 3d smpl vertices bounding box
    pose = np.concatenate([R, T], axis=1)           # (3, 4)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)
    # mask the original input human mask further
    msk = msk * bound_mask

    # sample nrays(batch_size) rays from H*W rays if it is train loader
    if split == 'train':
        nsampled_rays = 0                           # number of rays we've sampled
        face_sample_ratio = cfg.face_sample_ratio   # face_sample_ratio
        body_sample_ratio = cfg.body_sample_ratio   # body_sample_ratio
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        while nsampled_rays < nrays:
            # specify number of rays sampled from human body, face and bound mask just generated
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)   # num of rays sampled from human body
            n_face = int((nrays - nsampled_rays) * face_sample_ratio)   # num of rays sampled from human face
            n_rand = (nrays - nsampled_rays) - n_body - n_face          # num of rays remaining, sampled from bound mask

            # sample rays on body, msk != 0's places are all human body
            coord_body = np.argwhere(msk != 0)
            coord_body = coord_body[np.random.randint(0, len(coord_body), n_body)]
            # sample rays on face, msk == 13's place human face
            coord_face = np.argwhere(msk == 13)
            if len(coord_face) > 0:
                coord_face = coord_face[np.random.randint(0, len(coord_face), n_face)]
            # sample rays in the bound mask from bound_mask that just generated
            coord = np.argwhere(bound_mask == 1)
            coord = coord[np.random.randint(0, len(coord), n_rand)]

            # concatenate all the sampled rays indices btw [[0, 0], ... [H-1, W-1]]
            if len(coord_face) > 0:
                coord = np.concatenate([coord_body, coord_face, coord], axis=0)
            else:
                coord = np.concatenate([coord_body, coord], axis=0)
            
            # fetch ray_o, ray_d, and corresponding rgb using sampled indices
            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]        # (nbody+nrand, 3)
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]        # (nbody+nrand, 3)
            rgb_   =   img[coord[:, 0], coord[:, 1]]        # (nbody+nrand, 3)

            # generate near, far distance for each sampled rays, and further filter rays
            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_) # (n, )

            ray_o_list.append(ray_o_[mask_at_box])              # (n, 3), rays origin in world coordinates
            ray_d_list.append(ray_d_[mask_at_box])              # (n, 3), rays direction in world coordinates
            rgb_list.append(rgb_[mask_at_box])                  # (n, 3), rgb color of each corresponding ray
            near_list.append(near_)                             # (n,), near distance of each ray
            far_list.append(far_)                               # (n,), far distance of each ray
            coord_list.append(coord[mask_at_box])               # (n, 2), correspoding image indices (u, v) of n rays
            mask_at_box_list.append(mask_at_box[mask_at_box])   #? 这样不就是全 1 了吗?
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    
    # generate H*W rays and filter those who intersects with the 3d bounding box if it's test loader
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def sample_ray_h36m(img, msk, K, R, T, bounds, nrays, split):
    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0

    if split == 'train':
        nsampled_rays = 0
        face_sample_ratio = cfg.face_sample_ratio
        body_sample_ratio = cfg.body_sample_ratio
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_face = int((nrays - nsampled_rays) * face_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body - n_face

            # sample rays on body
            coord_body = np.argwhere(msk == 1)
            coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                      n_body)]
            # sample rays on face
            coord_face = np.argwhere(msk == 13)
            if len(coord_face) > 0:
                coord_face = coord_face[np.random.randint(
                    0, len(coord_face), n_face)]
            # sample rays in the bound mask
            coord = np.argwhere(bound_mask == 1)
            coord = coord[np.random.randint(0, len(coord), n_rand)]

            if len(coord_face) > 0:
                coord = np.concatenate([coord_body, coord_face, coord], axis=0)
            else:
                coord = np.concatenate([coord_body, coord], axis=0)

            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]

            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.zeros([len(rgb), 2]).astype(np.int64)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box


def get_smpl_data(ply_path):
    ply = trimesh.load(ply_path)
    xyz = np.array(ply.vertices)
    nxyz = np.array(ply.vertex_normals)

    if cfg.add_pointcloud:
        # add random points
        xyz_, ind_ = trimesh.sample.sample_surface_even(ply, 5000)
        nxyz_ = ply.face_normals[ind_]
        xyz = np.concatenate([xyz, xyz_], axis=0)
        nxyz = np.concatenate([nxyz, nxyz_], axis=0)

    xyz = xyz.astype(np.float32)
    nxyz = nxyz.astype(np.float32)

    return xyz, nxyz


def get_acc(coord, msk):
    border = 25
    kernel = np.ones((border, border), np.uint8)
    msk = cv2.dilate(msk.copy(), kernel)
    acc = msk[coord[:, 0], coord[:, 1]]
    acc = (acc != 0).astype(np.uint8)
    return acc


def rotate_smpl(xyz, nxyz, t):
    """
    t: rotation angle
    """
    xyz = xyz.copy()
    nxyz = nxyz.copy()
    center = (np.min(xyz, axis=0) + np.max(xyz, axis=0)) / 2
    xyz = xyz - center
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    R = R.astype(np.float32)
    xyz[:, :2] = np.dot(xyz[:, :2], R.T)
    xyz = xyz + center
    # nxyz[:, :2] = np.dot(nxyz[:, :2], R.T)
    return xyz, nxyz, center


def transform_can_smpl(xyz):
    center = np.array([0, 0, 0]).astype(np.float32)
    rot = np.array([[np.cos(0), -np.sin(0)], [np.sin(0), np.cos(0)]])
    rot = rot.astype(np.float32)
    trans = np.array([0, 0, 0]).astype(np.float32)
    if np.random.uniform() > cfg.rot_ratio:
        return xyz, center, rot, trans

    xyz = xyz.copy()

    # rotate the smpl
    rot_range = np.pi / 32
    t = np.random.uniform(-rot_range, rot_range)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    rot = rot.astype(np.float32)
    center = np.mean(xyz, axis=0)
    xyz = xyz - center
    xyz[:, [0, 2]] = np.dot(xyz[:, [0, 2]], rot.T)
    xyz = xyz + center

    # translate the smpl
    x_range = 0.05
    z_range = 0.025
    x_trans = np.random.uniform(-x_range, x_range)
    z_trans = np.random.uniform(-z_range, z_range)
    trans = np.array([x_trans, 0, z_trans]).astype(np.float32)
    xyz = xyz + trans

    return xyz, center, rot, trans


def unproject(depth, K, R, T):
    H, W = depth.shape
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    xyz = xy1 * depth[..., None]
    pts3d = np.dot(xyz, np.linalg.inv(K).T)
    pts3d = np.dot(pts3d - T.ravel(), R)
    return pts3d


def sample_world_points(ray_o, ray_d, near, far, split):
    # calculate the steps for each ray
    t_vals = np.linspace(0., 1., num=cfg.N_samples)
    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

    if cfg.perturb > 0. and split == 'train':
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = np.concatenate([mids, z_vals[..., -1:]], -1)
        lower = np.concatenate([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = np.random.rand(*z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    pts = ray_o[:, None] + ray_d[:, None] * z_vals[..., None]
    pts = pts.astype(np.float32)
    z_vals = z_vals.astype(np.float32)

    return pts, z_vals


def barycentric_interpolation(val, coords):
    """
    :param val: verts x 3 x d input matrix
    :param coords: verts x 3 barycentric weights array
    :return: verts x d weighted matrix
    """
    t = val * coords[..., np.newaxis]
    ret = t.sum(axis=1)
    return ret


def batch_rodrigues(poses):
    """ poses: N x 3
    """
    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat


def get_rigid_transformation(poses, joints, parents):
    """
    poses: 24 x 3
    joints: 24 x 3
    parents: 24
    """
    rot_mats = batch_rodrigues(poses)

    # obtain the relative joints
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([24, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    # obtain the rigid transformation
    padding = np.zeros([24, 1])
    joints_homogen = np.concatenate([joints, padding], axis=1)
    transformed_joints = np.sum(transforms * joints_homogen[:, None], axis=2)
    transforms[..., 3] = transforms[..., 3] - transformed_joints
    transforms = transforms.astype(np.float32)

    return transforms
