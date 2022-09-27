import torch
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder


class Renderer:
    def __init__(self, net):
        self.net = net

    def get_sampling_points(self, ray_o, ray_d, near, far):
        """ Sample N_sample Points from each Ray
        Args:
            ray_o - (batch_size, chunk, 3), rays origin in the world coordinates
            ray_d - (batch_size, chunk, 3), rays direction in world coordinates
            near  - (batch_size, chunk), near distance of each ray in world coordinates
            far   - (batch_size, chunk), far distance of each ray in world coordinates
        Returns:
            pts    - (batch_size, chunk, N_samples, 3), N_sample points for each ray in one frame and frame in this batch
            z_vals - (batch_size, chunk, N_samples), N_sample points' z value along the their corresponding ray direction
        """
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)       # (N_samples,)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals  # (batch_size, chunk, N_samples)

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]     # (batch_size, chunk, N_samples, 3)

        return pts, z_vals

    def prepare_sp_input(self, batch):
        """ Prepare Input for SparseConv, Add Frame Index for Voxel Coordinate
        Args:
            batch - dict(), original data batch returned by dataloader, 即原本 __getitem__ 返回的 dict 中的每一个元素都增加了 dim0=batch_size
        Returns:
            sp_input['coord']        - (batch_size*6890, 4), 在原本的每个 frame 的 6890 个 vertice 的三轴 voxel index 前面加了 frame index
            sp_input['out_sh']       - (3,), 该 batch 中的 batch_size 个 frame 里面最大的 whole voxel size 范围
            sp_input['batch_size']   - int, batch_size of current configuration
            sp_input['bounds']       - (batch_size, 2, 3), human vertices bound in smpl coordinates
            sp_input['R']            - (batch_size, 3, 3), rotation matrix of current smpl smpl2world
            sp_input['Th']           - (batch_size, 3), translation matrix of current smpl smpl2world
            sp_input['latent_index'] - (batch_size, 1), frame_index of frames inside this batch
        """
        # feature, coordinate, shape, batch size
        sp_input = {}

        # add frame index to batch['coord'], 因为 a batch 里有 batch_size 个 frame 数据
        # coordinate: [N, 4], batch_idx, z, y, x, where N = batch_size * 6890
        sh = batch['coord'].shape                               # (batch_size, 6890, 3) --> sh
        idx = [torch.full([sh[1]], i) for i in range(sh[0])]    # [(6890,), ..., (6890,)]
        idx = torch.cat(idx).to(batch['coord'])                 # (batch_size*6890,), same type as batch['coord']
        coord = batch['coord'].view(-1, sh[-1])                 # (batch_size*6890, 3)
        sp_input['coord'] = torch.cat([idx[:, None], coord], dim=1) # (batch_size*6890, 4)

        # find the biggest output shape, 即该 batch 所有 frame 最大的 whole voxel size
        out_sh, _ = torch.max(batch['out_sh'], dim=0)   # out_sh: (3,), 即该 batch 中所有 frame 中 voxel size 最大的那个
        sp_input['out_sh'] = out_sh.tolist()            # out_sh.type = list
        sp_input['batch_size'] = sh[0]                  # batch_size

        # used for feature interpolation
        sp_input['bounds'] = batch['bounds']    # (batch_size, 2, 3), human vertices bound in smpl coordinates
        sp_input['R'] = batch['R']              # (batch_size, 3, 3), rotation matrix of current smpl smpl2world
        sp_input['Th'] = batch['Th']            # (batch_size, 3), translation matrix of current smpl smpl2world

        # used for color function
        sp_input['latent_index'] = batch['latent_index']    # (batch_size, 1)

        return sp_input

    def get_density_color(self, wpts, viewdir, raw_decoder):
        """ Prepare Input Sampled-Points and View Directions for NeRF
        Args:
            wpts        - (batch_size, chunk, N_samples, 3), N_sample points for each ray in one frame and frame in this batch
            viewdir     - (batch_size, chunk, 3), normalized rays direction for each ray in this batch
            raw_decoder - a lambda function that call the true Network.calculate_density_color()
        Returns:
            raw - (), raw NeRF network output, namely rgb+alpha
        """
        n_batch, n_pixel, n_sample = wpts.shape[:3]         # batch_size, chunk, N_samples
        wpts = wpts.view(n_batch, n_pixel * n_sample, -1)   # (batch_size, chunk*N_samples, 3)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()    # (batch_size, chunk, N_samples, 3)
        viewdir = viewdir.view(n_batch, n_pixel * n_sample, -1)                 # (batch_size, chunk*N_samples, 3)
        raw = raw_decoder(wpts, viewdir)
        return raw

    def get_pixel_value(self, ray_o, ray_d, near, far, feature_volume,
                        sp_input, batch):
        """ Given ray_o, ray_d, near, far, feature_volume, Go Through NeRF to Get Density and Color
        Args:
            ray_o          - (batch_size, chunk, 3), rays origin in the world coordinates
            ray_d          - (batch_size, chunk, 3), rays direction in world coordinates
            near           - (batch_size, chunk), near distance of each ray in world coordinates
            far            - (batch_size, chunk), far distance of each ray in world coordinates
            feature_volume - [(bs, 32, d/2, h/2, w/2), (bs, 64, d/4, h/4, w/4), (bs, 128, d/8, h/8, w/8), (bs, 128, d/16, h/16, w/16)]
            sp_input       - dict(), same input for SparseConvNet()
            batch          - dict(), original input returned by dataloader
        Returns:
            rgb_map   - (batch_size*chunk, 3), RGB color of rays in this batch
            disp_map  - (batch_size*chunk,), disparity of rays in this batch
            acc_map   - (batch_size*chunk,), sum of weights along each ray
            weights   - (batch_size*chunk, N_samples),  weights assigned to each sampled color
            depth_map - (batch_size*chunk,), depth of rays in this batch
        """
        # sampling points along camera rays
        wpts, z_vals = self.get_sampling_points(ray_o, ray_d, near, far)    # (batch_size, chunk, N_samples, 3)

        # viewing direction
        viewdir = ray_d / torch.norm(ray_d, dim=2, keepdim=True)            # (batch_size, chunk, 3)

        raw_decoder = lambda x_point, viewdir_val: self.net.calculate_density_color(
            x_point, viewdir_val, feature_volume, sp_input)

        # compute the color and density
        wpts_raw = self.get_density_color(wpts, viewdir, raw_decoder)       # (batch_size, chunk*N_samples, 4)

        # volume rendering for wpts
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        raw = wpts_raw.reshape(-1, n_sample, 4)     # (batch_size*chunk, N_samples, 4)
        z_vals = z_vals.view(-1, n_sample)          # (batch_size*chunk, N_samples)
        ray_d = ray_d.view(-1, 3)                   # (batch_size*chunk, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, ray_d, cfg.raw_noise_std, cfg.white_bkgd)

        ret = {
            'rgb_map': rgb_map.view(n_batch, n_pixel, -1),
            'disp_map': disp_map.view(n_batch, n_pixel),
            'acc_map': acc_map.view(n_batch, n_pixel),
            'weights': weights.view(n_batch, n_pixel, -1),
            'depth_map': depth_map.view(n_batch, n_pixel)
        }

        return ret

    def render(self, batch):
        # fetch needed data from current batch
        ray_o = batch['ray_o']  # (batch_size, nrays, 3), rays origin in world coordinates
        ray_d = batch['ray_d']  # (batch_size, nrays, 3), rays direction in world coordinates
        near = batch['near']    # (batch_size, nrays), near distance of each ray in world coordinates
        far = batch['far']      # (batch_size, nrays), far distance of each ray in world coordinates
        sh = ray_o.shape        # (batch_size, nrays, 3) --> sh

        # prepare sparse input for sparse convolution
        sp_input = self.prepare_sp_input(batch)
        # perform SparseConvNet() to get diffused feature(code) volume
        feature_volume = self.net.encode_sparse_voxels(sp_input)

        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]  # n_batch=batch_size, n_pixel=nrays
        chunk = 2048                        # avoid CUDA out of memory
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk,
                                               feature_volume, sp_input, batch)
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        return ret
