import torch.nn as nn
import spconv.pytorch as spconv
import torch.nn.functional as F
import torch
from lib.config import cfg
from . import embedder


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.c = nn.Embedding(6890, 16)
        self.xyzc_net = SparseConvNet()

        self.latent = nn.Embedding(cfg.num_train_frame, 128)

        self.actvn = nn.ReLU()

        self.fc_0 = nn.Conv1d(352, 256, 1)
        self.fc_1 = nn.Conv1d(256, 256, 1)
        self.fc_2 = nn.Conv1d(256, 256, 1)
        self.alpha_fc = nn.Conv1d(256, 1, 1)

        self.feature_fc = nn.Conv1d(256, 256, 1)
        self.latent_fc = nn.Conv1d(384, 256, 1)
        self.view_fc = nn.Conv1d(346, 128, 1)
        self.rgb_fc = nn.Conv1d(128, 3, 1)

    def encode_sparse_voxels(self, sp_input):
        """ Code Diffusion via Sparse Convolution
        Args:
            sp_input['coord']  - (batch_size*6890, 4), indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
            sp_input['out_sh'] - (3,), 该 batch 中所有 frame 里面最大的 voxel shape(spatial shape of sparse tensor, spatial_shape[i] is shape of indices[:, 1+i])
            sp_input['batch_size'] - int, batch_size of current configuration, batch size of this sparse tensor
        Returns:
            feature_volume - [(bs, 32, d/2, h/2, w/2), (bs, 64, d/4, h/4, w/4), (bs, 128, d/8, h/8, w/8), (bs, 128, d/16, h/16, w/16)]
                             it is a list of output of layer [5, 9, 13, 17]
        """
        # fetch needed data
        coord = sp_input['coord']           # (batch_size*6890, 4)
        out_sh = sp_input['out_sh']         # (3,)
        batch_size = sp_input['batch_size'] # int

        # generate 6890 vertice codes for each frame using nn.Embedding()
        code = self.c(torch.arange(0, 6890).to(coord.device))   # (6890, 16)

        # generate SparseConvTensor() for SparseConv op
        xyzc = spconv.SparseConvTensor(code, coord, out_sh, batch_size)
        # go through SparseConvNet() to diffuse latent code to nearby volume
        feature_volume = self.xyzc_net(xyzc)

        return feature_volume

    def pts_to_can_pts(self, pts, sp_input):
        """ Transform pts from the World Coordinate to the smpl Coordinate
        Args:
            pts            - (batch_size, chunk*N_samples, 3), N_sample points in the world coordinates
            sp_input['Th'] - (batch_size, 3), translation matrix of current smpl smpl2world
            sp_input['R']  - (batch_size, 3, 3), rotation matrix of current smpl smpl2world
        Returns:
            pts - (batch_size, chunk*N_samples, 3), N_sample points in the smpl coordinates
        """
        Th = sp_input['Th']
        pts = pts - Th[:, None]     #! 有错，进行了一下修改
        R = sp_input['R']
        pts = torch.matmul(pts, R)
        return pts

    def get_grid_coords(self, pts, sp_input):
        # convert xyz to the voxel coordinate dhw
        dhw = pts[..., [2, 1, 0]]                           # (batch_size, chunk*N_samples, 3)
        min_dhw = sp_input['bounds'][:, 0, [2, 1, 0]]       # (batch_size, 3)
        dhw = dhw - min_dhw[:, None]                        # (batch_size, chunk*N_samples, 3)
        dhw = dhw / torch.tensor(cfg.voxel_size).to(dhw)    # (batch_size, chunk*N_samples, 3)
        # convert the voxel coordinate to [-1, 1]
        out_sh = torch.tensor(sp_input['out_sh']).to(dhw)   # (3,)
        dhw = dhw / out_sh * 2 - 1                          # (batch_size, chunk*N_samples, 3)
        # convert dhw to whd, since the occupancy is indexed by dhw
        grid_coords = dhw[..., [2, 1, 0]]
        return grid_coords

    def interpolate_features(self, grid_coords, feature_volume):
        features = []
        for volume in feature_volume:
            feature = F.grid_sample(volume,                 # (N, C, Din, Hin, Win) <- (bs, ci, d/i, h/i, w/i)
                                    grid_coords,            # (N, Dot, Hot, Wot, 3) <- (bs, 1, 1, chunk*N_samples, 3)
                                    padding_mode='zeros',
                                    align_corners=True)     # (N, C, Dot, Hot, Wot) <- (bs, ci, 1, 1, chunk*N_samples)
            features.append(feature)
        features = torch.cat(features, dim=1)               # (bs, c1+c2+c3+c4, 1, 1, chunk*N_samples)
        features = features.view(features.size(0), -1, features.size(4))    # (bs, c1+c2+c3+c4, chunk*N_samples)
        return features

    def calculate_density(self, wpts, feature_volume, sp_input):
        # interpolate features
        ppts = self.pts_to_can_pts(wpts, sp_input)
        grid_coords = self.get_grid_coords(ppts, sp_input)
        grid_coords = grid_coords[:, None, None]
        xyzc_features = self.interpolate_features(grid_coords, feature_volume)

        # calculate density
        net = self.actvn(self.fc_0(xyzc_features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))

        alpha = self.alpha_fc(net)
        alpha = alpha.transpose(1, 2)

        return alpha

    def calculate_density_color(self, wpts, viewdir, feature_volume, sp_input):
        """ Compute Latent Code for each Sampled Point and Go Through NeRF Net to Compute its Color+Density
        Args:
            wpts    - (batch_size, chunk*N_samples, 3), N_sample points for each ray in one frame and frame in this batch
            viewdir - (batch_size, chunk*N_samples, 3), normalized rays direction for each ray in this batch
            feature_volume - [(bs, 32, d/2, h/2, w/2), (bs, 64, d/4, h/4, w/4), (bs, 128, d/8, h/8, w/8), (bs, 128, d/16, h/16, w/16)]
            sp_input       - dict(), same input for SparseConvNet()
        Returns:
            raw - (batch_size, chunk*N_samples, 4), raw NeRF network output, namely rgb+alpha
        """
        # interpolate features
        ppts = self.pts_to_can_pts(wpts, sp_input)          # (batch_size, chunk*N_samples, 3)
        grid_coords = self.get_grid_coords(ppts, sp_input)  # (batch_size, chunk*N_samples, 3)
        grid_coords = grid_coords[:, None, None]    # (batch_size, 1, 1, chunk*N_samples, 3)
        xyzc_features = self.interpolate_features(grid_coords, feature_volume)  # (batch_size, c1+c2+c3+c4, chunk*N_samples)

        # calculate density
        net = self.actvn(self.fc_0(xyzc_features))  # (batch_size, 256, chunk*N_samples)
        net = self.actvn(self.fc_1(net))            # (batch_size, 256, chunk*N_samples)
        net = self.actvn(self.fc_2(net))            # (batch_size, 256, chunk*N_samples)

        alpha = self.alpha_fc(net)                  # (batch_size, 1, chunk*N_samples)

        # calculate color
        features = self.feature_fc(net)             # (batch_size, 256, chunk*N_samples)

        # latent code for each frame(这是建模)
        latent = self.latent(sp_input['latent_index'])  # (batch_size, 128)
        latent = latent[..., None].expand(*latent.shape, net.size(2))   # (batch_size, 128, chunk*N_samples)
        features = torch.cat((features, latent), dim=1) # (batch_size, 384, chunk*N_samples)
        features = self.latent_fc(features)             # (batch_size, 256, chunk*N_samples)

        viewdir = embedder.view_embedder(viewdir)
        viewdir = viewdir.transpose(1, 2)
        light_pts = embedder.xyz_embedder(wpts)
        light_pts = light_pts.transpose(1, 2)

        features = torch.cat((features, viewdir, light_pts), dim=1)

        net = self.actvn(self.view_fc(features))
        rgb = self.rgb_fc(net)                      # (batch_size, 3, chunk*N_samples)

        raw = torch.cat((rgb, alpha), dim=1)        # (batch_size, 4, chunk*N_samples)
        raw = raw.transpose(1, 2)                   # (batch_size, chunk*N_samples, 4)

        return raw

    def forward(self, sp_input, grid_coords, viewdir, light_pts):
        coord = sp_input['coord']
        out_sh = sp_input['out_sh']
        batch_size = sp_input['batch_size']

        p_features = grid_coords.transpose(1, 2)
        grid_coords = grid_coords[:, None, None]

        code = self.c(torch.arange(0, 6890).to(p_features.device))
        xyzc = spconv.SparseConvTensor(code, coord, out_sh, batch_size)

        xyzc_features = self.xyzc_net(xyzc, grid_coords)

        net = self.actvn(self.fc_0(xyzc_features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))

        alpha = self.alpha_fc(net)

        features = self.feature_fc(net)

        latent = self.latent(sp_input['latent_index'])
        latent = latent[..., None].expand(*latent.shape, net.size(2))
        features = torch.cat((features, latent), dim=1)
        features = self.latent_fc(features)

        viewdir = viewdir.transpose(1, 2)
        light_pts = light_pts.transpose(1, 2)
        features = torch.cat((features, viewdir, light_pts), dim=1)
        net = self.actvn(self.view_fc(features))
        rgb = self.rgb_fc(net)

        raw = torch.cat((rgb, alpha), dim=1)
        raw = raw.transpose(1, 2)

        return raw


class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(16, 16, 'subm0')
        self.down0 = stride_conv(16, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x):
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()

        volumes = [net1, net2, net3, net4]

        return volumes


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())
