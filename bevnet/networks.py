import functools
import numpy as np
import spconv
import torch
import torch.nn as nn
import fchardnet
import convgru


class VoxelFeatureExtractorV3(nn.Module):
    def __init__(self):
        super(VoxelFeatureExtractorV3, self).__init__()

    def forward(self, features, num_voxels):
        # features: [concated_num_points, num_voxel_size, n_dim]
        # num_voxels: [concated_num_points]
        points_mean = features.sum(
            dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()


class VoxelFeatureExtractorV3MultiStep(nn.Module):
    def __init__(self):
        super(VoxelFeatureExtractorV3MultiStep, self).__init__()

    def forward(self, features, num_voxels):
        # features: T x [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: T x [concated_num_points]
        # returns list of T x [num_voxels]
        t = len(features)
        output = []
        for i in range(t):
            features_single = features[i]
            num_single = num_voxels[i]
            points_mean = features_single.sum(
                dim=1, keepdim=False) / num_single.type_as(features_single).view(-1, 1)
            output.append(points_mean.contiguous())
        return output


class SpMiddleNoDownsampleXY(nn.Module):
    """
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self,
                 output_shape,
                 num_input_features):
        super(SpMiddleNoDownsampleXY, self).__init__()

        BatchNorm1d = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        SpConv3d = functools.partial(spconv.SparseConv3d, bias=False)
        SubMConv3d = functools.partial(spconv.SubMConv3d, bias=False)

        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 32, 3, indice_key="subm0"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 32, 3, indice_key="subm0"),
            BatchNorm1d(32),
            nn.ReLU(),
            SpConv3d(32, 64, 3, (2, 1, 1), padding=[1, 1, 1]),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, 3, (2, 1, 1), padding=[0, 1, 1]),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SpConv3d(64, 64, (3, 1, 1), (2, 1, 1)),
            BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleNoDownsampleXYMultiStep(SpMiddleNoDownsampleXY):
    """
    No gradients!
    """
    def __init__(self, *args, **kwargs):
        super(SpMiddleNoDownsampleXYMultiStep, self).__init__(*args, **kwargs)

    def forward(self, voxel_features, coors, batch_size):
        self.eval()
        with torch.no_grad():
            t = len(voxel_features)
            output = []
            for i in range(t):
                voxel_features_i = voxel_features[i]
                coors_i = coors[i]
                coors_i = coors_i.int()
                ret = spconv.SparseConvTensor(voxel_features_i, coors_i, self.sparse_shape, batch_size)
                ret = self.middle_conv(ret)
                ret = ret.dense()
                N, C, D, H, W = ret.shape
                ret = ret.view(N, C * D, H, W)
                output.append(ret.detach())
            return output


class SpMiddleNoDownsampleXYNoExpand(nn.Module):
    """
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self,
                 output_shape,
                 num_input_features):
        super(SpMiddleNoDownsampleXYNoExpand, self).__init__()

        BatchNorm1d = functools.partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        SubMConv3d = functools.partial(spconv.SubMConv3d, bias=False)

        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        self.middle_conv = spconv.SparseSequential(
            SubMConv3d(num_input_features, 32, 3, indice_key="subm0"),
            BatchNorm1d(32),
            nn.ReLU(),
            SubMConv3d(32, 64, 3, indice_key="subm0"),
            BatchNorm1d(64),
            nn.ReLU(),
            # SubMConv3d(32, 64, 3, (2, 1, 1), padding=[1, 1, 1]),
            spconv.SparseMaxPool3d(3, (2, 1, 1), padding=[1, 1, 1]),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm1"),
            BatchNorm1d(64),
            nn.ReLU(),
            # SubMConv3d(64, 64, 3, (2, 1, 1), padding=[0, 1, 1]),
            spconv.SparseMaxPool3d(3, (2, 1, 1), padding=[0, 1, 1]),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            SubMConv3d(64, 64, 3, indice_key="subm2"),
            BatchNorm1d(64),
            nn.ReLU(),
            # SubMConv3d(64, 64, (3, 1, 1), (2, 1, 1)),
            spconv.SparseMaxPool3d((3, 1, 1), (2, 1, 1)),
            BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        ret = self.middle_conv(ret)
        ret = ret.dense()
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class SpMiddleNoDownsampleXYNoExpandMultiStep(SpMiddleNoDownsampleXYNoExpand):
    """
    No gradients!
    """
    def __init__(self, *args, **kwargs):
        super(SpMiddleNoDownsampleXYNoExpandMultiStep, self).__init__(*args, **kwargs)

    def forward(self, voxel_features, coors, batch_size):
        self.eval()
        with torch.no_grad():
            t = len(voxel_features)
            output = []
            for i in range(t):
                voxel_features_i = voxel_features[i]
                coors_i = coors[i]
                coors_i = coors_i.int()
                ret = spconv.SparseConvTensor(voxel_features_i, coors_i, self.sparse_shape, batch_size)
                ret = self.middle_conv(ret)
                ret = ret.dense()
                N, C, D, H, W = ret.shape
                ret = ret.view(N, C * D, H, W)
                output.append(ret.detach())
            return output


class MiddleNoDownsampleXY(nn.Module):
    """
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self, output_shape, num_input_features):
        super(MiddleNoDownsampleXY, self).__init__()
        Conv3d = functools.partial(nn.Conv3d, bias=True)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        self.sparse_shape = sparse_shape
        self.voxel_output_shape = output_shape
        self.middle_conv = nn.Sequential(
            Conv3d(num_input_features, 32, 3, padding=1),
            nn.ReLU(),
            Conv3d(32, 64, 3, (2, 1, 1), padding=1),  # Downsample z
            nn.ReLU(),
            Conv3d(64, 64, 3, stride=(2, 1, 1), padding=[0, 1, 1]),  # Downsample z
            nn.ReLU(),
            Conv3d(64, 64, kernel_size=(3, 1, 1), stride=(2, 1, 1)),  # Downsample z
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        inputs = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size).dense()
        ret = self.middle_conv(inputs)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class InpaintingFCHardnetRecurrentBase(object):
    def __init__(self,
                 aggregation_type='pre',
                 gru_input_size=(256, 256),
                 gru_input_dim=448,
                 gru_hidden_dims=[448],
                 gru_cell_type='standard',
                 noisy_pose=False, **kwargs):
        super(InpaintingFCHardnetRecurrentBase, self).__init__(**kwargs)

        assert aggregation_type in ['pre', 'post', 'none'], aggregation_type
        self.aggregation_type = aggregation_type

        if aggregation_type != 'none':
            ### Amirreza: GRU parameters are hardcoded for now
            self.gru = convgru.ConvGRU(input_size=gru_input_size,
                                       input_dim=gru_input_dim,
                                       hidden_dim=gru_hidden_dims,
                                       kernel_size=(3, 3),
                                       num_layers=len(gru_hidden_dims),
                                       dtype=torch.cuda.FloatTensor,
                                       batch_first=True,
                                       bias=True,
                                       return_all_layers=True,
                                       noisy_pose=noisy_pose,
                                       cell_type=gru_cell_type)

            def get_poses(input_pose):
                # convert to matrix
                mat = torch.zeros(input_pose.shape[0], # batch_size
                                  input_pose.shape[1], # t
                                  3, 3, dtype=input_pose.dtype,
                                  device=input_pose.device)

                mat[:, :, 0] = input_pose[:, :, :3]
                mat[:, :, 1] = input_pose[:, :, 3:6]
                mat[:, :, 2, 2] = 1.0

                # We are using two GRU cells with the same poses
                return mat[:, :, None]

            self.get_poses = get_poses

    def forward(self, x, seq_start=None, input_pose=None):
        n, c, h, w = x[0].shape
        t = len(x)

        if isinstance(x, list):
            x = torch.cat(x, dim=0)
        elif isinstance(x, torch.Tensor):
            x = x.view((-1,) + x.size()[2:])  # Fuse dim 0 and 1

        if self.aggregation_type != 'none':
            if seq_start is None:
                self.hidden_state = None
            else:
                # sanity check: only the first index can be True
                assert(torch.any(seq_start[1:]) == False)

                if seq_start[0]:  # start of a new sequence
                    self.hidden_state = None
        if self.aggregation_type == 'pre':
            layer_output_list, last_state_list = self.gru(x[None],
                                                          self.get_poses(input_pose[None]),
                                                          hidden_state=self.hidden_state)
            x = layer_output_list[-1].squeeze(0)

        out = self.fchardnet(x)

        if self.aggregation_type == 'post':
            layer_output_list, last_state_list = self.gru(out[None],
                                                          self.get_poses(input_pose[None]),
                                                          hidden_state=self.hidden_state)
            out = layer_output_list[-1].squeeze(0)

        if self.aggregation_type != 'none':
            self.hidden_state = []
            for state in last_state_list:
                dstate = state[0].detach()
                dstate.requires_grad = True
                self.hidden_state.append(dstate)

        num_class = out.shape[1]
        out = out.reshape((t, n, num_class, h, w))
        ret_dict = {
            "bev_preds": out,
        }
        return ret_dict


class InpaintingFCHardNetSkip1024(nn.Module):
    def __init__(self,
                 num_class=2,
                 num_input_features=128):
        super(InpaintingFCHardNetSkip1024, self).__init__()
        self.fchardnet = fchardnet.HardNet1024Skip(num_input_features, num_class)

    def forward(self, x, *args, **kwargs):
        out = self.fchardnet(x)
        ret_dict = {
            "bev_preds": out,
        }
        return ret_dict


class InpaintingFCHardNetSkipGRU512(InpaintingFCHardnetRecurrentBase, InpaintingFCHardNetSkip1024):
    pass
