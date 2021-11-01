import functools
import MinkowskiEngine as ME
import numpy as np
import spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
import fchardnet
import convgru
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18


class VoxelFeatureExtractorV3(nn.Module):
    def __init__(self):
        super(VoxelFeatureExtractorV3, self).__init__()

    def forward(self, features, num_voxels):
        # features: [concated_num_points, num_voxel_size, n_dim]
        # num_voxels: [concated_num_points]
        points_mean = features.sum(
            dim=1, keepdim=False) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()


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


class MiddleMEBase(object):
    def _prune(self, a, output_shape):
        """
        Currently this method doesn't consider the tensor stride of @param a, so
        you must make sure that @param output_shape considers tensor stride!
        For example, if you maxpool the z dimension, the dense output_shape will
        be halved, but the tensor_stride of @param a also increases from 1 to 2.
        Because of this, you need to properly adjust output_shape to make sure
        this doesn't prune useful data.

        Args:
            a: a MinkowskiEngine sparse tensor
            output_shape: C x D x H x W

        Returns:
            Tensor with out of range points pruned. This is necessary before
            converting the tensor to the dense version.
        """
        co = a.C
        mask = (co[:, 1] >= 0) & (co[:, 1] < output_shape[1]) & \
               (co[:, 2] >= 0) & (co[:, 2] < output_shape[2]) & \
               (co[:, 3] >= 0) & (co[:, 3] < output_shape[3])
        prune = ME.MinkowskiPruning()
        return prune(a, mask)


class MiddleMENoDownsampleXY(MiddleMEBase, nn.Module):
    """
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self,
                 output_shape,
                 num_input_features,
                 expand_coords=False):
        super(MiddleMENoDownsampleXY, self).__init__()
        self.output_shape = tuple(output_shape)
        self.convs = nn.Sequential(
            ME.MinkowskiConvolution(num_input_features, 32, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(32, 32, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(32, 64, kernel_size=3, stride=(2, 1, 1), dimension=3,
                                    expand_coordinates=expand_coords),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(64, 64, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(64, 64, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(64, 64, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(64, 64, kernel_size=3, stride=(2, 1, 1), dimension=3,
                                    expand_coordinates=expand_coords),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(64, 64, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(64, 64, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(64, 64, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(64, 64, kernel_size=(3, 1, 1), stride=(2, 1, 1), dimension=3,
                                    expand_coordinates=expand_coords),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        inputs = ME.SparseTensor(features=voxel_features, coordinates=coors)
        out = self.convs(inputs)
        out = self._prune(out, self.output_shape)
        ret, min_coord, stride = out.dense(
            shape=torch.Size((batch_size,) + self.output_shape),
            min_coordinate=torch.zeros(3, dtype=torch.int32, device='cpu'))
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class MiddleMENoDownsampleXYMaxPool(MiddleMEBase, nn.Module):
    """
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self,
                 output_shape,
                 num_input_features):
        super(MiddleMENoDownsampleXYMaxPool, self).__init__()
        self.output_shape = tuple(output_shape)
        self.convs = nn.Sequential(
            ME.MinkowskiConvolution(num_input_features, 32, kernel_size=3, stride=1, dimension=3),
            # ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(32, 64, kernel_size=3, stride=1, dimension=3),
            # ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),

            ME.MinkowskiMaxPooling(3, stride=(2, 1, 1), dimension=3),
            # ME.MinkowskiBatchNorm(64),
            # ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(64, 64, kernel_size=3, stride=1, dimension=3),
            # ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(64, 64, kernel_size=3, stride=1, dimension=3),
            # ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(64, 64, kernel_size=3, stride=1, dimension=3),
            # ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),

            ME.MinkowskiMaxPooling(3, stride=(2, 1, 1), dimension=3),
            # ME.MinkowskiBatchNorm(64),
            # ME.MinkowskiReLU(),

            ME.MinkowskiConvolution(64, 64, kernel_size=3, stride=1, dimension=3),
            # ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(64, 64, kernel_size=3, stride=1, dimension=3),
            # ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(64, 64, kernel_size=3, stride=1, dimension=3),
            # ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(),

            ME.MinkowskiMaxPooling((3, 1, 1), stride=(2, 1, 1), dimension=3),
            # ME.MinkowskiBatchNorm(64),
            # ME.MinkowskiReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        inputs = ME.SparseTensor(features=voxel_features, coordinates=coors)
        out = self.convs(inputs)
        # out = self._prune(out, self.output_shape)
        ret, min_coord, stride = out.dense(
            shape=torch.Size((batch_size,) + self.output_shape),
            min_coordinate=torch.zeros(3, dtype=torch.int32, device='cpu'))
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret



class MiddleMENoDownsampleXYMaxPoolV2(MiddleMEBase, nn.Module):
    """
    Always put BatchNorm before Conv.
    Maxpool kernel size same as stride.
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self,
                 output_shape,
                 num_input_features,
                 batch_norm=True,
                 bottleneck_features=None):
        super(MiddleMENoDownsampleXYMaxPoolV2, self).__init__()
        self.output_shape = tuple(output_shape)

        if batch_norm:
            BatchNorm = ME.MinkowskiBatchNorm
            bias = False
        else:
            BatchNorm = nn.Identity
            bias = True

        Conv = functools.partial(ME.MinkowskiConvolution, bias=bias, dimension=3)

        if bottleneck_features is not None:
            out_ch = self.output_shape[0] * self.output_shape[1]
            self.bottleneck = nn.Conv2d(out_ch, bottleneck_features, 1, 1)
        else:
            self.bottleneck = None

        self.convs = nn.Sequential(
            Conv(num_input_features, 32, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
            BatchNorm(32),

            Conv(32, 64, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
            ME.MinkowskiMaxPooling((2, 1, 1), stride=(2, 1, 1), dimension=3),
            BatchNorm(64),

            Conv(64, 64, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
            BatchNorm(64),
            Conv(64, 64, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
            BatchNorm(64),
            Conv(64, 64, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
            ME.MinkowskiMaxPooling((2, 1, 1), stride=(2, 1, 1), dimension=3),
            BatchNorm(64),

            Conv(64, 64, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
            BatchNorm(64),
            Conv(64, 64, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
            BatchNorm(64),
            Conv(64, 64, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
            ME.MinkowskiMaxPooling((2, 1, 1), stride=(2, 1, 1), dimension=3),
            BatchNorm(64),
        )

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        inputs = ME.SparseTensor(features=voxel_features, coordinates=coors)
        out = self.convs(inputs)
        ret, min_coord, stride = out.dense(
            shape=torch.Size((batch_size,) + self.output_shape),
            min_coordinate=torch.zeros(3, dtype=torch.int32, device='cpu'))
        N, C, D, H, W = ret.shape
        out = ret.view(N, C * D, H, W)
        if self.bottleneck is not None:
            out = self.bottleneck(out)
        return out


class MiddleMENoDownsampleXYResMaxPool(MiddleMEBase, nn.Module):
    """
    Only downsample z. Do not downsample X and Y.
    """
    def __init__(self,
                 output_shape,
                 num_input_features,
                 batch_norm=True):
        super(MiddleMENoDownsampleXYResMaxPool, self).__init__()
        self.output_shape = tuple(output_shape)

        if batch_norm:
            BatchNorm = ME.MinkowskiBatchNorm
            bias = False
        else:
            BatchNorm = nn.Identity
            bias = True

        Conv = functools.partial(ME.MinkowskiConvolution, bias=bias, dimension=3)

        self.block1 = nn.Sequential(
            Conv(num_input_features, 32, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
            BatchNorm(32),

            Conv(32, 64, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
        )

        self.pool1 = nn.Sequential(
            ME.MinkowskiMaxPooling((2, 1, 1), stride=(2, 1, 1), dimension=3),
            BatchNorm(64),
        )

        self.block2 = nn.Sequential(
            Conv(64, 64, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
            BatchNorm(64),
            Conv(64, 64, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
            BatchNorm(64),
            Conv(64, 64, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
        )

        self.pool2 = nn.Sequential(
            ME.MinkowskiMaxPooling((2, 1, 1), stride=(2, 1, 1), dimension=3),
            BatchNorm(64),
        )

        self.block3 = nn.Sequential(
            Conv(64, 64, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
            BatchNorm(64),
            Conv(64, 64, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
            BatchNorm(64),
            Conv(64, 64, kernel_size=3, stride=1),
            ME.MinkowskiReLU(),
        )

        self.pool3 = nn.Sequential(
            ME.MinkowskiMaxPooling((2, 1, 1), stride=(2, 1, 1), dimension=3),
            BatchNorm(64),
        )

    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        inputs = ME.SparseTensor(features=voxel_features, coordinates=coors)

        out = self.block1(inputs)
        out = self.pool1(out)
        out = self.block2(out) + out
        out = self.pool2(out)
        out = self.block3(out) + out
        out = self.pool3(out)

        ret, min_coord, stride = out.dense(
            shape=torch.Size((batch_size,) + self.output_shape),
            min_coordinate=torch.zeros(3, dtype=torch.int32, device='cpu'))
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret

class MergeUnit(nn.Module):
    def __init__(self,
            input_channels,
            rnn_input_channels=None,
            rnn_config=None,
            costmap_pose_name=None):
        super(MergeUnit, self).__init__()

        if rnn_input_channels is None:
            self.pre_rnn_conv = None
            rnn_input_channels = input_channels
        else:
            self.pre_rnn_conv = fchardnet.ConvLayer(input_channels,
                                                    rnn_input_channels,
                                                    kernel=1,
                                                    bn=True)

        self.costmap_pose_name = costmap_pose_name
        if rnn_config is None:
            self.rnn = None
        else:
            self.groups = rnn_config.get('groups', 1)
            hidden_dims = rnn_config['hidden_dims']

            if rnn_input_channels % self.groups:
                raise Exception(f'RNN input channels {rnn_input_channels}'
                                 ' is not divisible by groups!')
            if any([d % self.groups for d in hidden_dims]):
                raise Exception(f'Not all the hidden_dims are divisible by groups!')


            rnn_input_channels //= self.groups
            hidden_dims = [h//self.groups for h in hidden_dims]


            self.rnn = convgru.ConvGRU(input_size=rnn_config['input_size'],
                                       input_dim=rnn_input_channels,
                                       hidden_dim=hidden_dims,
                                       kernel_size=rnn_config.get('kernel_size', (3,3)),
                                       num_layers=len(hidden_dims),
                                       dtype=torch.cuda.FloatTensor,
                                       batch_first=True,
                                       bias=True,
                                       return_all_layers=True,
                                       noisy_pose=rnn_config.get('noisy_pose', False),
                                       cell_type=rnn_config.get('cell_type', 'standard'))

    def forward(self, x, t=1, bos=None, pose=None):
        if self.pre_rnn_conv is not None:
            x = self.pre_rnn_conv(x)

        if self.rnn is not None:
            assert(bos is not None and pose is not None)

            ### reshape (bt, c, h, w) --> (b, t, c, h, w)
            bt, c, h, w = x.shape
            b = bt//t
            bos = bos.reshape(b, t)
            pose = pose.reshape(b, t, 3, 3)

            if self.groups > 1:
                bg = b * self.groups

                # move groups to batch
                assert(c % self.groups == 0)
                x = x.reshape(b, t, self.groups, c//self.groups, h, w)
                # t <-> self.groups
                x = x.transpose(1, 2)
                x = x.reshape(bg, t, c//self.groups, h , w)

                bos = bos.repeat(self.groups, 1)
                pose = pose.repeat(self.groups, 1 , 1, 1)
            else:
                x = x.reshape(b, t, c, h, w)

            # We simplify things assuming that bos[:, t] is *all* True or False
            assert(torch.all(torch.all(bos, axis=0) ^ torch.all(~bos, axis=0)))

            # Also we furthur simplify things assuming that only bos[:, 0] can be true :)
            assert(torch.any(bos[0, 1:]) == False), ('Only the first element in the chunk '
                                                     'can be begging of the sequence. '
                                                     'Make sure "miniseq_sampler.len" is '
                                                     'divisible by "miniseq_sampler.chunk_len".')

            if bos[0, 0]:  # start of a new sequence
                self.hidden_state = None

            ## We keep the translation and resolution the same for
            # all the rnn layers so we can use same pose for all
            # layers.
            pose = pose[:, :, None].expand(-1, -1, self.rnn.num_layers, -1, -1)
            layer_output_list, last_state_list = self.rnn(x, pose,
                                                          hidden_state=self.hidden_state)


            self.hidden_state = []
            for state in last_state_list:
                assert(len(state) == 1)
                dstate = state[0].detach()
                dstate.requires_grad = True
                self.hidden_state.append(dstate)

            x = layer_output_list[-1]

            if self.groups > 1:
                x = x.reshape(b, self.groups, t, c//self.groups, h, w)
                # self.groups <-> t
                x = x.transpose(1, 2)

            x = x.reshape(bt, c, h, w)

        return x

class InpaintingFCHardNetSkip1024(nn.Module):
    def __init__(self,
                 num_class=2,
                 num_input_features=128,
                 guide=[],
                 guide_num_channels=0):

        super(InpaintingFCHardNetSkip1024, self).__init__()
        self.fchardnet = fchardnet.HardNet1024Skip(num_input_features, num_class,
                guide=guide, guide_num_channels=guide_num_channels)

    def forward(self, x, *args, **kwargs):
        out = self.fchardnet(x, *args, **kwargs)
        ret_dict = {
            "bev_preds": out,
        }
        return ret_dict


class InpaintingResNet18(nn.Module):
    def __init__(self, num_input_features, num_class):
        super(InpaintingResNet18, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(
            num_input_features, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_class, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return dict(bev_preds=x)


class Up(nn.Module):
    def __init__(self, inC, outC, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=False
        )

        self.conv = nn.Sequential(
            nn.Conv2d(inC, outC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU(inplace=True),
            nn.Conv2d(outC, outC, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outC),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)
