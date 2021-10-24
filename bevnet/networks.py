import functools
import MinkowskiEngine as ME
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


class MiddleMENoDownsampleXYMultiStep(MiddleMENoDownsampleXY):
    """
    No gradients!
    """
    def __init__(self, *args, **kwargs):
        super(MiddleMENoDownsampleXYMultiStep, self).__init__(*args, **kwargs)

    def forward(self, voxel_features, coors, batch_size):
        self.eval()
        with torch.no_grad():
            t = len(voxel_features)

            coors2 = []
            for i in range(t):
                coors2.append(coors[i][:, 1:].int())  # Remove the batch indices

            batch_coords, batch_features = ME.utils.sparse_collate(coors2, voxel_features, device='cuda')

            inputs = ME.SparseTensor(features=batch_features, coordinates=batch_coords,
                                     minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT)

            out = self.convs(inputs)
            out = self._prune(out, self.output_shape)

            ret, min_coord, stride = out.dense(
                shape=torch.Size((t,) + self.output_shape),
                min_coordinate=torch.zeros(3, dtype=torch.int32, device='cpu'))

            N, C, D, H, W = ret.shape
            ret = ret.view(N, 1, C * D, H, W).detach()
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


class MiddleMENoDownsampleXYMaxPoolMultiStep(MiddleMENoDownsampleXYMaxPool):
    """
    No gradients!
    """
    def __init__(self, bottleneck_features=None, **kwargs):
        super(MiddleMENoDownsampleXYMaxPoolMultiStep, self).__init__(**kwargs)
        if bottleneck_features is not None:
            out_ch = self.output_shape[0] * self.output_shape[1]
            self.bottleneck = nn.Conv2d(out_ch, bottleneck_features, 1, 1)
        else:
            self.bottleneck = None

    def forward(self, voxel_features, coors, batch_size):
        self.eval()
        with torch.no_grad():
            t = len(voxel_features)

            coors2 = []
            for i in range(t):
                coors2.append(coors[i][:, 1:].int())  # Remove the batch indices

            batch_coords, batch_features = ME.utils.sparse_collate(coors2, voxel_features, device='cuda')

            inputs = ME.SparseTensor(features=batch_features, coordinates=batch_coords,
                                     minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT)

            out = self.convs(inputs)
            out = self._prune(out, self.output_shape)

            ret, min_coord, stride = out.dense(
                shape=torch.Size((t,) + self.output_shape),
                min_coordinate=torch.zeros(3, dtype=torch.int32, device='cpu'))

            N, C, D, H, W = ret.shape
            out = ret.view(N, 1, C * D, H, W)

        if self.bottleneck is not None:
            out = self.bottleneck(out.view(N, C * D, H, W))
            out = out.view(N, 1, -1, H, W)

        return out


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


class MiddleMENoDownsampleXYMaxPoolV2MultiStep(MiddleMENoDownsampleXYMaxPoolV2):
    def __init__(self, no_gradients=True, **kwargs):
        super(MiddleMENoDownsampleXYMaxPoolV2MultiStep, self).__init__(**kwargs)
        self.no_gradients = no_gradients

    def _forward_helper(self, voxel_features, coors, batch_size):
        assert batch_size == 1
        t = len(voxel_features)

        coors2 = []
        for i in range(t):
            coors2.append(coors[i][:, 1:].int())  # Remove the batch indices

        batch_coords, batch_features = ME.utils.sparse_collate(coors2, voxel_features, device='cuda')

        inputs = ME.SparseTensor(features=batch_features,
                                 coordinates=batch_coords,
                                 minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT)

        out = self.convs(inputs)
        ret, min_coord, stride = out.dense(
            shape=torch.Size((t,) + self.output_shape),
            min_coordinate=torch.zeros(3, dtype=torch.int32, device='cpu'))

        N, C, D, H, W = ret.shape
        return ret.view(N, 1, C * D, H, W)

    def forward(self, voxel_features, coors, batch_size):
        if self.no_gradients:
            self.eval()
            with torch.no_grad():
                return self._forward_helper(voxel_features, coors, batch_size)
        else:
            return self._forward_helper(voxel_features, coors, batch_size)


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
