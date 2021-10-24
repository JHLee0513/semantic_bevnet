import numpy as np
import torch
import yaml

from spconv.utils import VoxelGenerator
from bevnet import networks
from bevnet.utils import pprint_dict


def make_nets(config, device):
    ret = {}
    for net_name, spec in config.items():
        net_class = getattr(networks, spec['class'])
        net_args = spec.get('net_kwargs', {})
        net = net_class(**net_args).to(device)
        ret[net_name] = net
    return ret


class BEVNetBase(object):
    from functools import partial
    import spconv
    print('patch spconv to increase the allowable z range. This will not affect the point cloud range.')
    spconv.utils.points_to_voxel = partial(spconv.utils.points_to_voxel,
                                           height_threshold=-4.0,
                                           height_high_threshold=5.0)

    def __init__(self, weights_file, device='cuda'):
        self.weights_file = weights_file
        if weights_file:
            self._load(weights_file, device)

        self.device = device
        self.h = None

    def _load(self, weights_file, device):
        state_dict = torch.load(weights_file, map_location='cpu')
        print('loaded %s' % weights_file)
        g = state_dict.get('global_args', {})
        print('global args:')
        print(pprint_dict(g))

        self.g = g
        self.weights_file = weights_file

        if isinstance(g.model_config, dict):
            nets = make_nets(g.model_config, device)
        else:
            nets = make_nets(yaml.load(open(g.model_config).read(),
                                       Loader=yaml.SafeLoader), device)

        for name, net in nets.items():
            net.load_state_dict(state_dict['nets'][name])
            net.train(False)
        self.nets = nets

        self.voxelizer = self._make_voxelizer(self.g.voxelizer)

    def _make_voxelizer(self, cfg):
        return VoxelGenerator(
            voxel_size=list(cfg['voxel_size']),
            point_cloud_range=list(cfg['point_cloud_range']),
            max_num_points=cfg['max_number_of_points_per_voxel'],
            full_mean=cfg['full_mean'],
            max_voxels=cfg['max_voxels'])

    def _as_tensor(self, data, dtype=None):
        return torch.as_tensor(data, dtype=dtype).to(device=self.device, non_blocking=True)


class BEVNetSingle(BEVNetBase):
    def predict(self, points):
        with torch.no_grad():
            points_with_idx = np.concatenate([
                points, np.arange(len(points))[:, None].astype(points.dtype)], axis=-1)
            voxels, coords, num_points = self.voxelizer.generate(points_with_idx, max_voxels=90000)
            voxel_point_idxs = voxels[:, :, -1].astype(np.int32)  # num_voxels x max_num_points_per_voxel
            voxels = voxels[:, :, :-1]

            # Insert the batch dim
            coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)

            nets = self.nets

            voxels_th = self._as_tensor(voxels)
            num_points_th = self._as_tensor(num_points)
            coords_th = self._as_tensor(coords)

            voxel_features = nets['VoxelFeatureEncoder'](voxels_th, num_points_th)
            features = nets['MiddleSparseEncoder'](voxel_features, coords_th, 1)
            preds = nets['BEVClassifier'](features)['bev_preds']
            return preds


class BEVNetRecurrent(BEVNetBase):
    def __init__(self, *args, **kwargs):
        super(BEVNetRecurrent, self).__init__(*args, **kwargs)
        self.seq_start = None
        self.reset()

    def reset(self):
        self.seq_start = True

    def predict(self, points, pose):
        with torch.no_grad():
            points_with_idx = np.concatenate([
                points, np.arange(len(points))[:, None].astype(points.dtype)], axis=-1)
            voxels, coords, num_points = self.voxelizer.generate(points_with_idx, max_voxels=90000)
            voxel_point_idxs = voxels[:, :, -1].astype(np.int32)  # num_voxels x max_num_points_per_voxel
            voxels = voxels[:, :, :-1]

            # Insert the batch dim
            coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)

            nets = self.nets

            voxels_th = self._as_tensor(voxels)
            num_points_th = self._as_tensor(num_points)
            coords_th = self._as_tensor(coords)
            pose_th = self._as_tensor(pose)

            voxel_features = nets['VoxelFeatureEncoder']([voxels_th], [num_points_th])
            # print(voxel_features.size(), coords_th.size()); exit(0)
            features = nets['MiddleSparseEncoder'](voxel_features, [coords_th], 1)
            preds = nets['BEVClassifier'](features,
                                          seq_start=torch.tensor([self.seq_start]),
                                          input_pose=pose_th[None])['bev_preds']
            preds = preds.squeeze(1)

            self.seq_start = False

            return preds


if __name__ == '__main__':
    model = BEVNetSingle('../experiments/kitti4_100/single/include_unknown/default-logs/model.pth.4')
    scan = np.fromfile('../data/semantic_kitti_4class_100x100/sequences/valid/velodyne/00000.bin', dtype=np.float32)
    scan = scan.reshape(-1, 4)
    pred = model.predict(scan)
    print(pred.size())
