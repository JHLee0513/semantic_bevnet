import bisect
import os
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils import data
from bevnet.utils import file_utils
from easydict import EasyDict
from bevnet.dataset.augmentation import PointCloudAugmentation

def patch_spconv():
    # overwrite the default threshold values in spconv.utils.points_to_voxel
    from functools import partial
    import spconv
    print('patch spconv to increase the allowable z range. ' +
          'This will not affect the point cloud range.')
    spconv.utils.points_to_voxel = partial(spconv.utils.points_to_voxel,
                                           height_threshold=-4.0,
                                           height_high_threshold=5.0)


patch_spconv()


class Sequence(object):
    def __init__(self, data_format, path, name):
        self.name = name

        self.len = None
        self.poses = None
        pc_format = data_format['point_cloud']
        if 'poses' in pc_format:
            poses_fn = pc_format['poses']
            poses_fn = os.path.join(path, poses_fn)
            self.poses = torch.from_numpy(
                    file_utils.parse_poses(poses_fn).astype(np.float64))
            self.len = len(self.poses)


        depth_pcs, output_pcs, self.pcs, self.pc_fns = [], [], [], []

        self.label_fns = None

        self.costmap_proj = dict()

        ## Add label projection
        label_range = [pc_format['range'][i] for i in [0, 1, 3, 4]]
        label_map_shape = dict(range=label_range,
                               pixel_size=pc_format['voxel_size'][:2])
        self.costmap_proj['label_pose'] = self.point2map_projections(
                label_map_shape)
        ##

        for output in data_format['outputs']:
            t = output['type']
            if 'dir' in output:
                name = output['dir']
                fn = os.path.join(path, name)
            if t == 'lidar':
                output_pcs.append(name)
                self.pcs.append(name)
                self.pc_fns.append(self._get_data(fn, '.bin'))
            elif t == 'bev_label':
                self.label_fns = self._get_data(fn, '.png')
            elif t == 'costmap_pose':
                proj_name = output['name']
                assert(proj_name != 'label_pose'), 'label_pose is reserved.'
                self.costmap_proj[output['name']] = self.point2map_projections(
                       output['map_shape'])
            else:
                raise Exception(f'data type {t} is not supported.')


        self.depth_pc_idx = [self.pcs.index(n) for n in depth_pcs]
        self.output_pc_idx = [self.pcs.index(n) for n in output_pcs]

    def point2map_projections(self, map_shape):
        minx, miny = map_shape['range'][:2]
        gridw, gridh = map_shape['pixel_size']

        # scale
        inv_proj = np.diag([gridw, gridh, 1.0, 1.0]).astype(np.float64)

        # shift
        inv_proj[0, 3] = minx
        inv_proj[1, 3] = miny

        inv_proj = torch.from_numpy(inv_proj)

        return torch.linalg.inv(inv_proj), inv_proj

    def _get_data(self, dir, suffix):
        new_data = file_utils.listdir(dir, suffix)
        if self.len is None:
            self.len = len(new_data)
        elif self.len != len(new_data):
            raise Exception('Data size mismatch detected.')
        return new_data

    @property
    def has_output_pc(self):
        return bool(self.output_pc_idx)

    def output_pc(self, index):
        return self._pointclouds(index, self.output_pc_idx, merge=True)

    def label(self, index):
        img = np.array(Image.open(self.label_fns[index]))
        img = torch.from_numpy(img.astype(np.uint8))
        return img

    def _pointclouds(self, index, pc_ids, merge=False):
        pcs = []
        for i in pc_ids:
            pc = np.fromfile(self.pc_fns[i][index], dtype=np.float32
                        ).reshape(-1, 4)
            pc = torch.from_numpy(pc)
            pcs.append(pc)

        if merge:
            pcs = torch.cat(pcs)

        return pcs

    def __len__(self):
        return self.len

    # self.costmap_proj

class BEVDataset(data.Dataset):
    def __init__(self, path, data_format, reader_config, voxel_generator):
        ## Load sequences
        self.seqs, seq_lens = [], []
        for seq_name in reader_config['sequences']:
            seq_dir = os.path.join(path, seq_name)
            seq = Sequence(data_format, seq_dir, seq_name)
            self.seqs.append(seq)
            seq_lens.append(len(seq))

        self.seq_len_cumsum = np.cumsum(seq_lens) 
        self.voxel_generator = voxel_generator
        self.reader_config = reader_config
        self.data_format = data_format
        self.root = path

        if 'pc_augmentation' in reader_config:
            self.pc_augmentation = PointCloudAugmentation(
                    reader_config['pc_augmentation'])
        else:
            self.pc_augmentation = None

    @staticmethod
    def init(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return
        np.random.seed(worker_info.seed & 0xffffffff)


    def _locate_sample(self, idx):
        seq_idx = bisect.bisect_right(self.seq_len_cumsum, idx)
        if seq_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.seq_len_cumsum[seq_idx - 1]
        return self.seqs[seq_idx], sample_idx


    def voxelize(self, points):
        points = points.numpy()
        # Add an additional channel storing the point indices
        points_with_idx = np.concatenate([
            points, np.arange(len(points))[:, None].astype(points.dtype)], axis=-1)
        voxels, coords, num_points = self.voxel_generator.generate(
                points_with_idx, max_voxels=90000)
        # num_voxels x max_num_points_per_voxel
        voxel_point_idxs = voxels[:, :, -1].astype(np.int32)
        voxels = voxels[:, :, :-1]

        return (torch.from_numpy(voxels),
                torch.from_numpy(coords),
                torch.from_numpy(num_points))

    def __getitem__(self, index):
        seq, sample_idx = self._locate_sample(index)
        return self.get_frame_from_seq(seq, sample_idx)

    def get_frame_from_seq(self, seq, sample_idx):
        if self.pc_augmentation is not None:
            self.pc_augmentation.renew_transformation()
        ret = dict(info=dict(seq=seq.name, frame=sample_idx, len=len(seq)))
        label = seq.label(sample_idx)

        if self.pc_augmentation is not None:
            projs = seq.costmap_proj['label_pose']
            label, valid = self.pc_augmentation.transform_map(label, *projs)
            label[~valid] = 255
        ret['label'] = label

        if seq.has_output_pc:
            points = seq.output_pc(sample_idx)

            if self.pc_augmentation is not None:
                points = self.pc_augmentation.transform(points)
                self.pc_augmentation.renew_others(len(points))
                points = self.pc_augmentation.apply_others(points)

            voxels, coords, num_points = self.voxelize(points)
            ret['points'] = points.numpy()
            ret['voxels'] = voxels
            ret['coords'] = coords
            ret['num_points'] = num_points

        if seq.poses is not None:
            pose = seq.poses[sample_idx]
            # update pose with augmentation
            if self.pc_augmentation is not None:
                pose = self.pc_augmentation.correct_pose(pose)

            # Uses 3d pose to create costmap poses
            # for all the elements in seq.costmap_proj
            for name, (proj2map, inv_proj2map) in seq.costmap_proj.items():
                costmap_pose = proj2map @ pose @ inv_proj2map
                # remove z
                costmap_pose = costmap_pose[:, (0,1,3)][(0, 1, 3), :]
                ret[name] = costmap_pose

        return ret

    def __len__(self):
        return self.seq_len_cumsum[-1]

class BEVMiniSeqDataset(BEVDataset):

    def __init__(self, path, data_format, reader_config, voxel_generator, miniseq_len=None, stride=None):
        '''
            Similar to BEVDataset but it reads a mini-sequence instead of a single frame.

            miniseq_len: int. Lenght of the mini-sequences.
            strid: int or list of int. Frame stride of the mini-sequences.
        '''
        super(BEVMiniSeqDataset, self).__init__(path, data_format, reader_config, voxel_generator)

        self.miniseq_len = miniseq_len or 1

        if stride is None:
            stride = [1]
        elif isinstance(stride, int):
            stride = [stride]
        elif isinstance(stride, list):
            pass
        else:
            raise Exception(f'Stride can be int or list not {type(stride)}.')

        self.stride = stride

    def __getitem__(self, index):
        if self.miniseq_len is None: # return a single sample
            return [super(BEVMiniSeqDataset, self).__getitem__(index)]

        seq, sample_idx = self._locate_sample(index)

        stride = np.random.choice(self.stride)
        if stride == 0:
            indices = [sample_idx] * self.miniseq_len
        else:
            end = sample_idx + stride * self.miniseq_len
            indices = [min(i, len(seq) - 1) for i in range(sample_idx, end, stride)]

        return [self.get_frame_from_seq(seq, i) for i in indices]

class BEVDataLoader(data.DataLoader):

    def __init__(self, dataset,
                batch_size,
                num_workers,
                is_train,
                chunk_len=None):

        '''
            Reads a batch of mini-seuqences from the dataset. Each time __next__
            is called chunk_len frames from each mini-sequence is returned.

            It adds an additional tensor called bos that indicates the begging
            of a new sequence:

                1) If is_train is True, bos marks the first frame in each mini-sequence.
                2) If is_train is False, bos makrs the first frame in the original sequence.

            When is_train is False "batch_size", "chunk_len", and "dataset.miniseq_len" should
            all be set to 1. This is neccessary to read all the sequences sequentially.
        '''
        if not isinstance(dataset, BEVMiniSeqDataset):
            raise Exception('Dataset type is not BEVMiniSeqDataset.')

        super(BEVDataLoader, self).__init__(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=lambda data: list(zip(*data)),
                worker_init_fn=BEVDataset.init,
                drop_last=True,
                shuffle=is_train)


        self.chunk_len = chunk_len or 1

        self.is_train = is_train

        if not is_train:
            assert(batch_size == 1 and
                   self.chunk_len == 1 and
                   dataset.miniseq_len == 1)


        # We might read overlapping frames during training.
        # The stop_iter value is decided so the total number of
        # frames we read (overlapping or not) is equal to
        # the dataset size.

        # The original BEVDataLoader size is correct when
        # is_train == False
        if is_train:
            self.stop_iter = super(BEVDataLoader, self).__len__()
            self.stop_iter //= self.chunk_len


        # Meta information about the batch
        self.batch_meta_info = dict(chunk_len=torch.tensor(self.chunk_len),
                                    batch_size=torch.tensor(batch_size))

    def __len__(self):
        if self.is_train:
            return self.stop_iter
        else:
            return super(BEVDataLoader, self).__len__()

    def __iter__(self):
        self.dataset_iter = super(BEVDataLoader, self).__iter__()
        self.niter = 0
        self.dt = []

        # Begging of Sequence
        self.bos = []
        return self

    def __next__(self):
        if self.is_train and self.niter >= self.stop_iter:
            raise StopIteration

        ## make sure we have enough data
        while len(self.dt) < self.chunk_len:
            dt = next(self.dataset_iter)
            self.dt.extend(dt)

            ## populate bos
            if self.is_train:
                bos = [True] + [False] * (len(dt) - 1)
            else:
                # batch_size = 1, miniseq_len = 1
                assert(len(dt) == 1 and len(dt[0]) == 1)
                frame_id = dt[0][0]['info']['frame']
                seq_len = dt[0][0]['info']['len']
                bos = [frame_id == 0]

            self.bos.extend(bos)

        dt = self.dt[:self.chunk_len]
        bos = self.bos[:self.chunk_len]

        self.dt = self.dt[self.chunk_len:]
        self.bos = self.bos[self.chunk_len:]

        # dt is a list of list where the
        # outer list size is chunk_len and
        # the inner list size is batch_size.
        # We transpose that with zip
        dt = list(zip(*dt))

        flatten_dt = []
        for chunk in dt:
            for dp_bos, dp in zip(bos, chunk):
                dp['bos'] = torch.tensor(dp_bos)
                flatten_dt.append(dp)

        self.niter += 1
        return BEVDataLoader.collate_fn(flatten_dt,
                batch_meta_info=self.batch_meta_info)

    @staticmethod
    def collate_fn(data, batch_meta_info=None):
        out = dict()

        def make_batch(key):
            return [_[key] for _ in data]

        for key in data[0].keys():
            if key not in [
                    'voxels', 'coords', 'points', 'num_points',
                    'info'
                ]:
                try:
                    out[key] = torch.stack(make_batch(key))
                except:
                    print(key)
                    raise

        if 'coords' in data[0]:
            batch_coords = []
            pad = torch.nn.functional.pad

            for i in range(len(data)):
                # Add batch index
                batch_coords.append(pad(data[i]['coords'], (1, 0),
                                        mode='constant', value=i))
            out['coordinates'] = torch.cat(batch_coords)
            out['voxels'] = torch.cat(make_batch('voxels'))
            out['num_points'] = torch.cat(make_batch('num_points'))
            out['points'] = make_batch('points')
            out['info'] = make_batch('info')

        if batch_meta_info is not None:
            out.update(batch_meta_info)

        return out

