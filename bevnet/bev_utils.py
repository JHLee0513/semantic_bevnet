from io import BytesIO
import bisect
import glob
import os

import cv2
import PIL
import lz4
import numpy as np
import torch
import transforms3d
import yaml
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils import data


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in sorted(filenames)
        if filename.endswith(suffix)
    ]


def aggregate_scans(scans, poses):
    """
    Aggregate all scans into the coordinate frame of scans[0].

    Args:
        scans: a list of Nx4 np arrays
        poses: a list of 12-element np arrays

    Returns:
        Nx4 np array of the aggregated scans
    """
    ref_m = poses[0].reshape((3, 4))
    ref_r = ref_m[:3, :3]
    ref_t = ref_m[:3, 3]

    points = [scans[0]]

    for i in range(1, len(poses)):
        m = poses[i].reshape((3, 4))
        r = m[:3, :3]
        t = m[:3, 3]
        r2 = ref_r.T @ r
        t2 = ref_r.T @ (t - ref_t)

        xyz = scans[i][:, :3]
        scan_copy = scans[i].copy()
        scan_copy[:, :3] = xyz @ r2.T + t2
        points.append(scan_copy)

    return np.concatenate(points)


class BEVLoader(data.Dataset):
    # overwrite the default threshold values in spconv.utils.points_to_voxel
    from functools import partial
    import spconv
    print('patch spconv to increase the allowable z range. This will not affect the point cloud range.')
    spconv.utils.points_to_voxel = partial(spconv.utils.points_to_voxel,
                                           height_threshold=-4.0,
                                           height_high_threshold=5.0)

    def __init__(
        self,
        config,
        dataset_path,
        split="train",
        voxel_generator=None,
        label_dir_name='bev_labels',
        augment=False,
    ):
        """__init__

        :param config:
        :param split:
        """
        self.root = os.path.join(dataset_path, split)
        self.split = split
        self.input_base = os.path.join(self.root, 'velodyne')
        self.annotations_base = os.path.join(self.root, label_dir_name)
        self.files_input = recursive_glob(rootdir=self.input_base, suffix=".bin")
        self.poses = np.loadtxt(os.path.join(self.root, 'poses.txt'))

        if 'use_image' in config.keys():
            self.image_Ks = np.loadtxt(os.path.join(self.root, 'cam2_K.txt')).astype(np.float32)
            self.lidar_cam_transforms = np.loadtxt(
                os.path.join(self.root, 'lidar_cam2_transform.txt')).astype(np.float32)

        self.voxel_generator = voxel_generator
        self.label_shape = None
        self.aug = augment
        self.config = config

    def __len__(self):
        """__len__"""
        return len(self.files_input)

    def set_label_shape(self, label_shape):
        assert(isinstance(label_shape, tuple) and len(label_shape) == 2)
        self.label_shape = label_shape

    def augment(self, points, label):
        """
        Randomly rotate and translate points and label.
        Note that currently this only works for kitti_19 datasets!
        Args:
            points:
            label:

        Returns:

        """
        points = points.copy()
        label = label.copy()

        # cv2.imshow('label before', visualize_label(label))

        rng = np.random

        angle = rng.uniform(-np.deg2rad(45), np.deg2rad(45))
        translation = rng.uniform(-5.0, 5.0, size=2).astype(np.float32)
        resolution = 0.2  # Resolution of the label image. TODO: remove hardcoded value

        R = transforms3d.euler.euler2mat(0.0, 0.0, angle)
        points[:, :3] = points[:, :3] @ R.T
        points[:, :2] += translation[None]

        h, w = label.shape
        origin = self.config['origin']
        if origin == 'center':
            rotation_center = (w // 2, h // 2)
        elif origin == 'center_left':
            rotation_center = (0, h // 2)
        else:
            raise RuntimeError('Unsupported vehicle origin.')

        # Rotate around the center left.
        # Note that the angle is negated.
        M = cv2.getRotationMatrix2D(rotation_center, -np.rad2deg(angle), 1.0)
        label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        tm = np.array([
            [1.0, 0.0, translation[0] / resolution],
            [0.0, 1.0, translation[1] / resolution]
        ])
        label = cv2.warpAffine(label, tm, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        # cv2.imshow('label after', visualize_label(label))
        # cv2.waitKey(0)

        if 'noisy_z' in self.config.keys():
            points[:, 2] += np.clip(rng.randn(len(points)) * 0.5, -0.3, 0.3)

        if 'noisy_remission' in self.config.keys():
            points[:, 3] += rng.randn(len(points)) * 0.1

        return points, label

    def map_class_labels(self, labels):
        # FIXME: hardcoded kitti19 learning map
        learning_map = {0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10,
                        48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19, 99: 0,
                        252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5}
        u, inv = np.unique(labels, return_inverse=True)
        return np.array([learning_map[x] for x in u])[inv].reshape(labels.shape)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files_input[index].rstrip()
        lbl_path = img_path.replace(self.input_base, self.annotations_base)[:-4] + ".png"
        points = np.fromfile(img_path, dtype=np.float32, count=-1).reshape([-1, 4])

        if 'use_segmentation_label' in self.config.keys():
            label_dir = os.path.join(self.root, 'cy3d_pred_labels')
            basename = '0' + os.path.basename(img_path)[:-4] + '.label'
            segmentation_label = np.fromfile(os.path.join(label_dir, basename), dtype=np.uint32)
            segmentation_label = self.map_class_labels(segmentation_label & 0xFFFF)  # 0 - 20
            assert len(segmentation_label) == len(points)
        else:
            segmentation_label = None

        if self.split == "train" and self.aug:
            points, point_idxs = drop_points(points, 0.8)
            if segmentation_label is not None:
                segmentation_label = segmentation_label[point_idxs]

        if 'use_image' in self.config.keys():
            img_dir = os.path.join(self.root, 'image_2')
            basename = os.path.basename(img_path)[:-4] + '.png'
            img = np.array(Image.open(os.path.join(img_dir, basename)))
            # Use BGR Format to be compatible with pretrained FCHardNet
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            height, width = img.shape[:2]
            inrange, pixel_idxs = compute_pixel_indices(
                points,
                self.lidar_cam_transforms[index].reshape(4, 4),
                self.image_Ks[index].reshape(3, 3),
                width, height, img)

            # 0 index is a special case which we always treat as out of range.
            # This is to make handling empty slots in a voxel easier.
            inrange[0] = False

            img = img.transpose((2, 0, 1)).astype(np.float32) / 255
            mean = np.array([0.406, 0.456, 0.485], np.float32)
            std = np.array([0.225, 0.224, 0.229], np.float32)
            img = (img - mean[:, None, None]) / std[:, None, None]
        else:
            # Dummy values
            img = np.array([0.0], np.float32)
            inrange = np.array([0.0], np.float32)
            pixel_idxs = np.array([0.0], np.float32)

        label = Image.open(lbl_path)
        if self.label_shape is not None:
            label = label.resize(self.label_shape, resample=Image.NEAREST)

        label = np.array(label, dtype=np.uint8)

        if self.split == "train" and self.aug:
            points, label = self.augment(points, label)

        # Add an additional channel storing the point indices
        points_with_idx = np.concatenate([
            points, np.arange(len(points))[:, None].astype(points.dtype)], axis=-1)
        voxels, coords, num_points = self.voxel_generator.generate(points_with_idx, max_voxels=90000)

        voxel_point_idxs = voxels[:, :, -1].astype(np.int32)  # num_voxels x max_num_points_per_voxel
        voxels = voxels[:, :, :-1]

        if 'use_image' in self.config.keys():
            voxel_point_pixel_idxs = pixel_idxs[voxel_point_idxs].astype(np.int64)
            # note that index == 0 will be treated as out of range
            voxel_point_pixel_inrange = inrange[voxel_point_idxs].astype(np.float32)
        else:
            # Dummy values
            voxel_point_pixel_idxs = np.array([0.0], np.float32)
            voxel_point_pixel_inrange = np.array([0.0], np.float32)

        if 'use_segmentation_label' in self.config.keys():
            segmentation_label[0] = 0  # Set the label of the first point as unknown
            voxel_point_segmentation = segmentation_label[voxel_point_idxs]
        else:
            # Dummy values
            voxel_point_segmentation = np.array([0.0], np.float32)

        return {
            'voxels': voxels,
            'coords': coords,
            'num_points': num_points,
            'label': label,
            'points': points,
            'img': img,
            'voxel_point_pixel_idxs': voxel_point_pixel_idxs,
            'voxel_point_pixel_inrange': voxel_point_pixel_inrange,
            'voxel_point_idxs': voxel_point_idxs,
            'voxel_point_segmentation': voxel_point_segmentation,
        }


class BEVLoaderV2(data.Dataset):
    # overwrite the default threshold values in spconv.utils.points_to_voxel
    from functools import partial
    import spconv
    print('patch spconv to increase the allowable z range. This will not affect the point cloud range.')
    spconv.utils.points_to_voxel = partial(spconv.utils.points_to_voxel,
                                           height_threshold=-4.0,
                                           height_high_threshold=5.0)

    def __init__(
        self,
        config,
        dataset_path,
        voxel_generator=None,
        n_buffer_scans=1,
        buffer_scan_stride=1,
    ):
        """__init__

        :param config:
        """
        self.root = dataset_path
        sequences = config['sequences']

        self.poses = dict()
        self.scan_files = dict()
        seq_lens = []
        for seq in sequences:
            self.poses[seq] = np.loadtxt(os.path.join(self.root, seq, 'poses.txt'))
            self.scan_files[seq] = sorted(glob.glob(os.path.join(self.root, seq, 'velodyne', '*.bin')))
            seq_lens.append(len(self.scan_files[seq]))
        self.seq_len_cumsum = np.cumsum(seq_lens)

        self.voxel_generator = voxel_generator
        self.label_shape = None
        self.config = config

        self.buffer_scans = n_buffer_scans
        self.buffer_scan_stride = buffer_scan_stride

    def __len__(self):
        """__len__"""
        return sum([len(_) for _ in self.scan_files.values()])

    def set_label_shape(self, label_shape):
        assert(isinstance(label_shape, tuple) and len(label_shape) == 2)
        self.label_shape = label_shape

    def augment(self, points, label):
        """
        Randomly rotate and translate points and label.
        Note that currently this only works for kitti_19 datasets!
        Args:
            points:
            label:

        Returns:

        """
        points = points.copy()
        label = label.copy()

        # cv2.imshow('label before', visualize_label(label))

        rng = np.random

        angle = rng.uniform(-np.deg2rad(45), np.deg2rad(45))
        translation = rng.uniform(-5.0, 5.0, size=2).astype(np.float32)
        resolution = 0.2  # Resolution of the label image. TODO: remove hardcoded value

        R = transforms3d.euler.euler2mat(0.0, 0.0, angle)
        points[:, :3] = points[:, :3] @ R.T
        points[:, :2] += translation[None]

        h, w = label.shape
        origin = self.config['origin']
        if origin == 'center':
            rotation_center = (w // 2, h // 2)
        elif origin == 'center_left':
            rotation_center = (0, h // 2)
        else:
            raise RuntimeError('Unsupported vehicle origin.')

        # Rotate around the center left.
        # Note that the angle is negated.
        M = cv2.getRotationMatrix2D(rotation_center, -np.rad2deg(angle), 1.0)
        label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        tm = np.array([
            [1.0, 0.0, translation[0] / resolution],
            [0.0, 1.0, translation[1] / resolution]
        ])
        label = cv2.warpAffine(label, tm, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        # cv2.imshow('label after', visualize_label(label))
        # cv2.waitKey(0)

        if 'noisy_z' in self.config.keys():
            points[:, 2] += np.clip(rng.randn(len(points)) * 0.5, -0.3, 0.3)

        if 'noisy_remission' in self.config.keys():
            points[:, 3] += rng.randn(len(points)) * 0.1

        return points, label

    def map_class_labels(self, labels):
        # FIXME: hardcoded kitti19 learning map
        learning_map = {0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10,
                        48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19, 99: 0,
                        252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5}
        u, inv = np.unique(labels, return_inverse=True)
        return np.array([learning_map[x] for x in u])[inv].reshape(labels.shape)

    def _locate_sample(self, idx):
        seq_idx = bisect.bisect_right(self.seq_len_cumsum, idx)
        if seq_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.seq_len_cumsum[seq_idx - 1]
        return self.config['sequences'][seq_idx], sample_idx

    def _get_bev_label_file(self, seq_id, scan_file):
        name = os.path.basename(os.path.splitext(scan_file)[0])
        return os.path.join(self.root, seq_id, self.config['bev_label_dir'], name + '.png')

    def _merge_scans(self, scan_files, poses, end_idx, n):
        scans = []
        scan_poses = []
        for j in range(n):
            idx = max(end_idx - j * self.buffer_scan_stride, 0)
            scan = np.fromfile(scan_files[idx], dtype=np.float32).reshape(-1, 4)
            scans.append(scan)
            scan_poses.append(poses[idx])
        return aggregate_scans(scans, scan_poses)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        seq_id, sample_idx = self._locate_sample(index)

        scan_file = self.scan_files[seq_id][sample_idx]
        label_file = self._get_bev_label_file(seq_id, scan_file)
        points = self._merge_scans(self.scan_files[seq_id], self.poses[seq_id], sample_idx,
                                   self.buffer_scans)

        if 'drop_points' in self.config['augment']:
            points, point_idxs = drop_points(points, 0.8)

        label = Image.open(label_file)
        if self.label_shape is not None:
            label = label.resize(self.label_shape, resample=Image.NEAREST)
        label = np.array(label, dtype=np.uint8)

        if 'transform' and self.config['augment']:
            points, label = self.augment(points, label)

        # Add an additional channel storing the point indices
        points_with_idx = np.concatenate([
            points, np.arange(len(points))[:, None].astype(points.dtype)], axis=-1)
        voxels, coords, num_points = self.voxel_generator.generate(points_with_idx, max_voxels=90000)

        voxel_point_idxs = voxels[:, :, -1].astype(np.int32)  # num_voxels x max_num_points_per_voxel
        voxels = voxels[:, :, :-1]

        # Dummy values
        voxel_point_pixel_idxs = np.array([0.0], np.float32)
        voxel_point_pixel_inrange = np.array([0.0], np.float32)
        voxel_point_segmentation = np.array([0.0], np.float32)

        return {
            'voxels': voxels,
            'coords': coords,
            'num_points': num_points,
            'label': label,
            'points': points,
            'img': np.array([0.0], np.float32),  # Dummy value
            'voxel_point_pixel_idxs': voxel_point_pixel_idxs,
            'voxel_point_pixel_inrange': voxel_point_pixel_inrange,
            'voxel_point_idxs': voxel_point_idxs,
            'voxel_point_segmentation': voxel_point_segmentation,
        }


class BEVLoaderDataGeneration(BEVLoader):
    """
    A helper class for generating pretrained features
    """
    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files_input[index].rstrip()
        name = os.path.basename(img_path)[:-4]  # no extension
        points = np.fromfile(img_path, dtype=np.float32, count=-1).reshape([-1, 4])
        assert not self.aug
        voxels, coords, num_points = self.voxel_generator.generate(points, max_voxels=90000)
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        return name, voxels, coords, num_points


class BEVLoaderMultistepV2(data.IterableDataset):
    """
    Load multiple frames for training recurrent models.
    """
    # overwrite the default threshold values in spconv.utils.points_to_voxel
    from functools import partial
    import spconv
    print('patch spconv to increase the allowable z range. This will not affect the point cloud range.')
    spconv.utils.points_to_voxel = partial(spconv.utils.points_to_voxel,
                                           height_threshold=-4.0,
                                           height_high_threshold=5.0)

    def __init__(
            self,
            config,
            dataset_path,
            split="train",
            voxel_generator=None,
            label_dir_name='bev_labels',
            augment=False,
            use_pretrained_features=False,
            arl=False,  # A hack to test on arl bags.
    ):
        """__init__

        :param config:
        :param split:
        """
        self.root = os.path.join(dataset_path, split)
        self.split = split
        self.input_base = os.path.join(self.root, 'velodyne')
        self.annotations_base = os.path.join(self.root, label_dir_name)
        self.files_input = recursive_glob(rootdir=self.input_base, suffix=".bin")
        self.voxel_generator = voxel_generator
        self.label_shape = None
        self.aug = augment
        self.shuffle = config['shuffle']
        self.n_frame = config['n_frame']
        self.frame_strides = config['frame_strides'] if 'frame_strides' in config else [1]
        self.seq_len = config['seq_len'] if 'seq_len' in config else self.n_frame
        self.poses = np.loadtxt(os.path.join(self.root, 'costmap_poses.txt')).astype(np.float32)
        self.counters = yaml.load(open(os.path.join(self.root, 'counters.yaml')),
                                  Loader=yaml.SafeLoader)

        self.use_pretrained_features = use_pretrained_features
        if use_pretrained_features:
            self.pretrained_feature_dir = os.path.join(self.root, 'features')

        self.arl = arl

    def _valid_range(self, seq_start, seq_end):
        ''' Check if seq_start and seq_end are from the same sequence '''

        def _inrange(i, s, e):
            return s <= i and i < e

        for _, (start, end) in self.counters.items():
            if _inrange(seq_start, start, end):
                # seq_end is exlusive so we check (seq_end - 1) is
                # within the same sequence.
                return _inrange(seq_end - 1, start, end)

    def init_sequences(self):
        self.sequences = []
        for _, (start, end) in self.counters.items():
            max_len = end - start
            if self.seq_len is None:
                seq_len = max_len
            else:
                seq_len = min(self.seq_len, max_len)
            starts = np.arange(start, end - seq_len + 1)

            if not self.shuffle:
                assert(len(self.frame_strides) == 1)
            strides = np.random.choice(self.frame_strides, starts.shape)

            ends = starts + strides * seq_len

            sequences = [x for x in zip(starts, ends, strides) if
                         self._valid_range(x[0], x[1])]
            self.sequences.extend(sequences)

        if self.shuffle:
            np.random.shuffle(self.sequences)

    def __len__(self):
        """__len__"""
        return len(self.files_input)

    def set_label_shape(self, label_shape):
        assert (isinstance(label_shape, tuple) and len(label_shape) == 2)
        self.label_shape = label_shape

    def _make(self, index):
        """
        Make a single data sample
        """
        img_path = self.files_input[index].rstrip()
        lbl_path = img_path.replace(self.input_base, self.annotations_base)[:-4] + ".png"
        name = img_path.split(os.sep)[-1]

        points = np.fromfile(img_path, dtype=np.float32, count=-1).reshape([-1, 4])

        if self.arl:
            # Normalize remission
            points[:, 3] /= 120000.
            # Normalize z
            points[:, 2] -= 1.4

        if self.arl:
            label = np.zeros((512, 512), np.uint8)
        else:
            label = Image.open(lbl_path)
            if self.label_shape is not None:
                label = label.resize(self.label_shape,
                                     resample=Image.NEAREST)

        if self.split == "train" and self.aug:
            points, _ = drop_points(points, 0.8)

        label = np.array(label, dtype=np.uint8)
        pose = self.poses[index]

        if self.use_pretrained_features:
            def try_load_npz():
                basename = os.path.basename(img_path)[:-4] + '.npz'
                feature_path = os.path.join(self.pretrained_feature_dir, basename)
                if os.path.isfile(feature_path):
                    return np.load(feature_path)['arr_0']
                return None

            def try_load_lz4():
                basename = os.path.basename(img_path)[:-4] + '.lz4'
                feature_path = os.path.join(self.pretrained_feature_dir, basename)
                if os.path.isfile(feature_path):
                    with lz4.frame.open(feature_path, mode='r') as f:
                        features = np.load(f)
                    return features
                return None

            def load_features():
                features = try_load_lz4()
                if features is not None:
                    return features
                return try_load_npz()

            features = load_features()

            coords = np.array([0.0], np.float32)  # dummy values
            num_points = np.array([0.0], np.float32)  # dummy values
            print(features)
            return features, coords, num_points, label, points, pose, index

        voxels, coords, num_points = self.voxel_generator.generate(points, max_voxels=90000)

        # Add batch index. Note that we assume training with a single batch so that the batch index
        # is always zero. If we want to train with batch size > 1, we must remove this line and
        # insert batch index in the collate function.
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        # print((voxels, coords, num_points,
                # label, points, pose, index))
        return (voxels, coords, num_points,
                label, points, pose, index)

    @staticmethod
    def collate_wrapper(batch):
        return BEVMultiStepBatch(batch)

    @staticmethod
    def init(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return
        np.random.seed(worker_info.seed & 0xffffffff)
        assert(worker_info.num_workers == 1 or
               self.n_frame == self.seq_len)

    def __iter__(self):
        """__getitem__

        :param index:
        """
        self.init_sequences()
        count = 0
        for start, end, stride in self.sequences:
            def _make(index):
                return self._make(index) + (index == start,)
            frame_idxs = np.arange(start, end, stride)
            ## read the frame_idxs n_frame by n_frame
            for index in np.arange(0, len(frame_idxs), self.n_frame):
                nframe_idxs = [frame_idxs[min(index + i, len(frame_idxs) - 1)] for i in range(self.n_frame)]
                yield list(zip(*[_make(idx) for idx in nframe_idxs]))

                count += 1
                if count == len(self):
                    return


class BEVLoaderMultistepV3(data.IterableDataset):
    """
    Load multiple frames for training recurrent models.
    """
    # overwrite the default threshold values in spconv.utils.points_to_voxel
    from functools import partial
    import spconv
    print('patch spconv to increase the allowable z range. This will not affect the point cloud range.')
    spconv.utils.points_to_voxel = partial(spconv.utils.points_to_voxel,
                                           height_threshold=-4.0,
                                           height_high_threshold=5.0)

    def __init__(
            self,
            config,
            dataset_path,
            shuffle,
            n_frame,
            seq_len,
            frame_strides,
            voxel_generator=None,
            n_buffer_scans=1,
            buffer_scan_stride=1,
    ):
        """__init__

        :param config:
        :param split:
        """
        self.root = dataset_path
        sequences = config['sequences']

        self.poses = dict()
        self.costmap_poses = dict()
        self.scan_files = dict()
        self.end_points = dict()
        seq_lens = []
        start = 0
        for seq in sequences:
            self.poses[seq] = np.loadtxt(os.path.join(self.root, seq, 'poses.txt')).astype(np.float32)
            self.costmap_poses[seq] = np.loadtxt(os.path.join(self.root, seq, 'costmap_poses.txt')).astype(np.float32)
            self.scan_files[seq] = sorted(glob.glob(os.path.join(self.root, seq, 'velodyne', '*.bin')))
            l = len(self.scan_files[seq])
            seq_lens.append(l)
            self.end_points[seq] = (start, start + l)
            start += l
        self.seq_len_cumsum = np.cumsum(seq_lens)

        self.voxel_generator = voxel_generator
        self.label_shape = None

        self.shuffle = shuffle
        self.n_frame = n_frame
        self.frame_strides = frame_strides
        self.seq_len = seq_len

        self.buffer_scans = n_buffer_scans
        self.buffer_scan_stride = buffer_scan_stride

        self.config = config

    def _valid_range(self, seq_start, seq_end):
        ''' Check if seq_start and seq_end are from the same sequence '''
        def _inrange(i, s, e):
            return s <= i < e

        for _, (start, end) in self.end_points.items():
            if _inrange(seq_start, start, end):
                # seq_end is exlusive so we check (seq_end - 1) is
                # within the same sequence.
                return _inrange(seq_end - 1, start, end)

    def gen_sequences(self):
        sequences = []
        for seq_id in self.config['sequences']:
            start, end = self.end_points[seq_id]
            max_len = end - start
            if self.seq_len is None:
                seq_len = max_len
            else:
                seq_len = min(self.seq_len, max_len)
            starts = np.arange(start, end - seq_len + 1)

            if not self.shuffle:
                assert(len(self.frame_strides) == 1)
            strides = np.random.choice(self.frame_strides, starts.shape)

            ends = starts + strides * seq_len

            sequences.extend([x for x in zip(starts, ends, strides) if
                             self._valid_range(x[0], x[1])])

        if self.shuffle:
            np.random.shuffle(sequences)
        return sequences

    def __len__(self):
        return sum([len(_) for _ in self.scan_files.values()])

    def set_label_shape(self, label_shape):
        assert (isinstance(label_shape, tuple) and len(label_shape) == 2)
        self.label_shape = label_shape

    def _locate_sample(self, idx):
        seq_idx = bisect.bisect_right(self.seq_len_cumsum, idx)
        if seq_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.seq_len_cumsum[seq_idx - 1]
        return self.config['sequences'][seq_idx], sample_idx

    def _get_bev_label_file(self, seq_id, scan_file):
        name = os.path.basename(os.path.splitext(scan_file)[0])
        return os.path.join(self.root, seq_id, self.config['bev_label_dir'], name + '.png')

    def _merge_scans(self, scan_files, poses, end_idx, n):
        scans = []
        scan_poses = []
        for j in range(n):
            idx = max(end_idx - j * self.buffer_scan_stride, 0)
            scan = np.fromfile(scan_files[idx], dtype=np.float32).reshape(-1, 4)
            scans.append(scan)
            scan_poses.append(poses[idx])
        return aggregate_scans(scans, scan_poses)

    def _rotate(self, points, label, angle):
        """
        Rotate points and label.
        Args:
            points:
            label:
            angle:
        Returns:

        """
        points = points.copy()
        label = label.copy()

        R = transforms3d.euler.euler2mat(0.0, 0.0, angle)
        points[:, :3] = points[:, :3] @ R.T

        h, w = label.shape
        origin = self.config['origin']
        if origin == 'center':
            rotation_center = (w // 2, h // 2)
        elif origin == 'center_left':
            rotation_center = (0, h // 2)
        else:
            raise RuntimeError('Unsupported vehicle origin.')

        # Rotate around the center left.
        # Note that the angle is negated.
        M = cv2.getRotationMatrix2D(rotation_center, -np.rad2deg(angle), 1.0)
        label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        return points, label

    def _make(self, index, rotation):
        """
        Make a single data sample
        """
        seq_id, sample_idx = self._locate_sample(index)

        scan_file = self.scan_files[seq_id][sample_idx]
        label_file = self._get_bev_label_file(seq_id, scan_file)
        points = self._merge_scans(self.scan_files[seq_id], self.poses[seq_id], sample_idx,
                                   self.buffer_scans)

        label = Image.open(label_file)
        if self.label_shape is not None:
            label = label.resize(self.label_shape, resample=Image.NEAREST)
        label = np.array(label, dtype=np.uint8)

        costmap_pose = self.costmap_poses[seq_id][sample_idx]

        if 'drop_points' in self.config['augment']:
            points, _ = drop_points(points, 0.8)

        if rotation != 0.0:
            # TODO: remove hardcoded arguments
            pose = self.poses[seq_id][sample_idx].reshape((3, 4))
            pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])])
            minx = -51.2
            miny = -51.2
            resx = 0.2
            resy = 0.2
            K_inv = np.array([[resx, 0, 0, minx], [0, resy, 0, miny], [0, 0, 1, 0], [0, 0, 0, 1]])
            R = np.eye(4)
            R[:3, :3] = axis_angle_to_rotmat([0, 0, 1], -rotation)
            costmap_pose = pose @ R @ K_inv
            costmap_pose = costmap_pose[[0, 1, 3]][:, [0, 1, 3]]
            costmap_pose = costmap_pose[:2].reshape(-1)
            costmap_pose = costmap_pose.astype(np.float32)
            points, label = self._rotate(points, label, rotation)

        voxels, coords, num_points = self.voxel_generator.generate(points, max_voxels=90000)

        # Add batch index. Note that we assume training with a single batch so that the batch index
        # is always zero. If we want to train with batch size > 1, we must remove this line and
        # insert batch index in the collate function.
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        return (voxels, coords, num_points,
                label, points, costmap_pose, index)

    @staticmethod
    def collate_wrapper(batch):
        return BEVMultiStepBatch(batch)

    @staticmethod
    def init(worker_id):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return
        np.random.seed(worker_info.seed & 0xffffffff)
        assert(worker_info.num_workers == 1 or
               self.n_frame == self.seq_len)

    def __iter__(self):
        """__getitem__

        :param index:
        """
        sequences = self.gen_sequences()
        count = 0

        rng = np.random

        for start, end, stride in sequences:
            if 'rand_rotation' in self.config['augment']:
                rotation = rng.uniform(-np.pi / 4, np.pi / 4)
            else:
                rotation = 0.0

            def _make(index, rotation):
                return self._make(index, rotation) + (index == start,)

            frame_idxs = np.arange(start, end, stride)

            if 'sim_still' in self.config['augment']:
                p = rng.uniform(0.0, 1.0)
                if p < 0.2:
                    mid_frame = len(frame_idxs) // 2
                    for i in range(mid_frame + 1, len(frame_idxs)):
                        frame_idxs[i] = frame_idxs[mid_frame]

            ## read the frame_idxs n_frame by n_frame
            for index in np.arange(0, len(frame_idxs), self.n_frame):
                nframe_idxs = [frame_idxs[min(index + i, len(frame_idxs) - 1)]
                               for i in range(self.n_frame)]
                yield list(zip(*[_make(idx, rotation) for idx in nframe_idxs]))

                count += 1
                if count == len(self):
                    return


class BEVMultiStepBatch:
    def __init__(self, data):
        """
        Supports batched tensors of inconsistent dimensions.
        Args:
            data: a list of list of tuples.
                  Outer list is the batch. Inner list is the frames.
        """
        data = list(zip(*data))
        (self.voxels, self.coords, self.num_points,
         self.label, self.points, self.pose, self.seq_id,
         self.seq_start) = data

        # Convert to torch tensors. No stacking.
        self.voxels = self._to_torch(self.voxels)
        self.coords = self._to_torch(self.coords)
        self.num_points = self._to_torch(self.num_points)
        self.points = self._to_torch(self.points)

        # These tensors have fixed dimensions so we can stack them.
        self.label = torch.stack([torch.as_tensor(_) for _ in self.label])
        self.pose = torch.stack([torch.as_tensor(_) for _ in self.pose])
        self.seq_start = torch.stack([torch.as_tensor(_) for _ in
                                      self.seq_start])

    def _to_torch(self, l):
        return [[torch.as_tensor(frame_data) for frame_data in batch] for batch in l]

    def _pin_memory(self, l):
        return [[frame_data.pin_memory() for frame_data in batch] for batch in l]

    def pin_memory(self):
        self.voxels, self.coords, self.num_points, self.points = [
            self._pin_memory(_) for _ in
            (self.voxels, self.coords, self.num_points, self.points)]
        self.label = self.label.pin_memory()
        return self


def _1hot(y, num_classes):
    y_1hot = torch.BoolTensor(y.shape[0],
                              num_classes).to(device=y.device)
    y_1hot.zero_()
    y_1hot.scatter_(1, y[:, None].to(torch.int64), 1)
    return y_1hot


def _idiv(a, b):
    ignore = b == 0
    b[ignore] = 1
    div = a / b
    div[ignore] = np.nan
    return div


class NumpyEvaluator(object):
    def __init__(self, num_classes, ignore_label=255):
        self.labels = np.arange(num_classes)
        self.ignore_label = ignore_label
        self.conf_mat = None

    def reset(self):
        self.conf_mat = None

    def append(self, pred, label):
        valid = label != self.ignore_label

        pred, label = pred[valid], label[valid]

        conf = confusion_matrix(label.cpu(), pred.cpu(), labels=self.labels)

        if self.conf_mat is None:
            self.conf_mat = conf
        else:
            self.conf_mat += conf

    def classwiseAcc(self):
        tp = np.diag(self.conf_mat)
        total = self.conf_mat.sum(axis=0)
        return _idiv(tp, total)

    def acc(self):
        return np.nanmean(self.classwiseAcc())

    def classwiseIoU(self):
        tp = np.diag(self.conf_mat)
        fn = self.conf_mat.sum(axis=1) - tp
        tp_fp = self.conf_mat.sum(axis=0)
        return _idiv(tp, tp_fp + fn)

    def meanIoU(self):
        return np.nanmean(self.classwiseIoU())


class Evaluator(object):
    def __init__(self, num_classes, ignore_label=255):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.tp = None

    def reset(self):
        self.tp = None

    def append(self, pred, label, valid_mask=None):
        valid = label != self.ignore_label

        if valid_mask is not None:
            valid &= valid_mask

        pred, label = pred[valid], label[valid]

        pred_1hot = _1hot(pred, self.num_classes)
        label_1hot = _1hot(label, self.num_classes)

        tp = (pred_1hot & label_1hot).sum(0)
        pred_total = pred_1hot.sum(0)
        label_total = label_1hot.sum(0)

        if self.tp is None:
            self.tp = tp
            self.pred_total = pred_total
            self.label_total = label_total
        else:
            self.tp += tp
            self.pred_total += pred_total
            self.label_total += label_total

    def classwiseAcc(self):
        return (self.tp / self.pred_total).cpu().numpy()

    def acc(self):
        return np.nanmean(self.classwiseAcc())

    def classwiseIoU(self):
        iou = self.tp / (self.pred_total + self.label_total - self.tp)
        return iou.cpu().numpy()

    def meanIoU(self):
        return np.nanmean(self.classwiseIoU())


# Augmentations
 
def random_td_flip(points, label, probability=0.5):
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability])
    if enable:
        label = label.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        points[:, 1] = -points[:, 1]
    return points, label


def random_lr_flip(points, label, probability=0.5):
    enable = np.random.choice(
        [False, True], replace=False, p=[1 - probability, probability])
    if enable:
        label = label.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        points[:, 0] = -points[:, 0]
    return points, label


def bev_single_collate_fn(data):
    """
        {
            'voxels': voxels,
            'coords': coords,
            'num_points': num_points,
            'label': label,
            'points': points,
            'img': img,
            'voxel_point_pixel_idxs': voxel_point_pixel_idxs,
            'voxel_point_pixel_inrange': voxel_point_pixel_inrange,
            'voxel_point_idxs': voxel_point_idxs,
            'voxel_point_segmentation': voxel_point_segmentation,
        }
    """

    out = dict()

    def make_batch(key):
        return [_[key] for _ in data]

    for key in data[0].keys():
        if key not in ['voxels', 'coords', 'points', 'num_points',
                       'voxel_point_pixel_idxs',
                       'voxel_point_pixel_inrange',
                       'voxel_point_idxs',
                       'voxel_point_segmentation']:
            try:
                out[key] = torch.as_tensor(np.stack(make_batch(key)))
            except:
                print(key)
                raise

    batch_coords = []
    for i in range(len(data)):
        # Add batch index
        batch_coords.append(np.pad(data[i]['coords'], ((0, 0), (1, 0)),
                                   mode='constant', constant_values=i))
    out['coordinates'] = torch.as_tensor(np.concatenate(batch_coords))
    out['voxels'] = torch.as_tensor(np.concatenate(make_batch('voxels')))
    out['num_points'] = torch.as_tensor(np.concatenate(make_batch('num_points')))
    out['points'] = make_batch('points')

    out['voxel_point_segmentation'] = torch.as_tensor(
        np.concatenate(make_batch('voxel_point_segmentation')))

    return out


def compute_pixel_indices(scan, lidar_to_cam, K, width, height, img=None):
    """
    Args:
        scan:  N x (3 or 4) points.
        lidar_to_cam: a 4 x 4 matrix of the lidar to camera transform.
        K: camera intrinsics
        width: camera image width
        height: camera image height
    Returns:
        inrange: a boolean array of size N. It is a mask indicating points that are visible from the camera.
        pixel_idxs: an integer array of size M. It stores the corresponding pixel index of each visible point.
                    The pixel indices are computed by flattening the 2D image into 1D.
    """
    scan_homo = np.concatenate([scan[:, :3], np.ones((scan.shape[0], 1), scan.dtype)], axis=-1)
    scan_cam2 = scan_homo @ lidar_to_cam.T
    scan_cam2 = scan_cam2[:, :3]

    scan_proj = scan_cam2 @ K.T
    frontal_points_mask = scan_proj[:, 2] > 0
    scan_proj = scan_proj / scan_proj[:, 2][:, None]
    scan_proj = scan_proj[:, :2].astype(np.int32)  # N x (x, y) pixel coordinates

    # Mask of points that are visible in the image
    inrange = frontal_points_mask & \
              (0 <= scan_proj[:, 0]) & (scan_proj[:, 0] < width) & \
              (0 <= scan_proj[:, 1]) & (scan_proj[:, 1] < height)

    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # for i in range(len(scan_proj)):
    #     if not inrange[i]:
    #         continue
    #     x, y = scan_proj[i]
    #     cv2.circle(img, (int(x), int(y)), radius=1, color=(0, 255, 0), thickness=-1)
    # cv2.imshow('', img)
    # cv2.waitKey(0)

    pixel_idxs = scan_proj[:, 1] * width + scan_proj[:, 0]
    pixel_idxs = pixel_idxs * inrange  # Set out-of-range pixel idx to 0
    return inrange, pixel_idxs


def drop_points(points, keep_ratio):
    rng = np.random
    n_keep = int(len(points) * keep_ratio)
    idxs = rng.choice(len(points), n_keep, replace=False)
    rng.shuffle(idxs)
    return points[idxs], idxs


def axis_angle_to_rotmat(axis, angle):
    # Axis must be a unit vector.
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
