import glob
import os

import numpy as np

from .utils import get_data_dir


_env_dict = {}


def register(f):
    _env_dict[f.__name__] = f
    return f


def _helper(scan_dir, label_dir, pose_file, costmap_pose_file, img_dir=None):
    scan_files = sorted(glob.glob(os.path.join(scan_dir, '*.bin')))

    if label_dir is not None:
        label_files = sorted(glob.glob(os.path.join(label_dir, '*.png')))
        assert len(scan_files) == len(label_files)
    else:
        label_files = None

    if img_dir is not None:
        img_files = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    else:
        img_files = None

    poses = np.loadtxt(pose_file).astype(np.float32)
    assert len(scan_files) == len(poses)

    costmap_poses = np.loadtxt(costmap_pose_file).astype(np.float32)
    assert len(scan_files) == len(costmap_poses)

    return {
        'scan_dir': scan_dir,
        'label_dir': label_dir,
        'costmap_pose_file': costmap_pose_file,
        'scan_files': scan_files,
        'label_files': label_files,
        'image_files': img_files,
        'poses': poses,
        'costmap_poses': costmap_poses,
    }


@register
def rellis4():
    scan_dir = get_data_dir() + '/rellis_4class_100x100/00004/velodyne'
    label_dir = get_data_dir() + '/rellis_4class_100x100/00004/bev_labels'
    pose_file = get_data_dir() + '/rellis_4class_100x100/00004/poses.txt'
    costmap_pose_file = get_data_dir() + '/rellis_4class_100x100/00004/costmap_poses.txt'
    return _helper(
        scan_dir=scan_dir,
        label_dir=label_dir,
        pose_file=pose_file,
        costmap_pose_file=costmap_pose_file,
    )


@register
def rellis4_notrim():
    scan_dir = get_data_dir() + '/rellis_4class_100x100_notrim/00004/velodyne'
    label_dir = get_data_dir() + '/rellis_4class_100x100_notrim/00004/bev_labels'
    pose_file = get_data_dir() + '/rellis_4class_100x100_notrim/00004/poses.txt'
    costmap_pose_file = get_data_dir() + '/rellis_4class_100x100_notrim/00004/costmap_poses.txt'
    scan_files, label_files, poses, costmap_poses = _helper(
        scan_dir=scan_dir,
        label_dir=label_dir,
        pose_file=pose_file,
        costmap_pose_file=costmap_pose_file,
    )
    return {
        'scan_dir': scan_dir,
        'label_dir': label_dir,
        'costmap_pose_file': costmap_pose_file,
        'scan_files': scan_files,
        'label_files': label_files,
        'poses': poses,
        'costmap_poses': costmap_poses
    }


@register
def kitti4():
    seq_dir = get_data_dir() + '/semantic_kitti_4class_100x100/08/'
    scan_dir = os.path.join(seq_dir, 'velodyne')
    label_dir = os.path.join(seq_dir, 'bev_labels')
    pose_file = os.path.join(seq_dir, 'poses.txt')
    costmap_pose_file = os.path.join(seq_dir, 'costmap_poses.txt')
    return _helper(
        scan_dir=scan_dir,
        label_dir=label_dir,
        pose_file=pose_file,
        costmap_pose_file=costmap_pose_file
    )


def make(name):
    return _env_dict[name]()
