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
def hill_3_clean():
    scan_dir = get_data_dir() + '/hill-3-clean_4class_100x100/00000/velodyne'
    label_dir = None
    pose_file = get_data_dir() + '/hill-3-clean_4class_100x100/00000/poses.txt'
    costmap_pose_file = get_data_dir() + '/hill-3-clean_4class_100x100/00000/costmap_poses.txt'
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
def campus_2021_03_31_18_44_00():
    scan_dir = get_data_dir() + '/campus_2021-03-31-18-44-00/0/velodyne'
    label_dir = None
    pose_file = get_data_dir() + '/campus_2021-03-31-18-44-00/0/poses.txt'
    costmap_pose_file = get_data_dir() + '/campus_2021-03-31-18-44-00/0/costmap_poses.txt'
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
    scan_dir = get_data_dir() + '/semantic_kitti_4class_100x100/0/velodyne'
    label_dir = None
    pose_file = get_data_dir() + '/semantic_kitti_4class_100x100/0/poses.txt'
    costmap_pose_file = get_data_dir() + '/semantic_kitti_4class_100x100/0/costmap_poses.txt'
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
def debug():
    scan_dir = '/tmp/bag/velodyne'
    label_dir = None
    pose_file = '/tmp/bag/poses.txt'
    costmap_pose_file = '/tmp/bag/costmap_poses.txt'
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
def gravel_path_autonomous():
    scan_dir = '/mnt/ssd1/SARA/phoenix-r1/bags/arl/gravel-path-autonomous-7_2020-09-09/sequences/valid/velodyne'
    label_dir = None
    pose_file = '/mnt/ssd1/SARA/phoenix-r1/bags/arl/gravel-path-autonomous-7_2020-09-09/sequences/valid/poses.txt'
    costmap_pose_file = '/mnt/ssd1/SARA/phoenix-r1/bags/arl/gravel-path-autonomous-7_2020-09-09/sequences/valid/costmap_poses.txt'
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
def kitti_rellis_hill():
    # seq = 'kitti_08'
    seq = 'rellis_00004'
    scan_dir = get_data_dir() + '/kitti-rellis-hill/%s/velodyne' % seq
    label_dir = get_data_dir() + '/kitti-rellis-hill/%s/bev_labels' % seq
    pose_file = get_data_dir() + '/kitti-rellis-hill/%s/poses.txt' % seq
    costmap_pose_file = get_data_dir() + '/kitti-rellis-hill/%s/costmap_poses.txt' % seq
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
def weeds_ouster_2020_08_16_16_47_26():
    root = '/mnt/ssd1/SARA/phoenix-r1/bags/08-16-2020/weeds_ouster_2020-08-16-16-47-26/'
    scan_dir = root + 'velodyne'
    label_dir = None
    pose_file = root + 'poses.txt'
    costmap_pose_file = root + 'costmap_poses.txt'
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
def campus_autonomous_2020_11_12_19_14_34():
    root = '/mnt/ssd1/SARA/phoenix-r1/bags/11-12-2020/campus_autonomous_2020-11-12-19-14-34/'
    scan_dir = root + 'velodyne'
    label_dir = None
    pose_file = root + 'poses.txt'
    costmap_pose_file = root + 'costmap_poses.txt'
    return _helper(
        scan_dir=scan_dir,
        label_dir=label_dir,
        pose_file=pose_file,
        costmap_pose_file=costmap_pose_file,
        img_dir=root + 'images'
    )


@register
def weeds2_2021_02_20_16_18_02():
    root = '/mnt/ssd1/SARA/phoenix-r1/bags/02-20-2021/weeds2_2021-02-20-16-18-02/'
    scan_dir = root + 'velodyne'
    label_dir = None
    pose_file = root + 'poses.txt'
    costmap_pose_file = root + 'costmap_poses.txt'
    return _helper(
        scan_dir=scan_dir,
        label_dir=label_dir,
        pose_file=pose_file,
        costmap_pose_file=costmap_pose_file,
        img_dir=root + 'images'
    )


@register
def canal_ouster_2_2020_08_16_17_02_53():
    root = '/mnt/ssd1/SARA/phoenix-r1/bags/08-16-2020/canal_ouster_2_2020-08-16-17-02-53/sequences/00000/'
    scan_dir = root + 'velodyne'
    label_dir = None
    pose_file = root + 'poses.txt'
    costmap_pose_file = root + 'costmap_poses.txt'
    return _helper(
        scan_dir=scan_dir,
        label_dir=label_dir,
        pose_file=pose_file,
        costmap_pose_file=costmap_pose_file,
        img_dir=root + 'images',
    )


def make(name):
    return _env_dict[name]()
