# Combine multiple datasets into a single one.
# Normalize the scans such that
#   * z = 0 is at the bottom of the robot base
#   * remission values are in the range of [0, 1]

import glob
import os
import numpy as np
import shutil


datasets = {
    'kitti': {
        'path': '../data/semantic_kitti_4class_100x100',
        'lidar_z_offset': 1.7,
        'remission_scale': 1.0,
    },
    'rellis': {
        'path': '../data/rellis_4class_100x100',
        'lidar_z_offset': 1.06,
        'remission_scale': 100.0,
    },
    'hill-3': {
        'path': '../data/hill-3-clean_4class_100x100',
        'lidar_z_offset': 1.4,
        'remission_scale': 1.0
    }
}


def adjust_lidar_scan(scan, z_offset, remission_scale):
    scan[:, 2] += z_offset
    scan[:, 3] *= remission_scale


def process_dataset(name, cfg, out_dir):
    seq_ids = os.listdir(cfg['path'])
    for seq_id in seq_ids:
        seq_dir = os.path.join(cfg['path'], seq_id)
        out_seq_id = name + '_' + seq_id

        # Copy scans
        scan_dir = os.path.join(seq_dir, 'velodyne')
        scan_out_dir = os.path.join(out_dir, out_seq_id, 'velodyne')
        os.makedirs(scan_out_dir, exist_ok=True)
        scan_files = glob.glob(os.path.join(scan_dir, '*.bin'))
        for scan_fn in scan_files:
            scan = np.fromfile(scan_fn, np.float32).reshape(-1, 4)
            adjust_lidar_scan(scan, cfg['lidar_z_offset'], cfg['remission_scale'])
            scan.tofile(os.path.join(scan_out_dir, os.path.basename(scan_fn)))

        # Copy bev labels
        bev_label_dir = os.path.join(seq_dir, 'bev_labels')
        bev_label_out_dir = os.path.join(out_dir, out_seq_id, 'bev_labels')
        shutil.copytree(bev_label_dir, bev_label_out_dir)

        # Copy other files
        shutil.copy(os.path.join(seq_dir, 'poses.txt'), os.path.join(out_dir, out_seq_id))
        shutil.copy(os.path.join(seq_dir, 'costmap_poses.txt'), os.path.join(out_dir, out_seq_id))


OUT_DIR = '../data/kitti_rellis_hill'

for name, value in datasets.items():
    process_dataset(name, value, OUT_DIR)
