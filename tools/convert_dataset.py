# Convert previous dataset format into the new version where every sequence has its own directory.
import glob
import os
import numpy as np
import shutil
import tqdm
import yaml
import sys


def convert(src_dir, dst_dir):
    counters = yaml.load(open(os.path.join(src_dir, 'counters.yaml')), Loader=yaml.SafeLoader)
    counters.items()
    segments = sorted(list(counters.items()), key=lambda a: a[1][0])

    def copy_helper(seq_id, dname, fmt, start, end):
        src_subdir = os.path.join(os.path.join(src_dir, dname))
        dst_subdir = os.path.join(os.path.join(dst_dir, seq_id, dname))
        os.makedirs(dst_subdir, exist_ok=True)
        for i in range(start, end):
            shutil.copy(os.path.join(src_subdir, fmt % i), dst_subdir)

    poses = np.loadtxt(os.path.join(src_dir, 'poses.txt'))
    costmap_poses = np.loadtxt(os.path.join(src_dir, 'costmap_poses.txt'))

    for seq_id, (start, end) in segments:
        print('copying', seq_id)
        subdir = os.path.join(dst_dir, str(seq_id))
        copy_helper(str(seq_id), 'bev_labels', '%05d.png', start, end)
        copy_helper(str(seq_id), 'velodyne', '%05d.bin', start, end)
        np.savetxt(os.path.join(subdir, 'poses.txt'), poses[start:end], '%.8e')
        np.savetxt(os.path.join(subdir, 'costmap_poses.txt'), costmap_poses[start:end], '%.8e')


SRC = sys.argv[1]  # '../data/semantic_kitti_4class_100x100'
DST = sys.argv[2]  # '../data/semantic_kitti_4class_100x100_v2'

convert(os.path.join(SRC, 'sequences/train/'), DST)
convert(os.path.join(SRC, 'sequences/valid/'), DST)
