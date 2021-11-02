
import numpy as np
import cv2
from matplotlib.pyplot import imshow, show
import yaml
import os
from PIL import Image
import pickle
from joblib import Parallel, delayed

def create_mask(h, w, traj, thickness, weight=1.0):
    '''
        traj: np.int32 (Nx2) array. Each point traj[i] specify a *xy*
        location on the trajectory.
        weight: weight of the trajectory
    '''

    cvf = cv2.polylines(np.zeros((h, w), dtype=np.float32),
                        traj.reshape((1, -1, 1, 2)),
                        isClosed=False,
                        color=(weight,),
                        thickness=thickness)
    return cvf


def parse_poses(filename):
    """ read poses file with per-scan poses from given filename
        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    file = open(filename)

    poses = []

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(pose)

    return poses

def make_costmap_pose(pose, cfg):
    minx, miny = cfg['voxelizer']['point_cloud_range'][:2]
    gridw, gridh, _ = cfg['voxelizer']['voxel_size']


    vox_cfg = cfg['voxelizer']
    # scale
    inv_proj_mat = np.diag([gridw, gridh,
                            1.0, 1.0])

    # shift
    inv_proj_mat[0, 3] = minx
    inv_proj_mat[1, 3] = miny

    costmap_pose = np.matmul(pose, inv_proj_mat)
    # remove z dimension
    costmap_pose = costmap_pose[[0, 1, 3]][:, [0, 1, 3]]
    return costmap_pose

def convert_poses(base_frame, poses):
    """
    Convert set of calibration matrices (poses) with given frame of reference.
    Parameters
    -------
    base_frame: 4 x 4 numpy array
    poses: N x 4 x 4 numpy array
    """
    key_pose_inv = np.linalg.inv(base_frame)
    converted = np.matmul(key_pose_inv, poses)
    return converted

def create_traj_mask(base_frame, poses, cfg):
    traj = convert_poses(base_frame, poses)[:, :-1, -1]
    pt_range = cfg['voxelizer']

    minx, miny, _, maxx, maxy, _ = cfg['voxelizer']['point_cloud_range']
    gridw, gridh, _ = cfg['voxelizer']['voxel_size']

    h = np.floor((maxy - miny)/gridh).astype(np.int32)
    w = np.floor((maxx - minx)/gridw).astype(np.int32)
    xs = np.floor((traj[:, 0] - minx) / gridw).astype(np.int32)
    ys = np.floor((traj[:, 1] - miny) / gridh).astype(np.int32)


    locs = np.stack((xs, ys), 1)

    # Create mask
    mask = create_mask(h, w, locs, thickness=10)
    mask = mask.astype(np.uint8)

    # Create trajectories
    traj_segments = []

    inrange = (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
    pinrange = np.r_[False, inrange, False]

    #Get the shifting indices
    idx = np.flatnonzero(pinrange[1:] != pinrange[:-1])
    start, end = idx[::2], idx[1::2]

    for s, e in zip(start, end):
        lenght = e - s

        # skip small segments
        if e - s < 50:
            continue

        # remove duplications
        # usually happen when robot is not moving
        segment = np.unique(locs[s:e], axis=0)

        traj_segments.append(segment)

    trajectories = dict()
    trajectories['segments'] = traj_segments
    return mask, traj_segments

def make_color_map():
    semantic_cmap = np.zeros((256, 3), np.uint8) + 128
    semantic_cmap[0] = (0, 0, 0)
    semantic_cmap[1] = (255, 255, 255)
    return semantic_cmap

def process_seq(seq_dir):
    cmap = make_color_map()

    label_path = os.path.join(seq_dir, 'bev_trajectory_labels')
    traj_path = os.path.join(seq_dir, 'bev_trajectory')

    poses = parse_poses(os.path.join(seq_dir, 'poses.txt'))
    os.makedirs(label_path, exist_ok=True)
    os.makedirs(traj_path, exist_ok=True)

    output_files = [_[:-4] + '.png' for _ in os.listdir(os.path.join(seq_dir, 'velodyne'))
                    if _.endswith('.bin')]
    output_files = sorted(output_files)
    assert(len(output_files) == len(poses))
    costmap_poses = []
    for i, fn in enumerate(output_files):
        traj_mask, traj_properties = create_traj_mask(poses[i], poses, cfg)
        costmap_poses.append(make_costmap_pose(poses[i], cfg))
        traj_img = Image.fromarray(traj_mask, mode='P')
        traj_img.putpalette(cmap)
        traj_img.save(os.path.join(label_path, fn))
        with open(os.path.join(traj_path, fn + '.pkl'), 'wb') as f:
            pickle.dump(traj_properties, f)

    costmap_poses = np.array(costmap_poses)
    costmap_poses = costmap_poses.reshape((costmap_poses.shape[0], -1))
    np.savetxt(os.path.join(seq_dir,  'costmap_poses.txt'), costmap_poses,
               fmt='%.8e')

def process_dataset(path, cfg, n_jobs=12):
    seq_ids = os.listdir(path)

    def fn(seq_id):
        print(seq_id)
        seq_dir = os.path.join(path, seq_id)
        process_seq(seq_dir)

    if n_jobs == 1:
        for seq_id in seq_ids:
            fn(seq_id)
    else:
        Parallel(n_jobs=n_jobs)(delayed(fn)(seq_id) for seq_id in seq_ids)



db_path = './Dataset'

cfg_fn = 'dataset.yaml'

with open(cfg_fn, 'r') as f:
    cfg = yaml.safe_load(f)

process_dataset(db_path, cfg)
