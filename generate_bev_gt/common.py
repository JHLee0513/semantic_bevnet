from numba import jit
import cv2
import numpy as np
from scipy import ndimage
from numpy.linalg import inv
from postprocessing import BinningPostprocess
import torch


def parse_calibration(filename):
    """ read calibration file with given filename

        Returns
        -------
        dict
            Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def parse_poses(filename, calibration):
    """ read poses file with per-scan poses from given filename

        Returns
        -------
        list
            list of poses as 4x4 numpy arrays.
    """
    file = open(filename)

    poses = []

    Tr = calibration["Tr"]
    Tr_inv = inv(Tr)

    for line in file:
        values = [float(v) for v in line.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

    return poses


def map_labels(labels, learning_map):
    labels = np.bitwise_and(labels, 0xFFFF)

    ucls = np.unique(labels)
    new_labels = np.zeros_like(labels)
    for cls in ucls:
        if cls not in learning_map:
            new_labels[labels == cls] = 0
        else:
            new_labels[labels == cls] = learning_map[cls]
    return new_labels


def join_pointclouds(history, key_pose):
    # prepare single numpy array for all points that can be written at once.
    num_concat_points = sum([past["points"].shape[0] for past in history])
    concated_points = np.zeros(num_concat_points * 4, dtype=np.float32)
    concated_labels = np.zeros(num_concat_points, dtype=np.uint32)
    pc_ids = np.zeros(num_concat_points, dtype=np.int32)

    start = 0
    key_pose_inv = inv(key_pose)
    for i, past in enumerate(history):
        diff = np.matmul(key_pose_inv, past["pose"])
        tpoints = np.matmul(diff, past["points"].T).T
        tpoints[:, 3] = past["remissions"]
        tpoints = tpoints.reshape((-1))

        npoints = past["points"].shape[0]
        end = start + npoints
        assert(npoints == past["labels"].shape[0])
        concated_points[4 * start:4 * end] = tpoints
        concated_labels[start:end] = past["labels"]
        pc_ids[start:end] = i
        start = end

    return concated_points.reshape((-1, 4)), concated_labels, pc_ids


def postprocess(points, labels, cfg):
    with torch.no_grad():
        one_hot = torch.cuda.FloatTensor(labels.shape[0], 4).zero_()
        device = one_hot.device

        cuda_labels = torch.from_numpy(labels.astype(np.int32)).to(device, non_blocking=True)
        cuda_points = torch.from_numpy(points).to(device, non_blocking=True)

        one_hot[torch.arange(labels.shape[0]), cuda_labels.to(torch.long)] = 1

        postprocessor = BinningPostprocess(cfg, device)

        new_preds = postprocessor.process_pc(cuda_points, one_hot)
        assert(new_preds.shape[0] == points.shape[0])

        # Estimate and print the height of the Lidar
        if False:
            xy = cuda_points[:, :2]
            close = torch.linalg.norm(cuda_points[:, :2], dim=1) < 3.0
            close_ground = new_preds[close, :2].sum(axis=1).to(torch.bool)
            close_ground_z = cuda_points[close][close_ground][:, 2]
            if close_ground_z.shape[0] > 0:
                print(float(close_ground_z.min()),
                      float(close_ground_z.max()),
                      float(close_ground_z.mean()))

        _, new_labels = new_preds.max(axis=1)

        unknown = new_preds.sum(axis=1) == 0

        # unknown points are labeled as -1
        new_labels[unknown] = -1

    torch.cuda.empty_cache()
    return new_labels


def make_costmap_pose(pose, map_cfg):
    # scale
    inv_proj_mat = np.diag([map_cfg['gridw'],
                            map_cfg['gridh'],
                            1.0, 1.0])

    # shift
    inv_proj_mat[0, 3] = map_cfg['minx']
    inv_proj_mat[1, 3] = map_cfg['miny']

    costmap_pose = np.matmul(pose, inv_proj_mat)
    # remove z dimension
    costmap_pose = costmap_pose[[0, 1, 3]][:, [0, 1, 3]]
    return costmap_pose


def create_costmap(points, labels, cfg, pose=None,
                   pc_ids=None, key_scan_id=None,
                   force_return_cmap=False):
    # remove void
    mapped_labels = map_labels(labels, cfg["learning_map"])
    not_void = mapped_labels != 0
    mapped_labels = mapped_labels[not_void]

    map_cfg = cfg["costmap"]
    h = int(np.floor((map_cfg["maxy"] - map_cfg["miny"])/map_cfg["gridh"]))
    w = int(np.floor((map_cfg["maxx"] - map_cfg["minx"])/map_cfg["gridw"]))

    if mapped_labels.shape[0] == 0:
        if force_return_cmap:
            return np.full((h,w), 255, dtype=np.uint8), None, None
        else:
            return None, None, None

    points = points[not_void]
    mapped_labels = mapped_labels - 1  # valid labels 0,1,2,3

    assert (mapped_labels.min() >= 0 and mapped_labels.max() < 4)

    cu_labels = postprocess(points, mapped_labels, cfg["postprocessing"])

    # unknown and sky predictions are not valid
    valid = (cu_labels < 4) & (cu_labels >= 0)
    points = points[valid.cpu().numpy()]
    mapped_labels = cu_labels[valid].cpu().numpy()

    ## projection
    j_inds = np.floor((points[:, 0] - map_cfg["minx"]
                       ) / map_cfg["gridw"]).astype(np.int32)
    i_inds = np.floor((points[:, 1] - map_cfg["miny"]
                       ) / map_cfg["gridh"]).astype(np.int32)

    inrange = (i_inds >= 0) & (i_inds < h) & (j_inds >= 0) & (j_inds < w)

    i_inds, j_inds, mapped_labels = [x[inrange] for x in [i_inds, j_inds,
                                                          mapped_labels]]

    # Assume that cost of class j > cost of class i if j > i
    # Project the points down. Higher-cost classes overwrite low-cost classes.
    costmap = np.ones((h, w), np.uint8) * 255
    for l in [0, 1, 2, 3]:
        hist, _, _ = np.histogram2d(i_inds[mapped_labels == l],
                                    j_inds[mapped_labels == l],
                                    bins=(h, w),
                                    range=((0, h), (0, w)))
        costmap[hist != 0] = l

    #####
    costmap_pose = None
    if pose is not None:
        costmap_pose = make_costmap_pose(pose, map_cfg)

    #### extract postprocessed labels only for the key scan
    if pc_ids is not None and key_scan_id >= 0:
        key_labels_ = cu_labels[pc_ids[not_void] == key_scan_id]

        scan_len = (pc_ids == key_scan_id).sum()
        key_labels = np.full(scan_len, -1)

        key_labels[not_void[pc_ids == key_scan_id]] = key_labels_.cpu().numpy()

        # 0,1,2,3,4 are reserved so use 5 for unknown?
        key_labels[key_labels < 0] = 5

        return costmap, key_labels.astype(np.uint32), costmap_pose
    ####

    torch.cuda.empty_cache()

    return costmap, None, costmap_pose


@jit(nopython=True)
def make_top_down_map(points, mapped_labels, costmap, z_map, min_x, min_y, grid_w, grid_h, w, h):
    for i in range(len(points)):
        x, y, z = points[i, :3]
        grid_x = int(np.floor((x - min_x) / grid_w))
        grid_y = int(np.floor((y - min_y) / grid_h))
        if 0 <= grid_x < w and 0 <= grid_y < h and z > z_map[grid_y, grid_x]:
            costmap[grid_y, grid_x] = mapped_labels[i]
            z_map[grid_y, grid_x] = z


def create_topdown_semantic_map(points, labels, cfg, pose=None,
                                pc_ids=None, key_scan_id=None,
                                force_return_cmap=False):
    # remove void
    mapped_labels = map_labels(labels, cfg["learning_map"])
    not_void = mapped_labels != 0
    mapped_labels = mapped_labels[not_void]

    map_cfg = cfg["costmap"]
    h = int(np.floor((map_cfg["maxy"] - map_cfg["miny"])/map_cfg["gridh"]))
    w = int(np.floor((map_cfg["maxx"] - map_cfg["minx"])/map_cfg["gridw"]))

    if mapped_labels.shape[0] == 0:
        if force_return_cmap:
            return np.full((h, w), 255, dtype=np.uint8), None, None
        else:
            return None, None, None

    points = points[not_void]
    mapped_labels = mapped_labels - 1

    costmap = np.ones((h, w), np.uint8) * 255
    z_map = np.ones((h, w), np.float32) * -10
    make_top_down_map(points, mapped_labels, costmap, z_map,
                      map_cfg['minx'], map_cfg['miny'],
                      map_cfg['gridw'], map_cfg['gridh'],
                      w, h)

    costmap_pose = None
    if pose is not None:
        costmap_pose = make_costmap_pose(pose, map_cfg)

    return costmap, None, costmap_pose


def find_moving_objects(labels, moving_object_labels):
    mask = None  # moving object mask
    for l in moving_object_labels:
        if mask is not None:
            mask = np.logical_or(mask, labels == l)
        else:
            mask = (labels == l)
    return mask


def remove_moving_objects(history, moving_object_labels, key_id):
    def helper(data):
        mask = find_moving_objects(data["labels"] & 0xFFFF,
                                   moving_object_labels)
        mask = np.logical_not(mask)
        return {
            "scan": data["scan"][mask],
            "points": data["points"][mask],
            "labels": data["labels"][mask],
            "remissions": data["remissions"][mask],
            "pose": data["pose"],
            "filename": data["filename"]
        }

    new_history = []
    for i in range(len(history)):
        if i == key_id:
            # Skip key scan
            new_history.append(history[i])
        else:
            new_history.append(helper(history[i]))
    return new_history


def compute_convex_hull(costmap: np.ndarray):
    """
    Compute the convex hull of all labeled points.
    Args:
        costmap: H x W integer label map

    Returns:
        a bool numpy array of H x W containing the convex hull of the labeled points
    """
    ys, xs = np.nonzero(costmap != 255)  # 255 is unknown
    points = cv2.convexHull(np.stack([xs, ys], axis=-1))  # N_POINTS x 1 x 2
    points = points.transpose((1, 0, 2))
    mask = np.zeros(costmap.shape, np.uint8)
    cv2.drawContours(mask, points, -1, 255, thickness=-1)
    if False:  # Visualize
        foreground = (costmap != 255).astype(np.uint8) * 255
        viz = np.concatenate([foreground, mask], axis=1)
        cv2.imshow('', viz)
        cv2.waitKey(0)
    return mask.astype(np.bool)


def project_semantic_voxels(voxel_grid):
    projected = np.zeros(voxel_grid.shape[:2], np.uint16)
    for zi in range(voxel_grid.shape[2] - 1, -1, -1):
        not_filled = (projected == 0)
        layer = voxel_grid[:, :, zi]
        valid_mask = np.logical_and(layer != 0, layer != 1)
        valid_labels = layer * valid_mask.astype(layer.dtype) * not_filled.astype(layer.dtype)
        projected = np.bitwise_or(projected, valid_labels)
    return projected


def fill(data, invalid):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'.
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)

    Output:
        Return a filled array.
    """
    ind = ndimage.distance_transform_edt(invalid,
                                         return_distances=False,
                                         return_indices=True)
    return data[tuple(ind)]


def map_class_labels(voxel_grid, learning_map):
    u, inv = np.unique(voxel_grid, return_inverse=True)
    return np.array([learning_map[x] for x in u])[inv].reshape(voxel_grid.shape)


def project_voxels_to_bev_costmap(voxel_grid, cfg):
    # class 0 is unlabeled
    # class 1 is free
    # class 2 is low-lost
    # class 3 is lethal
    voxel_grid = map_class_labels(voxel_grid, cfg['learning_map'])

    free_voxels = (voxel_grid == 1)
    low_cost_voxels = (voxel_grid == 2)
    free_region = np.logical_or.reduce(free_voxels, axis=-1)
    low_cost_region = np.logical_or.reduce(low_cost_voxels, axis=-1)

    traversable_voxels = free_voxels | low_cost_voxels
    ground_z = np.argmax(traversable_voxels, axis=-1).astype(np.uint8)
    ground_z = ndimage.uniform_filter(ground_z, 7)

    valid_ground = np.logical_or(free_region, low_cost_region)
    ground_z = fill(ground_z, np.logical_not(valid_ground))
    # cv2.imshow('traversable', valid_ground.astype(np.uint8) * 255)
    # cv2.imshow('ground', np.clip(ground_z * 20, 0, 255).astype(np.uint8))

    max_height_thres = int(cfg['max_height'] / cfg['resolution'])
    min_height_thres = int(cfg['min_height'] / cfg['resolution'])

    costmap = np.zeros(voxel_grid.shape[:2], np.uint8)
    costmap[free_region] = 1
    costmap[low_cost_region] = 2

    # Iterate over the z axis to compute the cost classes.
    # Assume that the cost of class j > the cost of class i if j > i
    # class 0 is 'unlabeled' so that it has the lowest priority
    for i in range(voxel_grid.shape[-1]):
        in_range = np.logical_and(ground_z <= i - min_height_thres,
                                  ground_z >= i - max_height_thres)
        costmap[in_range] = np.maximum(costmap[in_range], voxel_grid[:, :, i][in_range])
    # cv2.imshow('after', costmap.astype(np.uint8) * 80)
    # cv2.waitKey(0)
    return costmap


def make_color_map(cfg):
    semantic_cmap = np.zeros((256, 3), np.uint8)
    for _ in range(256):
        l = _ + 1  # Add 1 to get the correct kitti label.
        if l in cfg['learning_map_inv']:
            bgr = cfg['color_map'][cfg['learning_map_inv'][l]]
            rgb = bgr[::-1]
            semantic_cmap[_] = rgb
    semantic_cmap[255] = (255, 250, 230)
    return semantic_cmap
