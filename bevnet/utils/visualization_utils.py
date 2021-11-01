import numpy as np
import torch
import cv2

def overlay(a, b, mask, alpha=0.25):
    c = (1-alpha) * a + alpha * b
    c[mask == 0] = a[mask == 0]
    return c.astype(a.dtype)

def pointcloud_bev_occupancy_map(coords, shape):
    mp = np.zeros(shape, dtype=np.uint8)
    mp[coords[:, 0], coords[:, 1]] = 255
    return np.tile(np.array(mp[..., None]), (1, 1, 3))


def visualize(pred, label, dataset_type,
        points=None, voxelizer=None, coords=None,
        criterion_meta_data=None):

    label_vis = visualize_label(label, dataset_type)

    scan_img_blue = None
    if points is not None:
        assert(voxelizer is not None)
        # Add an additional channel storing the point indices
        points_with_idx = np.concatenate([
                points, np.arange(len(points))[:, None].astype(points.dtype)], axis=-1)
        coords = voxelizer.generate(points_with_idx, max_voxels=90000)[1]


    if coords is not None:
        scan_img = pointcloud_bev_occupancy_map(coords[:, 1:], label.shape)
        scan_img_blue = scan_img.copy()
        scan_img_blue[..., :2] = 0
        scan_mask = scan_img[..., 0] != 0

        label_vis = overlay(label_vis, scan_img_blue, scan_mask, alpha=0.8)


    ## prepare prediction
    pred_vis = visualize_pred(pred, dataset_type)

    if dataset_type == 'heatmap':
        pred_vis = overlay(label_vis, pred_vis, mask=None, alpha=0.7)
    else:
        if scan_img_blue is not None:
            pred_vis = overlay(pred_vis, scan_img_blue, scan_mask, alpha=0.8)


    out = [label_vis, pred_vis]

    if criterion_meta_data is not None:
        for k, v in criterion_meta_data.items():
            ## Only visualize 2D arrays with correct image size
            if v.shape == pred_vis.shape[:2]:
                im = np.tile(np.array(v[..., None]), (1, 1, 3))
                im = (im - im.min()) / (im.max() - im.min())
                im = (255*im).astype(np.uint8)
                out.append(im)

    vis = np.concatenate(out, 1)
    
    return vis 


############ colormaps ####
def visualize_label(label, dataset_type):
    assert(label.ndim == 2) #HW

    if dataset_type == 'heatmap':
        return image_from_colormap(label, get_colormap('costmap_4'))
    if dataset_type in _color_maps:
        return image_from_colormap(label, get_colormap(dataset_type))

    ## Need to be two class bg/fg prediction

def visualize_pred(pred, dataset_type):
    assert(pred.ndim == 3) #CHW

    if dataset_type == 'heatmap':
        ## Need to be two class bg/fg prediction
        assert(pred.shape[0] == 2) #CHW
        pred = pred[0]
        assert(np.all((0 <= pred) & (pred <= 1)))
        pred = (pred * 255).astype(np.uint8)
        return np.tile(np.array(pred[..., None]), (1, 1, 3))  

    if dataset_type in _color_maps:
        return image_from_colormap(np.argmax(pred, axis=0), 
                get_colormap(dataset_type))

def image_from_colormap(label, cmap):
    """
    Args:
        label: H x W integer label
    Returns:
    """
    vis = np.zeros(label.shape + (3,), np.uint8)
    for k, v in cmap.items():
        vis[label == k] = v
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


# BGR colors
kitti_19class = {
        255: (255, 255, 255),
        0: (245, 150, 100),
        1: (245, 230, 100),
        2: (150, 60, 30),
        3: (180, 30, 80),
        4: (255, 0, 0),
        5: (30, 30, 255),
        6: (200, 40, 255),
        7: (90, 30, 150),
        8: (255, 0, 255),
        9: (255, 150, 255),
        10: (75, 0, 75),
        11: (75, 0, 175),
        12: (0, 200, 255),
        13: (50, 120, 255),
        14: (0, 175, 0),
        15: (0, 60, 135),
        16: (80, 240, 150),
        17: (150, 240, 255),
        18: (0, 0, 255),
        19: (255, 255, 255)} # Optional unknown class


costmap_4class = {
        0: (0, 255, 0),
        1: (0, 255, 255),
        2: (255, 0, 0),
        3: (0, 0, 255),
        4: (230, 250, 255),  # Optional unknown class, beige for consistency with generated data
        255: (230, 250, 255)}


costmap_4class_2 = {
        0: (0, 255, 0),
        1: (0, 255, 255),
        2: (0, 122, 255),
        3: (0, 0, 255),
        4: (230, 250, 255),  # Optional unknown class, beige for consistency with generated data
        255: (230, 250, 255)}


kitti_3class = {
        0: (0, 255, 0),
        1: (0, 255, 255),
        2: (0, 0, 255),
        255: (255, 255, 255)}


_color_maps = {
    'kitti_19': kitti_19class,
    'kitti_3': kitti_3class,
    'costmap_4': costmap_4class,
    'costmap_4_2': costmap_4class_2,
}


def get_colormap(dataset_type):
    return _color_maps[dataset_type]
