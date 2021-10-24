# BGR colors
kitti_19class = {255: (255, 255, 255), 0: (245, 150, 100), 1: (245, 230, 100), 2: (150, 60, 30), 3: (180, 30, 80),
                 4: (255, 0, 0), 5: (30, 30, 255), 6: (200, 40, 255), 7: (90, 30, 150), 8: (255, 0, 255),
                 9: (255, 150, 255), 10: (75, 0, 75), 11: (75, 0, 175), 12: (0, 200, 255), 13: (50, 120, 255),
                 14: (0, 175, 0), 15: (0, 60, 135), 16: (80, 240, 150), 17: (150, 240, 255), 18: (0, 0, 255)}


costmap_4class = {0: (0, 255, 0), 1: (0, 255, 255), 2: (255, 0, 0), 3: (0, 0, 255),
                  4: (230, 250, 255),  # Optional unknown class, beige for consistency with generated data
                  255: (230, 250, 255)}


costmap_4class_2 = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 122, 255), 3: (0, 0, 255),
                  4: (230, 250, 255),  # Optional unknown class, beige for consistency with generated data
                  255: (230, 250, 255)}


kitti_3class = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255), 255: (255, 255, 255)}


_color_maps = {
    'kitti_19': kitti_19class,
    'kitti_3': kitti_3class,
    'costmap_4': costmap_4class,
    'costmap_4_2': costmap_4class_2,
}


def get_colormap(dataset_type):
    return _color_maps[dataset_type]
