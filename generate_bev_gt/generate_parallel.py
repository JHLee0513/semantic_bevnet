import argparse
import os
from collections import deque
import time
import yaml
from PIL import Image
import multiprocessing
from copy import deepcopy
import numpy as np
import tqdm
import torch
from common import parse_calibration, parse_poses, make_color_map


# This is called inside the worker process.
def init(queue):
    global device
    device = queue.get()


def gen_costmap(kwargs, mem_frac):
    from common import (
        remove_moving_objects,
        join_pointclouds,
        create_costmap,
        compute_convex_hull,
    )

    global device
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(mem_frac)

    cfg, cmap, scan_files, label_files, poses = [kwargs[_] for _ in [
        'cfg', 'cmap', 'scan_files', 'label_files', 'poses'
    ]]

    assert len(scan_files) == len(label_files)
    assert len(scan_files) == len(poses)

    history = deque()

    for i in range(len(scan_files)):
        scan = np.fromfile(scan_files[i], dtype=np.float32)
        scan = scan.reshape((-1, 4))
        labels = np.fromfile(label_files[i], dtype=np.uint32)
        labels = labels.reshape((-1))

        # convert points to homogenous coordinates (x, y, z, 1)
        points = np.ones((scan.shape))
        points[:, 0:3] = scan[:, 0:3]
        remissions = scan[:, 3]

        scan_ground_frame = scan.copy()
        scan_ground_frame[:, 2] += cfg.get('lidar_height')

        # append current data to history queue.
        history.appendleft({
            "scan": scan.copy(),
            "scan_ground_frame": scan_ground_frame,
            "points": points,
            "labels": labels,
            "remissions": remissions,
            "pose": poses[i].copy(),
            "filename": scan_files[i]
        })

    key_scan_id = len(history) // 2
    key_scan = history[key_scan_id]

    new_history = remove_moving_objects(history,
                                        cfg["moving_classes"],
                                        key_scan_id)

    cat_points, cat_labels, pc_ids = join_pointclouds(new_history, key_scan["pose"])

    # Create costmap
    (costmap, key_scan_postprocess_labels,
     costmap_pose) = create_costmap(cat_points, cat_labels, cfg,
                                    pose=key_scan["pose"],
                                    pc_ids=pc_ids,
                                    key_scan_id=key_scan_id)

    if cfg.get('convex_hull', False):
        cvx_hull = compute_convex_hull(costmap)
        costmap[np.logical_not(cvx_hull)] = 4

    costimg = Image.fromarray(costmap, mode='P')
    costimg.putpalette(cmap)

    # Just the current snapshot
    points_1step, labels_1step, _ = join_pointclouds([key_scan], key_scan["pose"])
    costmap_1step, _, _ = create_costmap(points_1step, labels_1step, cfg,
                                         force_return_cmap=True)

    costimg_1step = Image.fromarray(costmap_1step, mode='P')
    costimg_1step.putpalette(cmap)

    return {
        'scan_ground_frame': key_scan['scan_ground_frame'],
        'labels': key_scan['labels'],
        'postprocessed_labels': key_scan_postprocess_labels,
        'costimg': costimg,
        'costimg_1step': costimg_1step,
        'pose': poses[key_scan_id],
        'costmap_pose': costmap_pose,
    }

def gen_voxel_costmap(kwargs, mem_frac):
    from common import (
        remove_moving_objects,
        join_pointclouds,
        create_costmap,
        compute_convex_hull,
    )

    global device
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(mem_frac)

    cfg, cmap, scan_files, label_files, poses = [kwargs[_] for _ in [
        'cfg', 'cmap', 'scan_files', 'label_files', 'poses'
    ]]

    assert len(scan_files) == len(label_files)
    assert len(scan_files) == len(poses)

    history = deque()

    for i in range(len(scan_files)):
        scan = np.fromfile(scan_files[i], dtype=np.float32)
        scan = scan.reshape((-1, 4))
        labels = np.fromfile(label_files[i], dtype=np.uint32)
        labels = labels.reshape((-1))

        # convert points to homogenous coordinates (x, y, z, 1)
        points = np.ones((scan.shape))
        points[:, 0:3] = scan[:, 0:3]
        remissions = scan[:, 3]

        scan_ground_frame = scan.copy()
        scan_ground_frame[:, 2] += cfg.get('lidar_height')

        # append current data to history queue.
        history.appendleft({
            "scan": scan.copy(),
            "scan_ground_frame": scan_ground_frame,
            "points": points,
            "labels": labels,
            "remissions": remissions,
            "pose": poses[i].copy(),
            "filename": scan_files[i]
        })

    key_scan_id = len(history) // 2
    key_scan = history[key_scan_id]

    new_history = remove_moving_objects(history,
                                        cfg["moving_classes"],
                                        key_scan_id)

    cat_points, cat_labels, pc_ids = join_pointclouds(new_history, key_scan["pose"])

    # Create costmap
    (costmap, key_scan_postprocess_labels,
     costmap_pose) = create_costmap(cat_points, cat_labels, cfg,
                                    pose=key_scan["pose"],
                                    pc_ids=pc_ids,
                                    key_scan_id=key_scan_id)

    if cfg.get('convex_hull', False):
        cvx_hull = compute_convex_hull(costmap)
        costmap[np.logical_not(cvx_hull)] = 4

    costimg = Image.fromarray(costmap, mode='P')
    costimg.putpalette(cmap)

    # Just the current snapshot
    points_1step, labels_1step, _ = join_pointclouds([key_scan], key_scan["pose"])
    costmap_1step, _, _ = create_costmap(points_1step, labels_1step, cfg,
                                         force_return_cmap=True)

    costimg_1step = Image.fromarray(costmap_1step, mode='P')
    costimg_1step.putpalette(cmap)

    return {
        'scan_ground_frame': key_scan['scan_ground_frame'],
        'labels': key_scan['labels'],
        'postprocessed_labels': key_scan_postprocess_labels,
        'costimg': costimg,
        'costimg_1step': costimg_1step,
        'pose': poses[key_scan_id],
        'costmap_pose': costmap_pose,
    }


if __name__ == '__main__':
    def run():
        start_time = time.time()

        parser = argparse.ArgumentParser("./generate_parallel.py")

        parser.add_argument(
            '--config',
            '-c',
            required=True,
            help='path to the config file')

        parser.add_argument(
            '--n_worker',
            type=int,
            required=True,
            help='Number of workers.')

        parser.add_argument(
            '--devices',
            type=str,
            required=False,
            default='cuda',
            help='A comma-separated list of cuda devices.'
        )

        parser.add_argument(
            '--mem_frac',
            type=float,
            required=True,
            help='GPU memory fraction per worker. Used to avoid out-of-memory issue.')

        FLAGS, unparsed = parser.parse_known_args()

        with open(FLAGS.config, 'r') as stream:
            cfg = yaml.safe_load(stream)

        FLAGS.dataset = cfg["input"]
        FLAGS.output = cfg["output"]

        sequence_length = cfg["sequence_length"]
        stride = cfg.get('stride', 1)

        sequences_dir = os.path.join(FLAGS.dataset, "sequences")
        cmap = make_color_map(cfg)

        for split in cfg["split"]:
            sequence_folders = cfg["split"][split]

            # Output directories
            output_folder = os.path.join(FLAGS.output, "sequences", split)
            velodyne_folder = os.path.join(output_folder, "velodyne")
            velodyne_labels_folder = os.path.join(output_folder, "labels")
            velodyne_pp_labels_folder = os.path.join(output_folder, "postprocessed_labels")
            labels_folder = os.path.join(output_folder, "bev_labels")
            labels_1step_folder = os.path.join(output_folder, "bev_1step_labels")

            os.makedirs(output_folder, exist_ok=True)
            os.makedirs(velodyne_folder, exist_ok=True)
            os.makedirs(velodyne_labels_folder, exist_ok=True)
            os.makedirs(velodyne_pp_labels_folder, exist_ok=True)
            os.makedirs(labels_folder, exist_ok=True)
            os.makedirs(labels_1step_folder, exist_ok=True)

            counters = dict()
            all_poses = []
            all_costmap_poses = []

            counter = 0

            for folder in sequence_folders:
                input_folder = os.path.join(sequences_dir, folder)
                scan_files = [
                    f for f in sorted(os.listdir(os.path.join(input_folder,
                                                              'velodyne')))
                    if f.endswith(".bin")
                ]

                calibration = parse_calibration(os.path.join(input_folder, 'calib.txt'))
                poses = parse_poses(os.path.join(input_folder, 'poses.txt'), calibration)

                start_counter = counter
                print("Processing {} ".format(folder), end="", flush=True)

                job_args = []

                for i in range(len(scan_files)):
                    start_idx = i - (sequence_length // 2 * stride)
                    history = []

                    data_idxs = [start_idx + _ * stride for _ in range(sequence_length)]
                    if data_idxs[0] < 0 or data_idxs[-1] >= len(scan_files):
                        # Out of range
                        continue

                    for data_idx in data_idxs:
                        scan_file = scan_files[data_idx]
                        basename = os.path.splitext(scan_file)[0]
                        scan_path = os.path.join(input_folder, "velodyne", scan_file)
                        label_path = os.path.join(input_folder, "labels", basename + ".label")
                        history.append((scan_path, label_path, poses[data_idx]))
                    history = history[::-1]

                    if len(history) < sequence_length:
                        continue
                    assert len(history) == sequence_length

                    hist_scan_files, hist_label_files, hist_poses = zip(*list(history))
                    job_args.append({
                        'cfg': cfg,
                        'cmap': cmap,
                        'scan_files': hist_scan_files,
                        'label_files': hist_label_files,
                        'poses': hist_poses,
                    })

                devices = FLAGS.devices.split(',')
                manager = multiprocessing.Manager()
                worker_init_queue = manager.Queue()
                for i in range(FLAGS.n_worker):
                    worker_init_queue.put(devices[i % len(devices)])

                ctx = multiprocessing.get_context('spawn')
                with ctx.Pool(FLAGS.n_worker, initializer=init, initargs=(worker_init_queue,)) as pool:
                    async_results = [pool.apply_async(gen_costmap, (job, FLAGS.mem_frac)) for job in job_args]
                    for future in tqdm.tqdm(async_results):
                        ret = future.get()
                        ret['scan_ground_frame'].tofile(
                            os.path.join(velodyne_folder, '{:05d}.bin'.format(counter)))
                        ret["labels"].tofile(
                            os.path.join(velodyne_labels_folder, '{:05d}.label'.format(counter)))
                        ret['postprocessed_labels'].tofile(
                            os.path.join(velodyne_pp_labels_folder, '{:05d}.label'.format(counter)))
                        ret['costimg'].save(
                            os.path.join(labels_folder, "{:05d}.png".format(counter)))
                        ret['costimg_1step'].save(
                            os.path.join(labels_1step_folder, "{:05d}.png".format(counter)))

                        all_poses.append(ret['pose'][:3])
                        all_costmap_poses.append(ret['costmap_pose'][:2])
                        counter += 1

                    counters[folder] = [start_counter, counter]

            # Save metadatas.
            yaml.dump(counters, open(os.path.join(output_folder, 'counters.yaml'), 'w'))
            # Flatten 4x4 matrix to 1D
            if len(all_poses) > 0:
                all_poses = np.array(all_poses)
                all_poses = all_poses.reshape((all_poses.shape[0], -1))
                np.savetxt(os.path.join(output_folder, 'poses.txt'), all_poses, fmt='%.8e')

            if len(all_costmap_poses) > 0:
                all_costmap_poses = np.array(all_costmap_poses)
                all_costmap_poses = all_costmap_poses.reshape((all_costmap_poses.shape[0], -1))
                np.savetxt(os.path.join(output_folder, 'costmap_poses.txt'),
                           all_costmap_poses, fmt='%.8e')

        print("execution time: {}".format(time.time() - start_time))

    run()
