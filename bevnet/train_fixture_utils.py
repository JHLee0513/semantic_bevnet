import json
import glob
import os

import cv2
import parse
import numpy as np
import tabulate
import torch
from torch.utils.tensorboard import SummaryWriter

from .colormaps import get_colormap


def save_model(state, step, dir, filename):
    path = os.path.join(dir, '%s.%d' % (filename, step))
    torch.save(state, path)


def load_model(dir, filename, step=None, load_to_cpu=False):
    '''
    :param model:
    :param dir:
    :param filename:
    :param step: if None. Load the latest.
    :return: the saved state dict
    '''
    if not step:
        files = glob.glob(os.path.join(dir, '%s.*' % filename))
        if len(files) == 0:
            files = [os.path.join(dir, filename)]

        parsed = []
        for fn in files:
            r = parse.parse('{}.{:d}', fn)
            if r:
                parsed.append((r, fn))
        if not parsed:
            return None

        step, path = max(parsed, key=lambda x: x[0][1])
    else:
        path = os.path.join(dir, '%s.%d' % (filename, step))

    if os.path.isfile(path):
        if load_to_cpu:
            return torch.load(path, map_location=lambda storage, location: storage)
        else:
            return torch.load(path)

    raise Exception('Failed to load model')


def load_state_helper(module, state_dict, tolerant):
    if tolerant:
        strict = True
        for name, para in module.named_parameters():
            if para.requires_grad:
                if name not in state_dict:
                    strict = False
                elif para.shape != state_dict[name].shape:
                    state_dict.pop(name)
                    strict = False

        missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=strict)
        if not strict:
            if missing_keys:
                print('Skipped parameters {}'.format(missing_keys))
            if unexpected_keys:
                print('Unexpected parameters {}'.format(unexpected_keys))
    else:
        module.load_state_dict(state_dict)


def make_train(nets):
    for module in nets.values():
        module.train()


def make_eval(nets):
    for module in nets.values():
        module.eval()


def set_device(nets, device):
    for module in nets.values():
        module.to(device)


def parameters(nets):
    params = []
    for net in nets.values():
        params += list(net.parameters())
    return params


def module_grad_stats(module):
    headers = ['layer', 'max', 'min']

    def maybe_max(x):
        return x.max() if x is not None else 'None'

    def maybe_min(x):
        return x.min() if x is not None else 'None'

    data = [
        (name, maybe_max(param.grad), maybe_min(param.grad))
        for name, param in module.named_parameters()
    ]
    return tabulate.tabulate(data, headers, tablefmt='psql')


def module_weights_stats(module):
    headers = ['layer', 'max', 'min']

    def maybe_max(x):
        return x.max() if x is not None else 'None'

    def maybe_min(x):
        return x.min() if x is not None else 'None'

    data = [
        (name, maybe_max(param), maybe_min(param))
        for name, param in module.named_parameters()
    ]
    return tabulate.tabulate(data, headers, tablefmt='psql')


def make_label_vis(label, cmap):
    """
    Args:
        label: H x W integer label
    Returns:
    """
    vis = np.zeros(label.shape + (3,), np.uint8)
    for k, v in cmap.items():
        vis[label == k] = v
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)


def visualize_predictions(writer, pred, gt, dataset_type):
    """
    Args:
        pred: N x N_CLASS x H x W
        gt:  N x H x W

    Returns:
    """
    n = gt.shape[0]
    cmap = get_colormap(dataset_type)
    gt_vis = np.zeros(gt.shape + (3,), dtype=np.uint8)
    for i in range(n):
        gt_vis[i] = make_label_vis(gt[i], cmap)

    pred_vis = np.zeros(gt_vis.shape, dtype=np.uint8)
    for i in range(n):
        pred_vis[i] = make_label_vis(np.argmax(pred[i], axis=0), cmap)

    writer.add_images('label', gt_vis, global_step=0, dataformats='NHWC')
    writer.add_images('pred', pred_vis, global_step=0, dataformats='NHWC')


def log_dict(writer: SummaryWriter, tag, d, step):
    text = ''.join(['\t' + _ for _ in json.dumps(d, indent=4).splitlines(True)])
    writer.add_text(tag, text, global_step=step)
