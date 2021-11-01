import json
import glob
import os

import cv2
import parse
import numpy as np
import tabulate
import torch
from torch.utils.tensorboard import SummaryWriter
from bevnet import networks

def build_nets(config, device):
    ret = {}
    for net_name, spec in config.items():
        net_class = getattr(networks, spec['class'])
        net_args = spec.get('net_kwargs', {})
        net = net_class(**net_args).to(device)
        if spec['opt'] != 'none':
            ret[net_name] = {
                'net': net,
                'opt': getattr(torch.optim, spec['opt'])(net.parameters(), **spec['opt_kwargs'])
            }
        else:
            # handles parameter-less networks e.g. VoxelFeatureEncoderV3
            ret[net_name] = {
                'net': net,
                'opt': None
            }
    return ret


def numpify_criterion_meta_data(md):
    ''' md is a dictionary of tensor values with shape [bx...]
        We make a list of b dictionary with the same key and numpy array
        shape of [...].
    '''
    b = None
    pmd = []
    for k,v in md.items():
        if b is None:
            b = v.shape[0]
            pmd = [dict() for _ in range(b)]
        else:
            assert(b == v.shape[0])

        assert(isinstance(v, torch.Tensor))

        for m, vu in zip(pmd, v.unbind()):
            m[k] = vu.detach().cpu().numpy()

    return pmd


def get_last_model(dir_fn, prefix='model.pth.'):
    files = os.listdir(dir_fn)
    matched = [x for x in files if x.startswith(prefix)]

    if matched:
        def id(fn):
            fn = fn[len(prefix):]
            try:
                return int(fn)
            except:
                return fn

        last_model = os.path.join(dir_fn, sorted(matched, key=lambda fn: id(fn))[-1])
        return last_model
    else:
        raise Exception('No valid model in {}'.format(dir_fn))


def batch_create_traj_mask(h, w, traj_list, thickness, weight=1.0):
    '''
        traj_list is a list of b traj numpy array each with size kx2 in xy format.
    '''
    masks = []
    for traj in traj_list:
        m = create_traj_mask(h, w, traj, thickness, weight)
        masks.append(m)

    return np.array(masks)

def create_traj_mask(h, w, traj, thickness, weight=1.0):
    '''
        traj: np.int32 (Nx2) array. Each point traj[i] specify a *xy*
        location on the trajectory.
        weight: weight of the trajectory
    '''

    return cv2.polylines(np.zeros((h, w), dtype=np.float32),
                         traj.reshape((1, -1, 1, 2)),
                         isClosed=False,
                         color=(weight,),
                         thickness=thickness)

def get_label_generator(loss_config, include_unknown, num_class):
    if 'label_generator' not in loss_config:

        ## No processing just use labels
        def label_generator(**kwargs):
            labels = kwargs['labels']
            if include_unknown:
                # Note that `num_class` already includes the unknown label
                labels[labels == 255] = num_class - 1

            # no meta_data
            meta_data = None
            return kwargs['labels'].long(), meta_data
    else:
        lg_config = loss_config.label_generator
        if lg_config.name == 'start2goal':

            # Generate start to goal trajectory as label
            def label_generator(**kwargs):
                traj = kwargs['trajectory']
                # Not necessary to have it but makes things easier
                labels = kwargs['labels']
                _, h, w = labels.shape
                thickness = lg_config.get('traj_thickness', 1)
                traj_labels_np = batch_create_traj_mask(h, w, traj, thickness)

                # send out start and goal in yx format
                start = torch.tensor([t[0, (1,0)] for t in traj], device=labels.device)
                goal = torch.tensor([t[-1, (1,0)] for t in traj], device=labels.device)
                meta_data = dict(start=start, goal=goal)

                return torch.from_numpy(traj_labels_np).long().to(labels.device), meta_data

        else:
            raise NotImplementedError("Label generator {} is not supported.".format(
                lg_config.name))

    return label_generator

def get_loss_function(loss_config, class_weights, include_unknown):

    if loss_config.name == "VIPlanner": ## MEDIRL Loss
        vi_planner = VIPlanner(loss_config.planner)

        def _vi_planner_loss(pred, labels, meta_data):
            h,w = pred.shape[-2:]
            assert(h == w), 'You need to do clip for y and x coordinates seperately.'

            ## make sure start and goal are within the image space
            ## start and goal are in yx format
            start = torch.clip(meta_data['start'], 0, h - 1).to(torch.int64)
            goal = torch.clip(meta_data['goal'], 0, h - 1).to(torch.int64)

            # reward = fg - bg
            # Do not like this? Make sure everything works fine with num_class = 1
            # I don't see if it can be any issue
            reward = pred[:, 1] - pred[:, 0]
            #print(reward.min().item(), reward.max().item())
            loss = vi_planner.loss(reward, labels, start, goal)
            return loss, loss.meta_data

        return _vi_planner_loss

    if loss_config.name == "CrossEntropy":
        if include_unknown:
            ignore_idx = -100
        else:
            ignore_idx = 255

        if 'mask' in loss_config: # Masked CE loss
            mask_config = loss_config.mask
            if mask_config.type == "gaussian_blur":
                kernel = torchvision.transforms.GaussianBlur(mask_config.kernel_size,
                                                           mask_config.sigma)
            elif mask_config.type == 'max_pool':
                assert(mask_config.kernel_size % 2)
                pad_size = mask_config.kernel_size // 2
                kernel = torch.nn.MaxPool2d(mask_config.kernel_size,
                                            stride=1,
                                            padding=pad_size)

            else:
                raise NotImplementedError()

            loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_idx,
                                             weight=class_weights)

            mask_scale = mask_config.get('scale', 1.0)

            def _masked_loss(pred, labels, meta_data):
                bool_mask = (labels == mask_config.fg_label)
                float_mask = bool_mask.to(pred.dtype)
                with torch.no_grad():
                    pixel_weights = kernel(float_mask[:, None])[:, 0]
                    pixel_weights[bool_mask] = 1.0
                    pixel_weights = torch.clip(pixel_weights*mask_scale, 0.0, 1.0)


                #from pylab import show, imshow, figure
                #labels[labels == 255] = 0
                #imshow(labels[0].cpu())
                #figure()
                #imshow(pixel_weights[0].cpu())
                #show()

                loss_val = loss(pred, labels)
                meta_data = None
                return (loss_val * pixel_weights).sum() / pixel_weights.sum(), meta_data

            return _masked_loss

        else:   ## CE loss
            cs_loss = torch.nn.CrossEntropyLoss(reduction="mean",
                                                ignore_index=ignore_idx,
                                                weight=class_weights)
            def _ce_loss(pred, labels, meta_data):
                loss = cs_loss(pred, labels)
                meta_data = None
                return loss, meta_data

            return _ce_loss
           
    raise NotImplementedError("Loss {} is not supported.".format(loss_config.name))

def get_criterion(g):
    loss_config = g.loss

    if 'class_weights' in loss_config:
        class_weights = torch.tensor(loss_config.class_weights).to(g.train_device)
        assert len(class_weights) == g.num_class
    else:
        class_weights = None

    label_generator = get_label_generator(loss_config, g.include_unknown, g.num_class)
    loss_function = get_loss_function(loss_config, class_weights, g.include_unknown)


    def criterion(pred, **kwargs):
        labels, lg_meta_data = label_generator(**kwargs)
        loss, l_meta_data = loss_function(pred, labels, lg_meta_data)

        ## Aggregate meta data
        meta_data = dict()
        for x in [lg_meta_data, l_meta_data]:
            if x is not None:
                meta_data.update(x)

        return labels, loss, meta_data

    return criterion

def load_nets(model_file, nets, net_opts, tolerant):
    state = load_state_dict(os.path.dirname(model_file),
                            os.path.basename(model_file), load_to_cpu=True)
    epoch = int(state['epoch'])

    for name, net in nets.items():
        load_state_helper(net, state['nets'][name], tolerant)

    # TODO: currently don't load net_opts, but we might want to do that later.
    return epoch


def save_nets(nets, net_opts, epoch, global_args, model_file):
    state = {
        'epoch': epoch,
        'global_args': global_args,
        'optims': {
            name: opt.state_dict() for name, opt in net_opts.items()
        },
        'nets': {
            name: net.state_dict() for name, net in nets.items()
        }
    }
    save_state_dict(state, epoch, '', model_file)


def save_state_dict(state, step, dir, filename):
    path = os.path.join(dir, '%s.%d' % (filename, step))
    print('Saving step={} model in {}'.format(step, path))
    torch.save(state, path)


def load_state_dict(dir, filename, step=None, load_to_cpu=False):
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

def log_dict(writer: SummaryWriter, tag, d, step):
    text = ''.join(['\t' + _ for _ in json.dumps(d, indent=4).splitlines(True)])
    writer.add_text(tag, text, global_step=step)
