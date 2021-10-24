import argparse

import torch
import yaml
from easydict import EasyDict
from torch import optim

from bevnet import networks
from bevnet import train_fixture
from bevnet.args import add_common_arguments


def build_nets(config, device):
    ret = {}
    for net_name, spec in config.items():
        net_class = getattr(networks, spec['class'])
        net_args = spec.get('net_kwargs', {})
        net = net_class(**net_args).to(device)
        if spec['opt'] != 'none':
            ret[net_name] = {
                'net': net,
                'opt': getattr(optim, spec['opt'])(net.parameters(), **spec['opt_kwargs'])
            }
        else:
            # handles parameter-less networks e.g. VoxelFeatureEncoderV3
            ret[net_name] = {
                'net': net,
                'opt': None
            }
    return ret


if __name__ == "__main__":
    torch.manual_seed(2021)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    add_common_arguments(parser)

    train_funcs = {
      'default': train_fixture.train_single,
    }

    g = EasyDict(vars(parser.parse_args()))
    dataset_cfg = yaml.load(open(g.dataset_config).read(), Loader=yaml.SafeLoader)
    g.update(dataset_cfg)

    model_spec = yaml.load(open(g.model_config).read(), Loader=yaml.SafeLoader)
    ret = build_nets(model_spec, g.train_device)
    g.model_config = model_spec

    nets = {
        name: spec['net'] for name, spec in ret.items()
    }
    net_opts = {
        name: spec['opt'] for name, spec in ret.items() if spec['opt'] is not None
    }

    train_funcs[g.model_variant](nets, net_opts, g)
