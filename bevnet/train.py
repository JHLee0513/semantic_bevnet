import argparse

import torch
import yaml
from easydict import EasyDict

from bevnet.args import add_common_arguments


import os

import numpy as np
import tabulate
from spconv.utils import VoxelGenerator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from bevnet.utils import train_utils as tu
from bevnet.utils import visualization_utils as vis_utils
from bevnet.utils import eval_utils, file_utils
from bevnet.dataset import data_loader
import cv2

def forward_nets(nets, x, batch_size):
    bev_features = []
    keep_tensors = dict()
    if 'VoxelFeatureEncoder' in nets:
        # forward 3 modules: VoxelFeatureEncoder, MiddleSparseEncoder, BEVClassifier
        voxels = nets['VoxelFeatureEncoder'](x['voxels'], x['num_points'])
        if 'MiddleSparseEncoder' not in nets:
            raise Exeption('MiddleSparseEncoder network is not specified.')
        voxel_features = nets['MiddleSparseEncoder'](voxels, x['coordinates'], batch_size)

        keep_tensors['voxel_features'] = voxel_features
        bev_features.append(voxel_features)

    bev_features = torch.cat(bev_features, dim=1)
    keep_tensors['bev_features'] = bev_features

    if 'MergeUnit' in nets:
        net = nets['MergeUnit']

        t = int(x.get('chunk_len', 1))
        bos = x.get('bos', None)

        pose = None
        if net.costmap_pose_name is not None:
            name = net.costmap_pose_name
            if name not in x:
                raise Exeption(f'Could not find {name} in the input. '
                                'Make sure {name} is mentioned in the datalayer '
                                '"data_format.outputs" config file.')
            pose = x[name]

        bev_features = net(bev_features, t=t, bos=bos, pose=pose)

        keep_tensors['merged_bev_features'] = bev_features

    preds = nets['BEVClassifier'](bev_features)
    preds.update(keep_tensors)

    return preds

def step(nets, inputs, criterion=None):
    batch_size = int(inputs['batch_size'])
    model_outputs = forward_nets(nets, inputs, batch_size)
    pred = model_outputs['bev_preds']
    loss = 0.0
    if criterion is not None:
        loss_input = dict(labels=inputs['label'])
        label, loss, meta_data = criterion(pred, **loss_input)

    return pred, label, loss, tu.numpify_criterion_meta_data(meta_data)

def prepare_inputs(batch_data, device):
    def to_device(key):
        return batch_data[key].to(device, non_blocking=True)

    inputs = dict()
    for key in batch_data:
        if key in ['points', 'info']:
            inputs[key] = batch_data[key]
        else:
            inputs[key] = to_device(key)

    inputs['batch_size'] = len(inputs['label'])

    return inputs

def train(nets, net_opts, trainloader, criterion, epoch, writer, device):
    tu.make_train(nets)
    loss_avg = []
    itr = tqdm(trainloader)

    for i, batch_data in enumerate(itr):
        # if i > 10:
        #     break
        for _, opt in net_opts.items():
            opt.zero_grad()

        inputs = prepare_inputs(batch_data, device)
        pred, label, loss, criterion_meta_data = step(nets, inputs, criterion=criterion)
        loss.backward()

        if g.log_interval > 0 and i % g.log_interval == 0:
            print('learning rate:\n%s' % tabulate.tabulate([
                (name, opt.param_groups[0]['lr']) for name, opt in net_opts.items()]))

            n_iter = (epoch - 1) * len(trainloader) + i + 1
            writer.add_scalar('Train/loss', loss.item(), n_iter)
            [writer.add_scalar(f'Train/lr_{name}', opt.param_groups[0]['lr'], n_iter)
                for name, opt in net_opts.items()]
            itr.set_description('train loss: %3f' % loss.item())
            loss_avg += [loss.item()]

            vis = []

            # can we assume last layer is sm?
            pred_sm = torch.softmax(pred, axis=1)
            for b in range(pred.shape[0]):
                coords = inputs['coordinates']
                coords = coords[coords[:, 0] == b][:, 1:]
                meta = criterion_meta_data[b] if criterion_meta_data else None
                v = vis_utils.visualize(
                        pred_sm[b].detach().cpu().numpy(),
                        label[b].detach().cpu().numpy(),
                        g.dataset_type,
                        coords=coords.detach().cpu().numpy(),
                        criterion_meta_data=meta)
                vis.append(v)

            writer.add_images('label/pred', np.array(vis), global_step=0, dataformats='NHWC')


            for name, net in nets.items():
                print('%s weights:\n%s' % (name, tu.module_weights_stats(net)))
            for name, net in nets.items():
                print('%s grad:\n%s' % (name, tu.module_grad_stats(net)))

        for _, opt in net_opts.items():
            opt.step()

        if i % 20 == 0:
            torch.cuda.empty_cache()

    return loss_avg

def validate(nets, validloader, criterion, n_iter, writer, device, vis_dir=None):
    itr = tqdm(validloader)
    tu.make_eval(nets)
    loss = []

    if g.include_unknown:
        ignore_idx = g.num_class - 1
    else:
        ignore_idx = 255

    evaluator = eval_utils.Evaluator(num_classes=g.num_class, ignore_label=ignore_idx)

    torch.cuda.empty_cache()

    it = 0
    for i, batch_data in enumerate(itr):
        # if i > 10:
        #     break
        inputs = prepare_inputs(batch_data, device)

        with torch.no_grad():
            pred_, label, loss_, criterion_meta_data = step(nets, inputs, criterion=criterion)

        loss += [float(loss_.item())]

        #pred = pred_[:, :-1].argmax(dim=1)
        pred = pred_.argmax(dim=1)
        evaluator.append(pred, label)
        itr.set_description("Acc: %3f, IoUMean: %3f" % (evaluator.acc(),
                                                        evaluator.meanIoU()))

        if vis_dir is not None:
            # can we assume last layer is sm?
            pred_sm = torch.softmax(pred_, axis=1)

            for b in range(label.shape[0]):
                coords = inputs['coordinates']
                coords = coords[coords[:, 0] == b][:, 1:]

                meta = criterion_meta_data[b] if criterion_meta_data else None
                
                v = vis_utils.visualize(
                        pred_sm[b].cpu().numpy(),
                        label[b].cpu().numpy(), g.dataset_type,
                        coords=coords.cpu().numpy(),
                        criterion_meta_data=meta)

                v = cv2.cvtColor(v, cv2.COLOR_RGB2BGR)

                info = inputs['info'][b]
                seq_dir = os.path.join(vis_dir, info['seq'])
                os.makedirs(seq_dir, exist_ok=True)
                cv2.imwrite(os.path.join(seq_dir,
                            '{:05}.png'.format(info['frame'])), v)
                it += 1
    loss_avg = np.array(loss).mean()

    # logging
    writer.add_scalar('Val/loss', loss_avg.item(), n_iter)

    cw_iou = evaluator.classwiseIoU()
    cw_acc = evaluator.classwiseAcc()

    for cls, (ciou, cacc) in enumerate(zip(cw_iou, cw_iou)):
        try: # try to get the class name
            cls_name = g.data_format.labels[f'i{cls}']
        except:
            cls_name = f'class{cls}'

        writer.add_scalar(f'Val/iou/{cls_name}', ciou, n_iter)
        writer.add_scalar(f'Val/acc/{cls_name}', cacc, n_iter)

    writer.add_scalar('Val/iou', np.nanmean(cw_iou), n_iter)
    writer.add_scalar('Val/acc', np.nanmean(cw_acc), n_iter)

    return loss_avg


@torch.no_grad()
def test_single(nets, g):
    """
    Testing single-frame BEVNet.

    Args:
      nets: dictionary of models to forward/backprop in sequential order.
      g: global arguments dictionary.
    """
    voxel_cfg = g.voxelizer
    pc_format = g.data_format.point_cloud
    voxel_generator = VoxelGenerator(
        voxel_size=list(pc_format['voxel_size']),
        point_cloud_range=list(pc_format['range']),
        max_num_points=voxel_cfg['max_number_of_points_per_voxel'],
        full_mean=voxel_cfg['full_mean'],
        max_voxels=voxel_cfg['max_voxels'])


    if 'test_input_reader' not in g:
        print('test_input_reader not found. Using eval_input_reader instead.')
        g.test_input_reader = g.eval_input_reader

    ## TODO: add g.buffer_scans and g.buffer_scan_stride to BEVLoader?
    test_dataset = data_loader.BEVMiniSeqDataset(g.dataset_path,
                                                 g.data_format,
                                                 g.test_input_reader,
                                                 voxel_generator)


    testloader = data_loader.BEVDataLoader(
            test_dataset,
            batch_size=1,
            num_workers=g.num_workers,
            is_train=False)

    criterion = tu.get_criterion(g)

    if g.resume:
        model_fn = g.resume
    else:
        model_fn = tu.get_last_model(g.output)

    nepoch = tu.load_nets(model_fn, nets, None, False)
    print('Loaded model {}'.format(model_fn))

    output = os.path.join(g.output, 'test', 'epoch-{}'.format(nepoch))
    vis_dir = os.path.join(output, 'vis')
    
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(output, exist_ok=True)
    writer = SummaryWriter(log_dir=output)
    tu.log_dict(writer, 'global_args', g, 0)


    test_loss = validate(nets, testloader, criterion, nepoch,
                        writer, g.train_device, vis_dir)

    writer.close()

def train_single(nets, net_opts, g):
    """
    Training single-frame BEVNet.

    Args:
      nets: dictionary of models to forward/backprop in sequential order.
      net_opts: optim related modules for training nets.
      g: global arguments dictionary.
    """
    voxel_cfg = g.voxelizer
    pc_format = g.data_format.point_cloud
    voxel_generator = VoxelGenerator(
        voxel_size=list(pc_format['voxel_size']),
        point_cloud_range=list(pc_format['range']),
        max_num_points=voxel_cfg['max_number_of_points_per_voxel'],
        full_mean=voxel_cfg['full_mean'],
        max_voxels=voxel_cfg['max_voxels'])

    ## TODO: add g.buffer_scans and g.buffer_scan_stride to BEVLoader?
    valid_dataset = data_loader.BEVMiniSeqDataset(g.dataset_path,
                                                  g.data_format,
                                                  g.eval_input_reader,
                                                  voxel_generator)


    if 'miniseq_sampler' in g.train_input_reader:
        miniseq_sampler = g.train_input_reader.miniseq_sampler
        miniseq_len = miniseq_sampler.get('len', 1)
        miniseq_chunk_len = miniseq_sampler.get('chunck_len', 1)
        miniseq_stride = miniseq_sampler.get('stride', 1)
    else:
        miniseq_len = 1
        miniseq_chunk_len = 1
        miniseq_stride = 1

    train_dataset = data_loader.BEVMiniSeqDataset(g.dataset_path,
                                                  g.data_format,
                                                  g.train_input_reader,
                                                  voxel_generator,
                                                  miniseq_len=miniseq_len,
                                                  stride=miniseq_stride)


    trainloader = data_loader.BEVDataLoader(
        train_dataset,
        batch_size=g.batch_size,
        num_workers=g.num_workers,
        is_train=True,
        chunk_len=miniseq_chunk_len)

    validloader = data_loader.BEVDataLoader(
            valid_dataset,
            batch_size=1,
            num_workers=g.num_workers,
            is_train=False)


    criterion = tu.get_criterion(g)
    output = g.output
    os.makedirs(output, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output))
    tu.log_dict(writer, 'global_args', g, 0)

    best_valid_loss = np.inf

    resume_epoch = 0
    if g.resume:
        save_epoch = tu.load_nets(g.resume, nets, net_opts, False)
        print('loaded', g.resume, 'epoch', save_epoch)
        if g.resume_epoch >= 0:
            resume_epoch = g.resume_epoch
        else:
            resume_epoch = save_epoch

    net_scheds = {
        name: torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=g.lr_decay_epoch,
            gamma=g.lr_decay,
            last_epoch=-1)
        for name, opt in net_opts.items()
    }
    tu.save_nets(nets, net_opts, resume_epoch, g, os.path.join(output, 'model.pth'))
    for epoch in range(1, g.epochs + 1):
        if resume_epoch < epoch:
            train(nets, net_opts, trainloader, criterion, epoch, writer, g.train_device)
            tu.save_nets(nets, net_opts, epoch, g, os.path.join(output, 'model.pth'))

            n_iter = epoch * len(trainloader)
            val_loss = validate(nets, validloader, criterion, n_iter, writer, g.train_device)

            if val_loss < best_valid_loss:
                print("new best valid loss at %3f, saving model..." % val_loss)
                best_valid_loss = val_loss
                tu.save_nets(nets, net_opts, epoch, g, os.path.join(output, 'best.pth'))

        for _, sched in net_scheds.items():
            sched.step()

    writer.close()

if __name__ == "__main__":
    torch.manual_seed(2021)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    add_common_arguments(parser)

    train_funcs = {
      'default': train_single,
    }

    g = EasyDict(vars(parser.parse_args()))
    dataset_cfg = file_utils.load_yaml(g.dataset_config)
    g.update(dataset_cfg)

    model_spec = file_utils.load_yaml(g.model_config)

    ## pop loss, num_class and any key starts with '_'.
    # The rest are model configs.
    g.loss = model_spec.pop('loss')
    g.num_class = model_spec.pop('num_class')
    for n in list(model_spec.keys()):
        if n.startswith('_'):
            model_spec.pop(n)
    ##

    ret = tu.build_nets(model_spec, g.train_device)
    g.model_config = model_spec
    nets = {
        name: spec['net'] for name, spec in ret.items()
    }

    if g.test:
        test_single(nets, g)
        exit(0)

    net_opts = {
        name: spec['opt'] for name, spec in ret.items() if spec['opt'] is not None
    }

    train_funcs[g.model_variant](nets, net_opts, g)
