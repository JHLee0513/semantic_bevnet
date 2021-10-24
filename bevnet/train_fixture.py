import os

import numpy as np
import tabulate
import torch
from spconv.utils import VoxelGenerator
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from . import bev_utils
from . import train_fixture_utils as tfu


def _load_model(model_file, nets, net_opts, tolerant):
    state = tfu.load_model(os.path.dirname(model_file),
                           os.path.basename(model_file), load_to_cpu=True)
    epoch = int(state['epoch'])

    for name, net in nets.items():
        tfu.load_state_helper(net, state['nets'][name], tolerant)

    # TODO: currently don't load net_opts, but we might want to do that later.
    return epoch


def _save_model(nets, net_opts, epoch, global_args, model_file):
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
    tfu.save_model(state, epoch, '', model_file)


def train_single(nets, net_opts, g):
    """
    Training single-frame BEVNet.
    
    Args:
      nets: dictionary of models to forward/backprop in sequential order.
      net_opts: optim related modules for training nets.
      g: global arguments dictionary.
    """
    def forward_nets(nets, x):
        # forward 3 modules: VoxelFeatureEncoder, MiddleSparseEncoder, BEVClassifier
        voxels = nets['VoxelFeatureEncoder'](x['voxels'], x['num_points'])
        voxel_features = nets['MiddleSparseEncoder'](voxels, x['coordinates'], g.batch_size)
        preds = nets['BEVClassifier'](voxel_features)
        return preds

    def step(nets, inputs, labels=None, criterion=None):
        model_outputs = forward_nets(nets, inputs)
        pred = model_outputs['bev_preds']
        loss = 0.0
        if g.include_unknown:
            # Note that `num_class` already includes the unknown label
            labels[labels == 255] = g.num_class - 1
        if labels is not None and criterion is not None:
            labels = labels.long()
            loss = criterion(pred, labels)
        return pred, loss

    def train(nets, net_opts, trainloader, criterion, epoch, writer, device):
        tfu.make_train(nets)
        loss_avg = []
        itr = tqdm(trainloader)

        for i, batch_data in enumerate(itr):
            for _, opt in net_opts.items():
                opt.zero_grad()

            def to_device(key):
                return batch_data[key].to(device, non_blocking=True)

            label = to_device('label')
            inputs = dict()
            inputs['batch_size'] = len(label)
            for key in batch_data:
                if key in ['points']:
                    continue
                inputs[key] = to_device(key)

            pred, loss = step(nets, inputs, labels=label, criterion=criterion)
            loss.backward()

            if g.log_interval > 0 and i % g.log_interval == 0:
                print('learning rate:\n%s' % tabulate.tabulate([
                    (name, opt.param_groups[0]['lr']) for name, opt in net_opts.items()]))

                n_iter = (epoch - 1) * len(trainloader) + i + 1
                writer.add_scalar('Train/loss', loss.item(), n_iter)
                itr.set_description("train loss: %3f" % loss.item())
                loss_avg += [loss.item()]

                tfu.visualize_predictions(writer,
                                          pred.data.cpu().numpy(),
                                          label.cpu().numpy(),
                                          g.dataset_type)

                for name, net in nets.items():
                    print('%s weights:\n%s' % (name, tfu.module_weights_stats(net)))
                for name, net in nets.items():
                    print('%s grad:\n%s' % (name, tfu.module_grad_stats(net)))

            for _, opt in net_opts.items():
                opt.step()

            if i % 10 == 0:
                torch.cuda.empty_cache()

        return loss_avg

    def validate(nets, validloader, criterion, n_iter, writer, device):
        itr = tqdm(validloader)
        tfu.make_eval(nets)
        loss = []

        if g.include_unknown:
            ignore_idx = g.num_class - 1
        else:
            ignore_idx = 255

        evaluator = bev_utils.Evaluator(num_classes=g.num_class, ignore_label=ignore_idx)

        torch.cuda.empty_cache()

        for i, batch_data in enumerate(itr):
            def to_device(key):
                return batch_data[key].to(device, non_blocking=True)

            label = to_device('label')

            inputs = dict()
            inputs['batch_size'] = len(label)
            for key in batch_data:
                if key in ['points']:
                    continue
                inputs[key] = to_device(key)

            with torch.no_grad():
                pred_, loss_ = step(nets, inputs, labels=label, criterion=criterion)

            loss += [float(loss_)]
            pred = pred_.argmax(dim=1)
            evaluator.append(pred, label)
            itr.set_description("Acc: %3f, IoUMean: %3f" % (evaluator.acc(),
                                                            evaluator.meanIoU()))

        loss_avg = np.array(loss).mean()

        # logging
        writer.add_scalar('Val/loss', loss_avg.item(), n_iter)

        cw_iou = evaluator.classwiseIoU()
        cw_acc = evaluator.classwiseAcc()

        for cls, (ciou, cacc) in enumerate(zip(cw_iou, cw_iou)):
            writer.add_scalar('Val/iou_class{}'.format(cls), ciou, n_iter)
            writer.add_scalar('Val/acc_class{}'.format(cls), cacc, n_iter)

        writer.add_scalar('Val/iou', np.nanmean(cw_iou), n_iter)
        writer.add_scalar('Val/acc', np.nanmean(cw_acc), n_iter)

        return loss_avg

    voxel_cfg = g.voxelizer
    voxel_generator = VoxelGenerator(
        voxel_size=list(voxel_cfg['voxel_size']),
        point_cloud_range=list(voxel_cfg['point_cloud_range']),
        max_num_points=voxel_cfg['max_number_of_points_per_voxel'],
        full_mean=voxel_cfg['full_mean'],
        max_voxels=voxel_cfg['max_voxels'])

    train_dataset = bev_utils.BEVLoaderV2(g.train_input_reader,
                                          g.dataset_path,
                                          voxel_generator=voxel_generator,
                                          n_buffer_scans=g.buffer_scans,
                                          buffer_scan_stride=g.buffer_scan_stride)
    valid_dataset = bev_utils.BEVLoaderV2(g.eval_input_reader,
                                          g.dataset_path,
                                          voxel_generator=voxel_generator,
                                          n_buffer_scans=g.buffer_scans,
                                          buffer_scan_stride=g.buffer_scan_stride)

    class_weights = torch.tensor(g.class_weights).to(g.train_device)
    assert len(class_weights) == g.num_class

    if g.include_unknown:
        ignore_idx = -100
    else:
        ignore_idx = 255

    criterion = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_idx,
                                          weight=class_weights)
    trainloader = data.DataLoader(
        train_dataset,
        batch_size=g.batch_size,
        num_workers=g.num_workers,
        collate_fn=bev_utils.bev_single_collate_fn,
        drop_last=True,
        shuffle=True)

    validloader = data.DataLoader(
        valid_dataset,
        batch_size=g.batch_size,
        num_workers=g.num_workers,
        collate_fn=bev_utils.bev_single_collate_fn,
        drop_last=True,
        shuffle=False)

    output = g.output
    os.makedirs(output, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output))
    tfu.log_dict(writer, 'global_args', g, 0)

    best_valid_loss = np.inf

    resume_epoch = 0
    if g.resume:
        save_epoch = _load_model(g.resume, nets, net_opts, False)
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

    for epoch in range(1, g.epochs + 1):
        if resume_epoch < epoch:
            train(nets, net_opts, trainloader, criterion, epoch, writer, g.train_device)
            _save_model(nets, net_opts, epoch, g, os.path.join(output, 'model.pth'))

            n_iter = epoch * len(trainloader)
            val_loss = validate(nets, validloader, criterion, n_iter, writer, g.train_device)

            if val_loss < best_valid_loss:
                print("new best valid loss at %3f, saving model..." % val_loss)
                best_valid_loss = val_loss
                _save_model(nets, net_opts, epoch, g, os.path.join(output, 'best.pth'))

        for _, sched in net_scheds.items():
            sched.step()

    writer.close()


def train_recurrent(nets, net_opts, g):
    def forward_nets(nets, x):
        # forward 3 modules: VoxelFeatureEncoder, MiddleSparseEncoder, BEVClassifier
        voxels = nets['VoxelFeatureEncoder'](x['voxels'], x['num_points'])
        voxel_features = nets['MiddleSparseEncoder'](voxels, x['coordinates'], g.batch_size)
        seq_start = x['seq_start'] #iif 'seq_start' in example else None
        pose = x['pose']
        preds = nets['BEVClassifier'](voxel_features, seq_start = seq_start, input_pose = pose)
        return preds

    def _step(nets, inputs, labels=None, criterion=None):
        model_outputs = forward_nets(nets, inputs)
        loss = 0
        pred = model_outputs['bev_preds']
        if g.include_unknown:
            # Note that `num_class` already includes the unknown label
            labels[labels == 255] = g.num_class - 1

        if labels is not None and criterion is not None:
            labels = labels.long()
            t, n, c, h, w = pred.shape
            pred = pred.reshape((t, c, h, w))
            labels = labels.squeeze(0)
            loss = criterion(pred, labels)
        return pred, loss

    def _train(nets, net_opts, trainloader, criterion, epoch, writer, device):
        tfu.make_train(nets)
        loss_avg = []
        itr = tqdm(trainloader)
        
        for i, batch in enumerate(itr):
            for _, opt in net_opts.items():
                opt.zero_grad()

            label = batch.label.to(device, non_blocking=True)
            input_voxels = {
                "voxels": [vox.to(device, non_blocking=True) for vox in batch.voxels[0]],
                "num_points": [num.to(device, non_blocking=True) for num in batch.num_points[0]],
                "coordinates": [co.to(device, non_blocking=True) for co in batch.coords[0]],
                "seq_start": batch.seq_start.to(device, non_blocking=True).squeeze(0),
                "pose": batch.pose.to(device, non_blocking=True).squeeze(0),
                "batch_size": 1,
            }

            pred, loss = _step(nets, input_voxels, labels=label, criterion=criterion)
            loss.backward()

            if g.log_interval > 0 and i % g.log_interval == 0:
                n_iter = (epoch - 1) * len(trainloader) + i + 1
                print('learning rate:\n%s' % tabulate.tabulate([
                    (name, opt.param_groups[0]['lr']) for name, opt in net_opts.items()]))
                # writer.add_scalar('Train/learning_rate',
                #                 optimizer.param_groups[0]['lr'],
                #                 n_iter)
                writer.add_scalar('Train/loss', loss.item(), n_iter)
                itr.set_description("train loss: %3f" % loss.item())
                loss_avg += [loss.item()]

                tfu.visualize_predictions(writer, pred.data.cpu().numpy(), label[0].cpu().numpy(),
                                          g.dataset_type)

                print('grad statistics:')
                for model in nets.values():
                    print(tfu.module_grad_stats(model))

            for _, opt in net_opts.items():
                opt.step()
            if i % 10 == 0:
                torch.cuda.empty_cache()

        return loss_avg

    def _validate(nets, validloader, criterion, n_iter, writer, device):
        itr = tqdm(validloader)
        tfu.make_eval(nets)
        loss = []

        if g.include_unknown:
            ignore_idx = g.num_class - 1
        else:
            ignore_idx = 255

        evaluator = bev_utils.Evaluator(num_classes=g.num_class, ignore_label=ignore_idx)

        with torch.no_grad():
            for i, batch in enumerate(itr):
                label = batch.label.to(device)
                input_voxels = {
                    "voxels": [vox.to(device) for vox in batch.voxels[0]],
                    "num_points": [num.to(device) for num in batch.num_points[0]],
                    "coordinates": [co.to(device) for co in batch.coords[0]],
                    "seq_start": batch.seq_start.to(device).squeeze(0),
                    "pose": batch.pose.to(device).squeeze(0),
                    "batch_size": 1,
                }
                pred_, loss_ = _step(nets, input_voxels, labels=label, criterion=criterion)
                loss += [float(loss_)]
                pred = pred_.argmax(dim=1).detach()
                label = label.squeeze(0).detach()

                evaluator.append(pred, label)
                itr.set_description("Acc: %3f, IoUMean: %3f" % (evaluator.acc(),
                                                                evaluator.meanIoU()))

            loss_avg = np.array(loss).mean()

            writer.add_scalar('Val/loss', loss_avg.item(), n_iter)

            cw_iou = evaluator.classwiseIoU()
            cw_acc = evaluator.classwiseAcc()

            for cls, (ciou, cacc) in enumerate(zip(cw_iou,cw_iou)):
                writer.add_scalar('Val/iou_class{}'.format(cls), ciou, n_iter)
                writer.add_scalar('Val/acc_class{}'.format(cls), cacc, n_iter)

            writer.add_scalar('Val/iou', np.nanmean(cw_iou), n_iter)
            writer.add_scalar('Val/acc', np.nanmean(cw_acc), n_iter)
            
            return loss_avg

    tfu.set_device(nets, g.train_device)
    voxel_cfg = g.voxelizer
    voxel_generator = VoxelGenerator(
        voxel_size=list(voxel_cfg['voxel_size']),
        point_cloud_range=list(voxel_cfg['point_cloud_range']),
        max_num_points=voxel_cfg['max_number_of_points_per_voxel'],
        full_mean=voxel_cfg['full_mean'],
        max_voxels=voxel_cfg['max_voxels'])

    train_dataset = bev_utils.BEVLoaderMultistepV3(
        g.train_input_reader,
        g.dataset_path,
        shuffle=True,
        n_frame=g.n_frame,
        seq_len=g.seq_len,
        frame_strides=g.frame_strides,
        voxel_generator=voxel_generator,
        n_buffer_scans=g.buffer_scans,
        buffer_scan_stride=g.buffer_scan_stride)

    valid_dataset = bev_utils.BEVLoaderMultistepV3(
        g.eval_input_reader,
        g.dataset_path,
        shuffle=False,
        n_frame=g.n_frame,
        seq_len=None,
        frame_strides=[1],
        voxel_generator=voxel_generator,
        n_buffer_scans=g.buffer_scans,
        buffer_scan_stride=g.buffer_scan_stride)

    net_scheds = {
        name: torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=g.lr_decay_epoch,
            gamma=g.lr_decay,
            last_epoch=-1)
        for name, opt in net_opts.items()
    }

    if g.include_unknown:
        ignore_idx = -100
    else:
        ignore_idx = 255
    class_weights = torch.tensor(g.class_weights).to(g.train_device)
    criterion = torch.nn.CrossEntropyLoss(
        reduction="mean", ignore_index=ignore_idx, weight=class_weights)

    trainloader = data.DataLoader(
        train_dataset,
        batch_size=g.batch_size,
        num_workers=g.num_workers,
        collate_fn=train_dataset.collate_wrapper,
        worker_init_fn=train_dataset.init
    )

    validloader = data.DataLoader(
        valid_dataset,
        batch_size=g.batch_size,
        num_workers=g.num_workers,
        collate_fn=valid_dataset.collate_wrapper,
        worker_init_fn=valid_dataset.init
    )

    output = g.output
    os.makedirs(output, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output))
    tfu.log_dict(writer, 'global_args', g, 0)

    best_valid_loss = np.inf

    resume_epoch = 0
    if g.resume:
        save_epoch = _load_model(g.resume, nets, net_opts, True)
        print('loaded', g.resume, 'epoch', save_epoch)
        if g.resume_epoch >= 0:
            resume_epoch = g.resume_epoch
        else:
            resume_epoch = save_epoch

    for epoch in range(1, g.epochs + 1):
        if resume_epoch < epoch:
            train_loss = _train(nets, net_opts, trainloader, criterion, epoch, writer, g.train_device)
            _save_model(nets, net_opts, epoch, g, os.path.join(output, 'model.pth'))
            n_iter = epoch * len(trainloader)
            val_loss = _validate(nets, validloader, criterion, n_iter, writer, g.train_device)
            if val_loss < best_valid_loss:
                print("new best valid loss at %3f, saving model..." % val_loss)
                best_valid_loss = val_loss
                _save_model(nets, net_opts, epoch, g, os.path.join(output, 'best.pth'))

        for _, sched in net_scheds.items():
            sched.step()

    writer.close()
