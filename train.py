import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import os
import math
import sys
import copy
import time
import random
import json
import re
import numpy as np
import argparse

from models import build_model
from util.utils import *
import util.misc as utils
import util.logger as loggers
from dataloader.dataloader import read_dataset
import evaluation.cafe_eval as evaluation

parser = argparse.ArgumentParser(description='Group Activity Detection train code')

# Dataset specification
parser.add_argument('--dataset', default='cafe', type=str, help='dataset name')
protocol_group = parser.add_mutually_exclusive_group()
protocol_group.add_argument(
    '--val_mode', dest='val_mode', action='store_true',
    help='train on TRAIN and evaluate/select checkpoints on VAL (default)')
protocol_group.add_argument(
    '--final_test', dest='val_mode', action='store_false',
    help='train on TRAIN+VAL and evaluate TEST once after the last epoch')
parser.set_defaults(val_mode=True)
parser.add_argument('--split', default='place', type=str, help='dataset split. place or view')
parser.add_argument('--data_path', default='../Dataset/', type=str, help='data path')
parser.add_argument('--tracks_path', default='', type=str,
                    help='optional gt_tracks.pkl path; defaults to <data_path>/cafe/gt_tracks.pkl')
parser.add_argument('--image_width', default=1280, type=int, help='Image width to resize')
parser.add_argument('--image_height', default=720, type=int, help='Image height to resize')
parser.add_argument('--random_sampling', action='store_true', help='random sampling strategy')
parser.add_argument('--num_frame', default=5, type=int, help='number of frames for each clip')
parser.add_argument('--num_class', default=6, type=int, help='number of activity classes')

# Backbone parameters
parser.add_argument('--backbone', default='resnet18', type=str, help='feature extraction backbone')
parser.add_argument('--dilation', action='store_true', help='use dilation or not')
parser.add_argument('--frozen_batch_norm', action='store_true', help='use frozen batch normalization')
parser.add_argument('--hidden_dim', default=256, type=int, help='transformer channel dimension')

# RoI Align parameters
parser.add_argument('--num_boxes', default=14, type=int, help='maximum number of actors')
parser.add_argument('--crop_size', default=5, type=int, help='roi align crop size')

# Group Transformer
parser.add_argument('--gar_nheads', default=4, type=int, help='number of heads')
parser.add_argument('--gar_enc_layers', default=6, type=int, help='number of group transformer layers')
parser.add_argument('--gar_ffn_dim', default=512, type=int, help='feed forward network dimension')
parser.add_argument('--position_embedding', default='sine', type=str, help='various position encoding')
parser.add_argument('--num_group_tokens', default=12, type=int, help='number of group tokens')
parser.add_argument('--aux_loss', action='store_true')
parser.add_argument('--group_threshold', default=0.5, type=float, help='post processing threshold')
parser.add_argument('--distance_threshold', default=0.2, type=float, help='distance mask threshold')

# Loss option
parser.add_argument('--temperature', default=0.2, type=float, help='consistency loss temperature')

# Loss coefficients (Individual)
parser.add_argument('--ce_loss_coef', default=1, type=float)
parser.add_argument('--eos_coef', default=1, type=float,
                    help="Relative classification weight of the no-object class")

# Loss coefficients (Group)
parser.add_argument('--group_eos_coef', default=1, type=float)
parser.add_argument('--group_ce_loss_coef', default=1, type=float)
parser.add_argument('--group_code_loss_coef', default=5, type=float)
parser.add_argument('--consistency_loss_coef', default=2, type=float)

# Matcher (Group)
parser.add_argument('--set_cost_group_class', default=1, type=float,
                    help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_membership', default=1, type=float,
                    help="Membership coefficient in the matching cost")

# Training parameters
parser.add_argument('--random_seed', default=1, type=int, help='random seed for reproduction')
parser.add_argument('--epochs', default=30, type=int, help='Max epochs')
parser.add_argument('--test_freq', default=1, type=int, help='validation frequency in epochs')
parser.add_argument('--batch', default=16, type=int, help='Batch size')
parser.add_argument('--test_batch', default=16, type=int, help='Test batch size')
parser.add_argument('--lr', default=1e-5, type=float, help='Initial learning rate')
parser.add_argument('--max_lr', default=1e-4, type=float, help='Max learning rate')
parser.add_argument('--warmup_epochs', default=5, type=int,
                    help='linear warmup length from lr to max_lr (paper: 5)')
parser.add_argument('--lr_schedule', default='linear', choices=['linear', 'cyclic'],
                    help='paper schedule is linear; cyclic reproduces the released code default')
parser.add_argument('--lr_step', default=4, type=int, help='step size for learning rate scheduler')
parser.add_argument('--lr_step_down', default=25, type=int, help='step down size (cyclic) for learning rate scheduler')
parser.add_argument('--weight_decay', default=0.0, type=float,
                    help='Adam weight decay (paper-faithful default: 0)')
parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')
parser.add_argument('--gradient_clipping', action='store_true', help='use gradient clipping')
parser.add_argument('--max_norm', default=1.0, type=float, help='gradient clipping max norm')
parser.add_argument('--max_train_steps', default=0, type=int,
                    help='limit optimizer steps per epoch for smoke tests (0 disables)')
parser.add_argument('--log_interval', default=10, type=int,
                    help='training progress interval in optimizer steps')
parser.add_argument('--num_workers', default=4, type=int, help='DataLoader worker count')

# GPU
parser.add_argument('--device', default="0, 1", type=str, help='GPU device')

# Load model
parser.add_argument('--load_model', action='store_true', help='load model')
parser.add_argument('--model_path', default="", type=str, help='pretrained model path')

# Visualization
parser.add_argument('--result_path', default="./result/")
parser.add_argument('--checkpoint_root', default='', type=str,
                    help='optional checkpoint root; defaults to the run directory')
parser.add_argument('--experiment_tag', default='', type=str,
                    help='stable run label appended to the timestamped directory')
parser.add_argument('--selection_metric', default='group_mAP_1.0',
                    choices=['group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU', 'loss'],
                    help='VAL metric used to save best_val.pth')

# Evaluation
parser.add_argument('--groundtruth', default='./evaluation/gt_tracks.txt', type=str)
parser.add_argument('--labelmap', default='./label_map/group_action_list.pbtxt', type=str)
parser.add_argument('--giou_thresh', default=1.0, type=float)
parser.add_argument('--eval_type', default="gt_base", type=str, help='gt_based or detection_based')
parser.add_argument('--eval_image_width', default=1280, type=int,
                    help='coordinate width used by evaluation ground truth')
parser.add_argument('--eval_image_height', default=720, type=int,
                    help='coordinate height used by evaluation ground truth')

args = parser.parse_args()
path = None

SEQS_CAFE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

ACTIVITIES = ['Queueing', 'Ordering', 'Drinking', 'Working', 'Fighting', 'Selfie', 'Individual', 'No']


def serializable_args(namespace):
    values = {}
    for key, value in vars(namespace).items():
        values[key] = value.name if hasattr(value, 'name') else value
    return values


def save_json(path_name, payload):
    temporary_path = path_name + '.tmp'
    with open(temporary_path, 'w') as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
    os.replace(temporary_path, path_name)


def save_checkpoint(path_name, state):
    temporary_path = path_name + '.tmp'
    torch.save(state, temporary_path)
    os.replace(temporary_path, path_name)


def paper_lr_factor(epoch_index):
    """Paper schedule: 1e-5 -> 1e-4 for 5 epochs, then linear decay."""
    start_factor = args.lr / args.max_lr
    if epoch_index < args.warmup_epochs:
        if args.warmup_epochs == 1:
            return 1.0
        return start_factor + (1.0 - start_factor) * epoch_index / (args.warmup_epochs - 1)

    decay_epochs = args.epochs - args.warmup_epochs
    if decay_epochs <= 0:
        return 1.0
    return max(0.0, (args.epochs - (epoch_index + 1)) / decay_epochs)


def main():
    global args, path

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if not torch.cuda.is_available():
        raise RuntimeError('B0-repro training requires CUDA')
    if args.split not in ('place', 'view'):
        raise ValueError('--split must be place or view')
    if args.test_freq < 1:
        raise ValueError('--test_freq must be positive')
    if args.max_train_steps < 0 or args.log_interval < 0 or args.num_workers < 0:
        raise ValueError('max_train_steps, log_interval, and num_workers must be non-negative')
    if args.lr <= 0 or args.max_lr <= 0 or args.lr > args.max_lr:
        raise ValueError('learning rates must satisfy 0 < lr <= max_lr')
    if args.lr_schedule == 'linear' and not 1 <= args.warmup_epochs <= args.epochs:
        raise ValueError('--warmup_epochs must be in [1, epochs]')
    if args.experiment_tag and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,99}', args.experiment_tag):
        raise ValueError('--experiment_tag must use letters, digits, dot, dash, or underscore')
    if not os.path.isfile(args.groundtruth):
        raise FileNotFoundError('evaluation ground truth not found: %s' % args.groundtruth)
    if not os.path.isfile(args.labelmap):
        raise FileNotFoundError('label map not found: %s' % args.labelmap)

    args.protocol = 'development' if args.val_mode else 'final_test'
    args.eval_split = 'val' if args.val_mode else 'test'
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    tag = '_[%s]' % args.experiment_tag if args.experiment_tag else ''
    exp_name = '[%s]_B0-repro%s_[%s]' % (args.dataset, tag, time_str)
    save_path = os.path.join(args.result_path, exp_name)
    checkpoint_dir = (os.path.join(args.checkpoint_root, exp_name)
                      if args.checkpoint_root else save_path)
    path = save_path
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_json(os.path.join(save_path, 'args.json'), serializable_args(args))

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_set, test_set = read_dataset(args)

    sampler_train = data.RandomSampler(train_set)
    sampler_test = data.SequentialSampler(test_set)

    batch_sampler_train = data.BatchSampler(sampler_train, args.batch, drop_last=True)

    train_loader = data.DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    test_loader = data.DataLoader(test_set, args.test_batch, sampler=sampler_test, drop_last=False,
                                  collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    model, criterion = build_model(args)
    model = torch.nn.DataParallel(model).cuda()

    # get the number of model parameters
    parameters = 'Number of full model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()]))
    print_log(save_path, '--------------------Number of parameters--------------------')
    print_log(save_path, parameters)

    # The paper trains all parameters with a single Adam learning rate.
    optimizer_lr = args.max_lr if args.lr_schedule == 'linear' else args.lr
    optimizer = torch.optim.Adam(model.parameters(), optimizer_lr, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=args.weight_decay)

    if args.lr_schedule == 'linear':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=paper_lr_factor)
    else:
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, args.lr, args.max_lr, step_size_up=args.lr_step,
            step_size_down=args.lr_step_down, mode='triangular2', cycle_momentum=False)

    history = {'train': [], 'val': [], 'test': []}
    selection_mode = 'min' if args.selection_metric == 'loss' else 'max'
    selection_state = {
        'metric': args.selection_metric,
        'mode': selection_mode,
        'epoch': None,
        'value': None,
    }

    if args.load_model:
        if not args.model_path:
            raise ValueError('--model_path is required with --load_model')
        checkpoint = torch.load(args.model_path, map_location='cuda')
        model.load_state_dict(checkpoint['state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        loaded_history = checkpoint.get('history')
        if isinstance(loaded_history, dict):
            history = loaded_history
            history.setdefault('train', [])
            history.setdefault('val', [])
            history.setdefault('test', [])
        loaded_selection = checkpoint.get('selection_state')
        if (isinstance(loaded_selection, dict)
                and loaded_selection.get('metric') == args.selection_metric
                and loaded_selection.get('mode') == selection_mode):
            selection_state.update(loaded_selection)
    else:
        start_epoch = 1

    args.groundtruth = open(args.groundtruth, 'r')
    args.labelmap = open(args.labelmap, 'r')
    metrics = evaluation.GAD_Evaluation(args)

    # training phase
    for epoch in range(start_epoch, args.epochs + 1):
        print_log(save_path, '----- %s at epoch #%d' % ("Train", epoch))
        train_log = train(train_loader, model, criterion, optimizer, epoch)
        print_log(save_path, 'Loss: %.4f' % (train_log['loss']))
        print_log(save_path, 'Group class error: %.2f' % (train_log['group_class_error']))
        print('Current learning rate is %f' % scheduler.get_last_lr()[0])
        scheduler.step()

        history['train'].append({'epoch': epoch, **train_log})

        should_evaluate = (epoch % args.test_freq == 0) if args.val_mode else (epoch == args.epochs)
        is_new_best = False
        if should_evaluate:
            eval_name = 'Validation' if args.val_mode else 'Final Test'
            print_log(save_path, '----- %s at epoch #%d' % (eval_name, epoch))
            test_log, result = validate(test_loader, model, criterion, metrics, epoch)
            print_log(save_path, 'Loss: %.4f' % (test_log['loss']))
            print_log(save_path, 'Group class error: %.2f' % (test_log['group_class_error']))
            print_log(save_path, "group mAP at 1.0: %.2f" % result['group_mAP_1.0'])
            print_log(save_path, "group mAP at 0.5: %.2f" % result['group_mAP_0.5'])
            print_log(save_path, "outlier mIoU: %.2f" % result['outlier_mIoU'])

            eval_metrics = dict(test_log)
            eval_metrics.update({
                'group_mAP_1.0': float(result['group_mAP_1.0']),
                'group_mAP_0.5': float(result['group_mAP_0.5']),
                'outlier_mIoU': float(result['outlier_mIoU']),
            })
            history[args.eval_split].append({'epoch': epoch, **eval_metrics})

            if args.val_mode:
                current_value = float(eval_metrics[args.selection_metric])
                previous_value = selection_state['value']
                is_new_best = (
                    previous_value is None
                    or (selection_mode == 'min' and current_value < previous_value)
                    or (selection_mode == 'max' and current_value > previous_value)
                )
                if is_new_best:
                    selection_state.update({'epoch': epoch, 'value': current_value})

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'history': history,
            'selection_state': selection_state,
            'protocol': args.protocol,
            'eval_split': args.eval_split,
        }
        latest_path = os.path.join(checkpoint_dir, 'latest.pth')
        save_checkpoint(latest_path, state)
        if is_new_best:
            best_path = os.path.join(checkpoint_dir, 'best_val.pth')
            save_checkpoint(best_path, state)
            print_log(save_path, 'Saved new best validation checkpoint: %s' % best_path)

        save_json(os.path.join(save_path, 'summary.json'), {
            'completed': epoch == args.epochs,
            'args': serializable_args(args),
            'selection_state': selection_state,
            'history': history,
        })


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    criterion.train()

    # logger
    metric_logger = loggers.MetricLogger(mode="train", delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    space_fmt = str(len(str(args.epochs)))
    header = 'Epoch [{start_epoch: >{fill}}/{end_epoch}]'.format(start_epoch=epoch, end_epoch=args.epochs,
                                                                 fill=space_fmt)
    print_freq = args.log_interval if args.log_interval > 0 else len(train_loader)

    for i, (images, targets, infos) in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        images = images.cuda()  # [B, T, 3, H, W]
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        boxes = torch.stack([t['boxes'] for t in targets])
        dummy_mask = torch.stack([t['actions'] == args.num_class + 1 for t in targets]).squeeze(1)

        # compute output
        outputs = model(images, boxes, dummy_mask)

        loss_dict = criterion(outputs, targets, log=False)
        weight_dict = criterion.weight_dict

        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if args.gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(group_class_error=loss_dict_reduced['group_class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if args.max_train_steps and i + 1 >= args.max_train_steps:
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate(test_loader, model, criterion, metrics, epoch):
    model.eval()
    criterion.eval()

    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")
    metric_logger.add_meter('group_class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Evaluation Inference: '

    print_freq = len(test_loader)
    name_to_vid = {name: i + 1 for i, name in enumerate(SEQS_CAFE)}
    file_path = path + '/pred_group_epoch_%d.txt' % epoch
    # Validation may be repeated when resuming; never append to stale output.
    open(file_path, 'w').close()

    for i, (images, targets, infos) in enumerate(metric_logger.log_every(test_loader, print_freq, header)):
        images = images.cuda()  # [B, T, 3, H, W]
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        boxes = torch.stack([t['boxes'] for t in targets])
        dummy_mask = torch.stack([t['actions'] == args.num_class + 1 for t in targets]).squeeze(1)

        # compute output
        outputs = model(images, boxes, dummy_mask)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)

        metric_logger.update(group_class_error=loss_dict_reduced['group_class_error'])

        make_txt(boxes, infos, outputs, name_to_vid, file_path)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    with open(file_path, "r") as detections:
        result = metrics.evaluate(detections)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, result


def make_txt(boxes, infos, outputs, name_to_vid, file_path):
    for b in range(boxes.shape[0]):
        for t in range(boxes.shape[1]):
            image_w, image_h = args.eval_image_width, args.eval_image_height

            pred_group_actions = outputs['pred_activities'][b]
            pred_group_actions = F.softmax(pred_group_actions, dim=1)
            members = outputs['membership'][b]

            pred_membership = torch.argmax(members.transpose(0, 1), dim=1).detach().cpu()
            keep_membership = members.transpose(0, 1).max(-1).values > args.group_threshold
            pred_group_action = torch.argmax(pred_group_actions, dim=1).detach().cpu()

            for box_idx in range(boxes.shape[2]):
                x, y, w, h = boxes[b][t][box_idx]
                x1, y1, x2, y2 = (x - w / 2) * image_w, (y - h / 2) * image_h, (x + w / 2) * image_w, (
                            y + h / 2) * image_h

                pred_group_id = pred_membership[box_idx]
                pred_group_action_idx = pred_group_action[pred_group_id]
                pred_group_action_prob = pred_group_actions[pred_group_id][pred_group_action_idx]

                if not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
                    if pred_group_action_idx != (pred_group_actions.shape[-1] - 1):
                        if bool(keep_membership[box_idx]) is False:
                            pred_group_id = -1
                            pred_group_action_idx = args.num_class

                    pred_list = [name_to_vid[infos[b]['vid']], infos[b]['sid'], infos[b]['fid'][t],
                                 int(x1), int(y1), int(x2), int(y2),
                                 int(pred_group_id), int(pred_group_action_idx) + 1,
                                 float(pred_group_action_prob)]
                    str_to_be_added = [str(k) for k in pred_list]
                    str_to_be_added = (" ".join(str_to_be_added))

                    f = open(file_path, "a+")
                    f.write(str_to_be_added + "\r\n")
                    f.close()


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack([image for image in batch[0]])
    return tuple(batch)


if __name__ == '__main__':
    main()
