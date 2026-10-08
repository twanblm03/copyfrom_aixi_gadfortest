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
import numpy as np
import argparse

from models import build_model
from util.utils import *
import util.misc as utils
import util.logger as loggers
from dataloader.dataloader import read_dataset
import evaluation.cafe_eval as evaluation

parser = argparse.ArgumentParser(description='Group Activity Detection evaluation code')

# Dataset specification
parser.add_argument('--dataset', default='cafe', type=str, help='dataset name')
protocol_group = parser.add_mutually_exclusive_group()
protocol_group.add_argument('--val_mode', dest='val_mode', action='store_true',
                            help='evaluate the official validation split (default)')
protocol_group.add_argument('--final_test', dest='val_mode', action='store_false',
                            help='evaluate the official test split')
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
parser.add_argument('--batch', default=16, type=int, help='Batch size')
parser.add_argument('--test_batch', default=16, type=int, help='Test batch size')
parser.add_argument('--drop_rate', default=0.1, type=float, help='Dropout rate')
# GPU
parser.add_argument('--device', default="0, 1", type=str, help='GPU device')
parser.add_argument('--num_workers', default=4, type=int, help='DataLoader worker count')

# Load model
parser.add_argument('--model_path', default="", type=str, help='pretrained model path')

# Visualization
parser.add_argument('--result_path', default="./outputs/")

# Evaluation
parser.add_argument('--groundtruth', default='./evaluation/gt_tracks.txt', type=str)
parser.add_argument('--labelmap', default='./label_map/group_action_list.pbtxt', type=str)
parser.add_argument('--giou_thresh', default=1.0, type=float)
parser.add_argument('--eval_type', default="gt_base", type=str, help='gt_based or detection_based')
parser.add_argument('--eval_image_width', default=1280, type=int)
parser.add_argument('--eval_image_height', default=720, type=int)

args = parser.parse_args()
path = None

SEQS_CAFE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

ACTIVITIES = ['Queueing', 'Ordering', 'Drinking', 'Working', 'Fighting', 'Selfie', 'Individual', 'No']


def load_model_checkpoint(model, model_path):
    if not model_path:
        raise ValueError('--model_path is required')
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model_state = model.state_dict()
    compatible = {}
    for key, value in state_dict.items():
        if key in model_state:
            compatible[key] = value
        elif key.startswith('module.') and key[len('module.'):] in model_state:
            compatible[key[len('module.'):]] = value
        elif ('module.' + key) in model_state:
            compatible['module.' + key] = value
        else:
            compatible[key] = value
    model.load_state_dict(compatible, strict=True)


def main():
    global args, path

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if not torch.cuda.is_available():
        raise RuntimeError('B0-repro evaluation requires CUDA')
    if not os.path.isfile(args.groundtruth):
        raise FileNotFoundError('evaluation ground truth not found: %s' % args.groundtruth)
    if not os.path.isfile(args.labelmap):
        raise FileNotFoundError('label map not found: %s' % args.labelmap)

    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    exp_name = '[%s]_GAD_<%s>' % (args.dataset, time_str)

    path = os.path.join(args.result_path, exp_name)
    if not os.path.exists(path):
        os.makedirs(path)

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _, test_set = read_dataset(args)
    sampler_test = data.SequentialSampler(test_set)

    test_loader = data.DataLoader(test_set, args.test_batch, sampler=sampler_test, drop_last=False,
                                  collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    model, criterion = build_model(args)
    model = model.cuda()
    device_ids = [device_id.strip() for device_id in args.device.split(',') if device_id.strip()]
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model)
    load_model_checkpoint(model, args.model_path)

    args.groundtruth = open(args.groundtruth, 'r')
    args.labelmap = open(args.labelmap, 'r')
    metrics = evaluation.GAD_Evaluation(args)

    test_log, result = validate(test_loader, model, criterion, metrics)
    print("group mAP at 1.0: %.2f" % result['group_mAP_1.0'])
    print("group mAP at 0.5: %.2f" % result['group_mAP_0.5'])
    print("outlier mIoU: %.2f" % result['outlier_mIoU'])


@torch.no_grad()
def validate(test_loader, model, criterion, metrics):
    model.eval()
    criterion.eval()

    metric_logger = loggers.MetricLogger(mode="test", delimiter="  ")    
    header = 'Evaluation Inference: '

    print_freq = len(test_loader)
    name_to_vid = {name: i + 1 for i, name in enumerate(SEQS_CAFE)}
    file_path = path + '/pred_group_test_%s.txt' % args.split
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
