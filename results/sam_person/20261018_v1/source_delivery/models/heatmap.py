"""Instance occupancy on B0's layer4 grid. Boxes are normalized cxcywh.

Grid samples represent continuous pixel centers (x+.5,y+.5). ROI boxes are
scaled exactly once, then torchvision ROIAlign(aligned=True) subtracts .5.
No target, assignment or label is used in the forward prediction path.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align


VARIANTS = ('off', 'h1_sup', 'h1_match', 'h2_s')


def xyxy_grid(boxes, height, width):
    center, extent = boxes[..., :2], boxes[..., 2:]
    corners = torch.cat((center - extent / 2, center + extent / 2), -1)
    return corners * boxes.new_tensor([width, height, width, height])


def frame_validity(boxes, dummy_mask, supplied=None):
    corners = xyxy_grid(boxes, 1, 1)
    lo, hi = corners[..., :2].clamp(0, 1), corners[..., 2:].clamp(0, 1)
    valid = torch.isfinite(boxes).all(-1) & (boxes[..., 2:] > 1e-7).all(-1)
    valid = valid & (hi > lo).all(-1) & ~dummy_mask[:, None, :].bool()
    if supplied is not None:
        valid = valid & supplied.bool()
    return valid


def gaussian_targets(boxes, members, valid, height, width,
                     sigma_ratio=.25, sigma_min=.5, sigma_max=4.):
    """[T,N,4], [G,N], [T,N] -> [G,T,H,W], explicit [G,T]."""
    if not (0 < sigma_ratio and 0 < sigma_min <= sigma_max):
        raise ValueError('invalid Gaussian sigma settings')
    t, n, _ = boxes.shape
    g = members.shape[0]
    if not g:
        return boxes.new_zeros((0, t, height, width)), valid.new_zeros((0, t))
    clean = torch.where(valid[..., None], boxes, torch.zeros_like(boxes))
    scale = boxes.new_tensor([width, height])
    centers = clean[..., :2] * scale
    sigmas = (clean[..., 2:] * scale * sigma_ratio).clamp(sigma_min, sigma_max)
    xx = torch.arange(width, device=boxes.device, dtype=boxes.dtype) + .5
    yy = torch.arange(height, device=boxes.device, dtype=boxes.dtype) + .5
    dx = (xx[None, None, None, :] - centers[..., 0, None, None]) / sigmas[..., 0, None, None]
    dy = (yy[None, None, :, None] - centers[..., 1, None, None]) / sigmas[..., 1, None, None]
    actors = torch.exp(-.5 * (dx.square() + dy.square()))
    member_valid = members.bool()[:, None, :] & valid[None]
    maps = (actors[None] * member_valid[..., None, None]).max(2).values
    return maps, member_valid.any(2)


def roi_responses(masks, boxes, valid, output_size=3, sampling_ratio=2):
    """Continuous [B,K,T,H,W] -> per-frame [B,K,T,N], mean [B,K,N]."""
    b, k, t, h, w = masks.shape
    n = boxes.shape[2]
    grids = masks.permute(0, 2, 1, 3, 4).reshape(b*t, k, h, w)
    safe = torch.where(valid[..., None], boxes, torch.zeros_like(boxes))
    corners = xyxy_grid(safe, h, w).reshape(b*t, n, 4)
    # Clip to the image boundary, including boxes which straddle its edge.
    corners = torch.stack((corners[..., 0].clamp(0, w), corners[..., 1].clamp(0, h),
                           corners[..., 2].clamp(0, w), corners[..., 3].clamp(0, h)), -1)
    aligned = roi_align(grids, list(corners.unbind(0)), output_size=output_size,
                        spatial_scale=1., sampling_ratio=sampling_ratio, aligned=True)
    response = aligned.mean((-1, -2)).reshape(b, t, n, k).permute(0, 3, 1, 2)
    response = response * valid[:, None].to(response.dtype)
    mean = response.sum(2) / valid.sum(1)[:, None].clamp(min=1).to(response.dtype)
    return response, mean


class GroupMaskHead(nn.Module):
    def __init__(self, dim, mask_dim=64, bias=-2., alpha=20., output_size=3, sampling_ratio=2,
                 membership=False):
        super().__init__()
        self.pixel = nn.Conv2d(dim, mask_dim, 1)
        self.embedding = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, mask_dim))
        self.bias = nn.Parameter(torch.tensor(float(bias)))
        self.output_size, self.sampling_ratio = output_size, sampling_ratio
        self.mask_dim = mask_dim
        if membership:
            if alpha <= 0:
                raise ValueError('membership alpha must be positive')
            self.raw_alpha = nn.Parameter(torch.tensor(math.log(math.expm1(alpha))))
        else:
            self.register_parameter('raw_alpha', None)

    @property
    def alpha(self):
        return F.softplus(self.raw_alpha) + 1e-6

    def forward(self, scene, queries, boxes, valid):
        b, t, d, h, w = scene.shape
        pixel = self.pixel(scene.reshape(b*t, d, h, w)).reshape(b, t, self.mask_dim, h, w)
        embedding = self.embedding(queries)
        logits = torch.einsum('btkd,btdhw->bkthw', embedding, pixel) / math.sqrt(self.mask_dim) + self.bias
        masks = logits.sigmoid()
        response, pooled = roi_responses(masks, boxes, valid, self.output_size, self.sampling_ratio)
        out = {'group_mask_logits': logits, 'group_masks': masks,
               'spatial_response': response, 'actor_frame_valid': valid}
        if self.raw_alpha is not None:
            out['membership_logits'] = self.alpha * pooled
            out['membership'] = out['membership_logits'].softmax(1)
        return out


def soft_focal(logits, targets, gamma=2., alpha=.25):
    probability = logits.sigmoid()
    pt = targets * probability + (1-targets) * (1-probability)
    balance = 1. if alpha < 0 else alpha*targets + (1-alpha)*(1-targets)
    return balance * (1-pt).pow(gamma) * F.binary_cross_entropy_with_logits(logits, targets, reduction='none')


def mask_terms(logits, targets, gamma=2., alpha=.25):
    focal = soft_focal(logits, targets, gamma, alpha).mean((-1, -2))
    probability = logits.sigmoid()
    dice = 1 - (2*(probability*targets).sum((-1, -2)) + 1) / (
        probability.sum((-1, -2)) + targets.sum((-1, -2)) + 1)
    return focal, dice


def mask_pair_cost(logits, target, valid, gamma=2., alpha=.25):
    k, t, h, w = logits.shape
    g = target.shape[0]
    if not g:
        return logits.new_zeros((k, 0)), logits.new_zeros((k, 0))
    pred = logits[:, None].expand(k, g, t, h, w)
    truth = target[None].expand(k, g, t, h, w)
    focal, dice = mask_terms(pred, truth, gamma, alpha)
    weights = valid[None].to(logits.dtype)
    count = weights.sum(-1).clamp(min=1)
    return (focal*weights).sum(-1)/count, (dice*weights).sum(-1)/count


def prepare_targets(outputs, targets, args):
    h, w = outputs['group_mask_logits'].shape[-2:]
    prepared = []
    for i, target in enumerate(targets):
        item = dict(target)
        maps, frame_valid = gaussian_targets(target['boxes'], target['members'][0],
            outputs['actor_frame_valid'][i], h, w, args.heatmap_sigma_ratio,
            args.heatmap_sigma_min, args.heatmap_sigma_max)
        item['group_heatmaps'], item['group_frame_valid'] = maps, frame_valid
        prepared.append(item)
    return prepared


def heatmap_losses(outputs, targets, indices, args):
    logits = outputs['group_mask_logits']
    zero = logits.sum()*0
    focal_items, dice_items, empty_items = [], [], []
    for b, (src, tgt) in enumerate(indices):
        src, tgt = src.to(logits.device), tgt.to(logits.device)
        for slot, group in zip(src, tgt):
            valid = targets[b]['group_frame_valid'][group]
            if valid.any():
                focal, dice = mask_terms(logits[b, slot, valid], targets[b]['group_heatmaps'][group, valid],
                                         args.heatmap_focal_gamma, args.heatmap_focal_alpha)
                focal_items.append(focal.mean())
                dice_items.append(dice.mean())
        unmatched = torch.ones(logits.shape[1], device=logits.device, dtype=torch.bool)
        unmatched[src] = False
        frame_valid = outputs['actor_frame_valid'][b].any(-1)
        if unmatched.any() and frame_valid.any():
            empty = logits[b, unmatched][:, frame_valid]
            empty_items.append(F.binary_cross_entropy_with_logits(empty, torch.zeros_like(empty)))
    average = lambda values: torch.stack(values).mean() if values else zero
    return {'loss_mask_focal': average(focal_items), 'loss_mask_dice': average(dice_items),
            'loss_mask_empty': average(empty_items)}


def add_heatmap_arguments(parser):
    parser.add_argument('--research_b0_repro', action='store_true')
    parser.add_argument('--heatmap_variant', choices=VARIANTS, default='off')
    parser.add_argument('--heatmap_dim', type=int, default=64)
    parser.add_argument('--heatmap_bias', type=float, default=-2.)
    parser.add_argument('--heatmap_alpha', type=float, default=20.)
    parser.add_argument('--heatmap_sigma_ratio', type=float, default=.25)
    parser.add_argument('--heatmap_sigma_min', type=float, default=.5)
    parser.add_argument('--heatmap_sigma_max', type=float, default=4.)
    parser.add_argument('--heatmap_roi_size', type=int, default=3)
    parser.add_argument('--heatmap_roi_sampling', type=int, default=2)
    parser.add_argument('--heatmap_focal_gamma', type=float, default=2.)
    parser.add_argument('--heatmap_focal_alpha', type=float, default=.25)
    parser.add_argument('--heatmap_focal_coef', type=float, default=1.)
    parser.add_argument('--heatmap_dice_coef', type=float, default=1.)
    parser.add_argument('--heatmap_empty_coef', type=float, default=.1)
    parser.add_argument('--heatmap_cost_focal', type=float, default=1.)
    parser.add_argument('--heatmap_cost_dice', type=float, default=1.)
    parser.add_argument('--attention_probe', action='store_true')
