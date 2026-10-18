"""One clip assignment shared by activity, occupancy and membership."""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .criterion import SetCriterion
from .group_matcher import HungarianMatcher
from .heatmap import mask_pair_cost, prepare_targets, heatmap_losses


class HeatmapMatcher(HungarianMatcher):
    def __init__(self, args):
        super().__init__(args.set_cost_group_class, args.set_cost_membership)
        self.args = args
        self.last_cost_scales = []

    @torch.no_grad()
    def forward(self, outputs, targets):
        variant = self.args.heatmap_variant
        if variant in ('off', 'h1_sup'):
            return super().forward(outputs, targets)
        indices, self.last_cost_scales = [], []
        for b, target in enumerate(targets):
            activities, members = target['activities'][0], target['members'][0]
            valid = target['dummy_idx'][0].bool()
            if variant == 'h2_s':
                valid = valid & outputs['actor_frame_valid'][b].any(0)
            sim = outputs['membership'][b] * valid.to(members.dtype)[None]
            truth = members * valid.to(members.dtype)[None]
            cls = -outputs['pred_activities'][b].softmax(-1)[:, activities]
            mem = torch.cdist(sim, truth, p=2)
            focal, dice = mask_pair_cost(outputs['group_mask_logits'][b],
                target['group_heatmaps'], target['group_frame_valid'],
                self.args.heatmap_focal_gamma, self.args.heatmap_focal_alpha)
            cost = self.cost_class*cls + self.cost_code*mem + self.args.heatmap_cost_focal*focal + self.args.heatmap_cost_dice*dice
            if not torch.isfinite(cost).all():
                raise FloatingPointError('non-finite heatmap assignment cost')
            row, col = linear_sum_assignment(cost.cpu().numpy())
            indices.append((torch.as_tensor(row, dtype=torch.int64), torch.as_tensor(col, dtype=torch.int64)))
            self.last_cost_scales.append({key: float(value.abs().mean()) if value.numel() else None
                for key, value in dict(activity=cls, membership=mem, focal=focal, dice=dice).items()})
        return indices


class HeatmapCriterion(SetCriterion):
    """Preserve ordinary B0 losses; guard empty/invalid research inputs.

    B0's order-dependent consistency batch reduction is deliberately retained
    on normal batches. Correcting it would be a separate baseline change.
    """
    def __init__(self, *positional, **kwargs):
        super().__init__(*positional, **kwargs)
        self.args = kwargs['args']

    def forward(self, outputs, targets, log=True):
        if self.args.heatmap_variant != 'off':
            targets = prepare_targets(outputs, targets, self.args)
        return super().forward(outputs, targets, log=log)

    def get_group_loss(self, loss, outputs, targets, indices, **kwargs):
        if loss == 'heatmaps':
            return heatmap_losses(outputs, targets, indices, self.args)
        return super().get_group_loss(loss, outputs, targets, indices, **kwargs)

    def loss_labels(self, outputs, targets, num_boxes, log=True):
        if all(t['dummy_idx'].bool().any() for t in targets):
            return super().loss_labels(outputs, targets, num_boxes, log)
        terms = []
        for b, target in enumerate(targets):
            valid = target['dummy_idx'][0].bool()
            if valid.any():
                terms.append(F.cross_entropy(outputs['pred_actions'][b, valid], target['actions'][0, valid], self.empty_weight))
        zero = outputs['pred_actions'].sum()*0
        return {'loss_ce': torch.stack(terms).mean() if terms else zero, 'class_error': zero.detach()}

    def loss_group_labels(self, outputs, targets, group_indices, log=True):
        if sum(len(src) for src, _ in group_indices):
            return super().loss_group_labels(outputs, targets, group_indices, log)
        prediction = outputs['pred_activities']
        labels = torch.full(prediction.shape[:2], self.num_classes, device=prediction.device, dtype=torch.long)
        return {'loss_group_ce': F.cross_entropy(prediction.transpose(1, 2), labels, self.empty_group_weight),
                'group_class_error': prediction.sum().detach()*0}

    def loss_group_code(self, outputs, targets, indices, log=True):
        if self.args.heatmap_variant != 'h2_s' and all(t['dummy_idx'].bool().any() for t in targets):
            return super().loss_group_code(outputs, targets, indices, log)
        probability = outputs['membership']
        truth = torch.zeros_like(probability)
        terms = []
        for b, (src, tgt) in enumerate(indices):
            truth[b, src.to(probability.device)] = targets[b]['members'][0, tgt.to(probability.device)]
            valid = targets[b]['dummy_idx'][0].bool()
            if 'actor_frame_valid' in outputs:
                valid = valid & outputs['actor_frame_valid'][b].any(0)
            if valid.any():
                terms.append(F.binary_cross_entropy(probability[b, :, valid], truth[b, :, valid]))
        return {'loss_group_code': torch.stack(terms).mean() if terms else probability.sum()*0}

    def loss_group_consistency(self, outputs, targets, group_indices):
        safe = True
        for target in targets:
            labels = target['membership'][0, target['dummy_idx'][0].bool()]
            groups = labels[labels >= 0]
            safe = safe and bool(groups.numel()) and all((labels == group).sum() >= 2 for group in groups)
            positions=target['dummy_idx'][0].bool().nonzero(as_tuple=True)[0]
            safe = safe and torch.equal(positions,torch.arange(len(positions),device=positions.device))
        if safe:
            return super().loss_group_consistency(outputs, targets, group_indices)
        # Edge cases only: empty groups, singleton labels, invalid actors.
        losses = []
        for b, target in enumerate(targets):
            valid = target['dummy_idx'][0].bool()
            labels = target['membership'][0, valid]
            features = outputs['actor_embeddings'][b, valid]
            sim = F.cosine_similarity(features[:, None], features[None, :], dim=-1) / self.temperature
            for i, label in enumerate(labels):
                positive = (labels == label) & (torch.arange(len(labels), device=labels.device) != i)
                negative = labels != label
                if label >= 0 and positive.any():
                    losses.append(torch.logsumexp(sim[i, positive | negative], 0) - torch.logsumexp(sim[i, positive], 0))
        return {'loss_consistency': torch.stack(losses).mean() if losses else outputs['actor_embeddings'].sum()*0}
