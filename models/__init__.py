from .group_matcher import build_group_matcher
from .criterion import SetCriterion
from .models import GADTR


def build_model(args):
    research = getattr(args, 'research_b0_repro', False)
    variant = getattr(args, 'heatmap_variant', 'off')
    if variant != 'off' and not research:
        raise ValueError('Heatmap experiments require --research_b0_repro')
    if research:
        from .b0_repro import GADTRB0Repro
        from .heatmap_criterion import HeatmapCriterion, HeatmapMatcher
        model = GADTRB0Repro(args)
    else:
        model = GADTR(args)

    losses = ['labels', 'cardinality']
    group_losses = ['group_labels', 'group_cardinality', 'group_code', 'group_consistency']

    # Set loss coefficients
    weight_dict = {}
    weight_dict['loss_ce'] = args.ce_loss_coef
    weight_dict['loss_group_ce'] = args.group_ce_loss_coef
    weight_dict['loss_group_code'] = args.group_code_loss_coef
    weight_dict['loss_consistency'] = args.consistency_loss_coef
    if research and variant != 'off':
        group_losses.append('heatmaps')
        weight_dict.update(loss_mask_focal=args.heatmap_focal_coef,
                           loss_mask_dice=args.heatmap_dice_coef,
                           loss_mask_empty=args.heatmap_empty_coef)

    # Group matching
    group_matcher = HeatmapMatcher(args) if research else build_group_matcher(args)

    # Loss functions
    criterion_type = HeatmapCriterion if research else SetCriterion
    criterion = criterion_type(args.num_class, weight_dict=weight_dict, eos_coef=args.eos_coef,
                             losses=losses, group_losses=group_losses, group_matcher=group_matcher, args=args)

    if not getattr(args, 'criterion_on_cpu', False):
        criterion.cuda()

    return model, criterion
