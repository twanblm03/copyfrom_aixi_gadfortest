"""Parameter-free filtering on the exact B0 RoIAlign grid."""
import torch


def filter_actor_rois(rois, boxes_feature_xyxy, masks, actor_valid, raw_valid,
                      roi_align, empty_eps=1e-6):
    if masks.ndim != 5:
        raise ValueError('expected masks[B,T,N,Hf,Wf]')
    b, t, n, h, w = masks.shape
    count = b*t*n
    if rois.shape[0] != count or tuple(actor_valid.shape) != (b, t, n):
        raise ValueError('mask/actor/RoI identity mismatch')
    if not torch.isfinite(masks).all() or (masks < 0).any() or (masks > 1).any():
        raise ValueError('mask weights must be finite coverage in [0,1]')
    # Each actor is its own single-channel image, preserving actor order.
    weights = roi_align(masks.detach().reshape(count, 1, h, w).to(rois),
                        [box[None] for box in boxes_feature_xyxy])
    eligible = actor_valid.reshape(-1).bool()
    failed = eligible & ~raw_valid.reshape(-1).bool()
    grid_empty = eligible & ~failed & (weights.mean((1, 2, 3)) <= empty_eps)
    use = eligible & ~failed & ~grid_empty
    weights = torch.where(use[:, None, None, None], weights, torch.ones_like(weights))
    stats = dict(valid=int(eligible.sum()), raw_failed=int(failed.sum()),
                 feature_grid_empty=int(grid_empty.sum()), used=int(use.sum()),
                 fallback=int((failed | grid_empty).sum()))
    return rois * weights, stats
