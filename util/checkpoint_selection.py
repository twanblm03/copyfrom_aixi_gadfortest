"""Metrics reported together must come from the primary selected epoch."""
def selected_epoch_metrics(previous, epoch, metrics, is_new_best):
    if not is_new_best:
        return previous
    keys = ('group_mAP_1.0', 'group_mAP_0.5', 'outlier_mIoU', 'loss')
    return {key: {'epoch': epoch, 'value': float(metrics[key])}
            for key in keys if key in metrics}
