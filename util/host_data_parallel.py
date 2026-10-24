"""Explicit host-staged DataParallel transport for broken CUDA peer copies.

Only tensor transport changes: native replication, per-device forward,
batch sharding, gathered global loss and NCCL gradient reduction are retained.
NCCL_P2P_DISABLE=1 is required separately for parameter/gradient collectives.
"""
import os
import torch


def map_tensors(value, function):
    if isinstance(value, torch.Tensor):
        return function(value)
    if isinstance(value, dict):
        return type(value)((key, map_tensors(item, function)) for key,item in value.items())
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return type(value)(*(map_tensors(item, function) for item in value))
    if isinstance(value, tuple):
        return tuple(map_tensors(item, function) for item in value)
    if isinstance(value, list):
        return [map_tensors(item, function) for item in value]
    return value


class HostStagedDataParallel(torch.nn.DataParallel):
    """Native DataParallel with CPU-staged scatter/gather (autograd preserved)."""
    def __init__(self, *args, **kwargs):
        if os.environ.get('NCCL_P2P_DISABLE') != '1':
            raise RuntimeError('host transport requires NCCL_P2P_DISABLE=1 for collectives')
        super().__init__(*args, **kwargs)

    def scatter(self, inputs, kwargs, device_ids):
        # Blocking D2H copies avoid all peer reads for input distribution.
        inputs = map_tensors(inputs, lambda tensor: tensor.cpu())
        kwargs = map_tensors(kwargs, lambda tensor: tensor.cpu())
        return super().scatter(inputs, kwargs, device_ids)

    def gather(self, outputs, output_device):
        # Gather's own autograd backward scatters CPU gradients directly to
        # each replica, avoiding peer copies in BOTH forward and backward.
        host_outputs = super().gather(outputs, -1)
        destination = torch.device('cuda', output_device)
        return map_tensors(host_outputs, lambda tensor: tensor.to(destination))
