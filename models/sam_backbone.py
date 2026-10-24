"""Frozen SAM ViT-B image features for both GAD visual branches.

The dataset supplies RGB uint8 tensors resized to SAM's longest-side view.
Only the image encoder is built; no prompt encoder, mask decoder or ResNet
is present. Padding is removed from the output view before GAD sees it.
"""
import contextlib
import hashlib
import importlib
import inspect
from functools import partial
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _build_image_encoder(repo):
    repo = Path(repo).resolve()
    source = repo / 'segment_anything/modeling/image_encoder.py'
    if not source.is_file():
        raise FileNotFoundError('SAM image encoder source is missing: %s' % source)
    loaded = sys.modules.get('segment_anything')
    if loaded is not None:
        location = Path(loaded.__file__).resolve()
        if repo not in location.parents:
            raise ValueError('segment_anything is already imported from another repository: %s' % location)
    sys.path.insert(0, str(repo))
    try:
        module = importlib.import_module('segment_anything.modeling.image_encoder')
    finally:
        sys.path.pop(0)
    return module.ImageEncoderViT(
        img_size=1024, patch_size=16, in_chans=3, embed_dim=768,
        depth=12, num_heads=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), qkv_bias=True,
        use_rel_pos=True, global_attn_indexes=[2, 5, 8, 11],
        window_size=14, out_chans=256)


def align_sam_features(features, resized_hw, output_hw, image_size=1024,
                       patch_size=16):
    """Map the actual image rectangle into an unpadded GAD feature grid.

    Patch-aligned views use exact crop + bilinear resize. Other aspect ratios
    use pixel-center sampling, retaining fractional patch coverage rather than
    rounding the valid rectangle to an incorrect number of complete tokens.
    """
    h, w = (int(value) for value in resized_hw)
    oh, ow = (int(value) for value in output_hw)
    if not (0 < h <= image_size and 0 < w <= image_size and oh > 0 and ow > 0):
        raise ValueError('invalid SAM input/output image geometry')
    grid_size = image_size // patch_size
    if features.ndim != 4 or tuple(features.shape[-2:]) != (grid_size, grid_size):
        raise ValueError('SAM features must use the full padded image grid')
    if h % patch_size == 0 and w % patch_size == 0:
        valid = features[..., :h // patch_size, :w // patch_size]
        return F.interpolate(valid, size=(oh, ow), mode='bilinear', align_corners=False)
    xs = (torch.arange(ow, device=features.device, dtype=features.dtype) + .5)
    ys = (torch.arange(oh, device=features.device, dtype=features.dtype) + .5)
    xs = 2 * xs * w / (ow * image_size) - 1
    ys = 2 * ys * h / (oh * image_size) - 1
    grid = torch.stack((xs[None].expand(oh, ow), ys[:, None].expand(oh, ow)), -1)
    return F.grid_sample(features, grid[None].expand(features.shape[0], -1, -1, -1),
                         mode='bilinear', padding_mode='border', align_corners=False)


class SAMImageBackbone(nn.Module):
    num_channels = 256
    image_size = 1024
    patch_size = 16

    def __init__(self, args):
        super().__init__()
        self.output_hw = (int(getattr(args, 'sam_output_height', 23)),
                          int(getattr(args, 'sam_output_width', 40)))
        self.chunk_size = int(getattr(args, 'sam_encoder_chunk_size', 1))
        if self.chunk_size < 1 or min(self.output_hw) < 1:
            raise ValueError('SAM encoder chunk and output dimensions must be positive')
        if getattr(args, 'unfreeze_blocks', 0):
            raise ValueError('This SAM feature experiment uses a frozen image encoder')
        repo = Path(args.sam_repo).resolve()
        checkpoint_path = Path(args.sam_checkpoint).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError('SAM pretrained checkpoint is missing: %s' % checkpoint_path)
        checkpoint_hash = file_sha256(checkpoint_path)
        expected = getattr(args, 'sam_checkpoint_sha256', '')
        if expected and checkpoint_hash.lower() != expected.lower():
            raise ValueError('SAM checkpoint SHA256 does not match --sam_checkpoint_sha256')
        # SAM architecture initialization must not advance the GAD RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.image_encoder = _build_image_encoder(repo)
        load_options = dict(map_location='cpu')
        if 'weights_only' in inspect.signature(torch.load).parameters:
            load_options['weights_only'] = True
        state = torch.load(str(checkpoint_path), **load_options)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if not isinstance(state, dict):
            raise ValueError('SAM checkpoint must contain a tensor state dictionary')
        if state and all(key.startswith('module.') for key in state):
            state = {key[len('module.'):]: value for key, value in state.items()}
        encoder_state = {key[len('image_encoder.'):]: value for key, value in state.items()
                         if key.startswith('image_encoder.')}
        checkpoint_format = 'full_sam'
        if not encoder_state:
            encoder_state = state
            checkpoint_format = 'image_encoder_only'
        # Strict loading rejects random/missing parameters or another SAM scale.
        self.image_encoder.load_state_dict(encoder_state, strict=True)
        self.image_encoder.requires_grad_(False)
        self.image_encoder.eval()
        self.register_buffer('pixel_mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))
        source = repo / 'segment_anything/modeling/image_encoder.py'
        self.metadata = dict(
            family='sam_v1_vit_b', feature='post_neck', channels=256,
            image_size=self.image_size, patch_size=self.patch_size,
            output_hw=list(self.output_hw), encoder_chunk_size=self.chunk_size,
            frozen=True, precision='fp32', input_format='RGB uint8 CHW, longest side 1024',
            checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
            checkpoint_format=checkpoint_format, loaded_encoder_tensors=len(encoder_state),
            sam_repo=str(repo), image_encoder_source_sha256=file_sha256(source),
            alignment='crop patch-aligned view; otherwise pixel-center grid_sample; bilinear',
            pretrained_initialization='strict image_encoder state_dict load')
        self.last_feature_shapes = None

    def train(self, mode=True):
        super().train(mode)
        self.image_encoder.eval()
        return self

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] < 1:
            raise ValueError('SAM backbone expects a nonempty [BT,3,H,W] image tensor')
        if images.dtype != torch.uint8:
            raise ValueError('SAM backbone requires RGB uint8 input; use SAMRGBTransform, not ImageNet normalization')
        h, w = images.shape[-2:]
        if max(h, w) != self.image_size:
            raise ValueError('SAMRGBTransform must resize the longest side to 1024 before batching')
        if next(self.image_encoder.parameters()).dtype != torch.float32:
            raise ValueError('This frozen SAM implementation requires FP32 encoder parameters')
        outputs = []
        self.image_encoder.eval()
        autocast = (torch.cuda.amp.autocast(enabled=False) if images.is_cuda
                    else contextlib.nullcontext())
        with torch.no_grad(), autocast:
            for start in range(0, len(images), self.chunk_size):
                pixels = images[start:start + self.chunk_size].float()
                pixels = (pixels - self.pixel_mean) / self.pixel_std
                pixels = F.pad(pixels, (0, self.image_size - w, 0, self.image_size - h))
                features = self.image_encoder(pixels)
                if tuple(features.shape[1:]) != (256, 64, 64):
                    raise ValueError('Expected SAM ViT-B post-neck features [chunk,256,64,64]')
                outputs.append(align_sam_features(features, (h, w), self.output_hw))
        self.last_feature_shapes = dict(
            input=list(images.shape), padded_hw=[1024, 1024],
            native_feature_hw=[64, 64], valid_feature_hw=[h / 16., w / 16.],
            output=[len(images), 256] + list(self.output_hw))
        return torch.cat(outputs, dim=0)
