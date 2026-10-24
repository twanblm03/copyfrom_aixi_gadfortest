"""Input and CLI contracts for the frozen SAM image-encoder experiment.

This module does not import segment_anything or initialize a model.  The data
loader only resizes RGB to an unpadded uint8 view; SAM normalization, padding,
and feature extraction belong to the model backbone in the main process.
"""
import hashlib
from collections.abc import Mapping
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch


SAM_BACKBONE = 'sam_vit_b'
WORKTREE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAM_REPO = WORKTREE_ROOT / '.local_deps' / 'segment-anything'
DEFAULT_SAM_CHECKPOINT = (
    WORKTREE_ROOT / '.local_data' / 'sam' / 'checkpoints' / 'sam_vit_b_01ec64.pth')


def is_sam_backbone(args):
    """Whether an args namespace (or mapping) selects our SAM-only backbone."""
    value = args.get('backbone', '') if isinstance(args, Mapping) else getattr(args, 'backbone', '')
    return value == SAM_BACKBONE


def add_sam_encoder_arguments(parser):
    parser.add_argument('--sam_repo', default=str(DEFAULT_SAM_REPO),
                        help='local original segment-anything source directory')
    parser.add_argument('--sam_checkpoint', default=str(DEFAULT_SAM_CHECKPOINT),
                        help='original SAM ViT-B checkpoint; loaded locally without downloads')
    parser.add_argument('--sam_checkpoint_sha256', default='',
                        help='optional expected SHA256 for the original SAM checkpoint')
    parser.add_argument('--sam_encoder_chunk_size', default=1, type=int,
                        help='maximum frames per frozen SAM encoder call')
    parser.add_argument('--sam_output_height', default=23, type=int,
                        help='SAM valid-view feature grid height exposed to GAD')
    parser.add_argument('--sam_output_width', default=40, type=int,
                        help='SAM valid-view feature grid width exposed to GAD')


def validate_sam_encoder_args(args):
    """Fail before data/GPU setup for unsupported settings or missing assets.

    Non-SAM runs are deliberately untouched.  Call this after parsing CLI
    arguments (never while registering them), so --help needs no SAM assets.
    """
    if not is_sam_backbone(args):
        return
    if not getattr(args, 'research_b0_repro', False):
        raise ValueError('--backbone sam_vit_b requires --research_b0_repro')
    if not getattr(args, 'no_mae', False):
        raise ValueError('--backbone sam_vit_b requires --no_mae')
    if getattr(args, 'heatmap_variant', 'off') != 'off':
        raise ValueError('The frozen SAM encoder experiment requires --heatmap_variant off')
    if getattr(args, 'train_mask_mode', 'disabled') != 'disabled':
        raise ValueError('The SAM encoder experiment requires --train_mask_mode disabled')
    if getattr(args, 'dilation', False):
        raise ValueError('--dilation is not supported by the SAM encoder experiment')
    if getattr(args, 'unfreeze_blocks', 0) != 0:
        raise ValueError('The SAM encoder is frozen; --unfreeze_blocks must be 0')
    for name in ('sam_encoder_chunk_size', 'sam_output_height', 'sam_output_width',
                 'image_height', 'image_width'):
        value = getattr(args, name, None)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError('--%s must be a positive integer' % name)

    expected_sha = str(getattr(args, 'sam_checkpoint_sha256', '')).strip().lower()
    if expected_sha and not re.fullmatch(r'[0-9a-f]{64}', expected_sha):
        raise ValueError('--sam_checkpoint_sha256 must contain exactly 64 hexadecimal characters')
    args.sam_checkpoint_sha256 = expected_sha
    args.sam_repo = str(Path(args.sam_repo).expanduser().resolve())
    args.sam_checkpoint = str(Path(args.sam_checkpoint).expanduser().resolve())
    package_root = Path(args.sam_repo) / 'segment_anything'
    for relative_path in ('__init__.py', 'modeling/image_encoder.py', 'build_sam.py'):
        if not (package_root / relative_path).is_file():
            raise FileNotFoundError(
                'Missing original SAM source %s; set --sam_repo to a local segment-anything checkout'
                % (package_root / relative_path))
    checkpoint_path = Path(args.sam_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError('Missing SAM ViT-B checkpoint: %s' % checkpoint_path)
    if expected_sha:
        digest = hashlib.sha256()
        with checkpoint_path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha:
            raise ValueError('SAM checkpoint SHA256 mismatch: expected %s, got %s'
                             % (expected_sha, actual_sha))
    # First implementation always freezes the full encoder, with no LoRA.
    args.freeze_backbone = True


class SAMRGBTransform:
    """Resize PIL RGB using configured B0 geometry, yielding uint8 [3,H,W].

    image_hw is the configured (height, width), not the source image size.  We
    apply the same fixed resize geometry as B0 even if a source frame has a
    different aspect ratio.  The configured long side is mapped to 1024 using
    SAM's rounded resize-size convention.  No normalization or padding occurs
    in data-loader workers.
    """
    def __init__(self, image_hw, long_side=1024):
        if len(image_hw) != 2 or any(int(value) != value or value < 1 for value in image_hw):
            raise ValueError('image_hw must contain positive integer height and width')
        if not isinstance(long_side, int) or isinstance(long_side, bool) or long_side < 1:
            raise ValueError('long_side must be a positive integer')
        self.image_hw = tuple(int(value) for value in image_hw)
        self.long_side = long_side
        scale = float(long_side) / max(self.image_hw)
        self.resized_hw = tuple(max(1, int(value * scale + .5)) for value in self.image_hw)
        self.input_hw = self.resized_hw

    def __call__(self, image):
        image = image.convert('RGB')
        resampling = getattr(Image, 'Resampling', Image).BILINEAR
        resized = image.resize((self.resized_hw[1], self.resized_hw[0]), resample=resampling)
        pixels = np.array(resized, dtype=np.uint8, copy=True)
        return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


def _saved_config(checkpoint):
    config = {}
    for name in ('model_config', 'args'):
        value = checkpoint.get(name, {})
        if not isinstance(value, Mapping) and hasattr(value, '__dict__'):
            value = vars(value)
        if isinstance(value, Mapping):
            config.update(value)
    return config


def sam_resume_config(args):
    """Small serializable model contract stored alongside resumable weights."""
    fields = ('backbone', 'research_b0_repro', 'num_frame', 'num_class', 'num_boxes',
              'hidden_dim', 'crop_size', 'num_group_tokens', 'gar_nheads',
              'gar_enc_layers', 'gar_ffn_dim', 'position_embedding', 'no_mae',
              'heatmap_variant', 'train_mask_mode', 'image_height', 'image_width',
              'sam_output_height', 'sam_output_width')
    return {name: getattr(args, name) for name in fields if hasattr(args, name)}


def validate_sam_resume(args, checkpoint):
    """Reject SAM/other-backbone cross-loading before strict state loading.

    Historical checkpoints have no saved args.  For those, identify SAM by its
    image_encoder state keys.  Other historical B0 checkpoint behavior stays
    unchanged; PyTorch's strict loader still verifies all keys and shapes.
    """
    if not isinstance(checkpoint, Mapping):
        if is_sam_backbone(args):
            raise TypeError('SAM resume requires a checkpoint mapping')
        return
    config = _saved_config(checkpoint)
    state = checkpoint.get('state_dict', checkpoint)
    keys = state.keys() if isinstance(state, Mapping) else ()
    prefixes = ('backbone.0.image_encoder.', 'module.backbone.0.image_encoder.')
    state_is_sam = any(str(key).startswith(prefixes) for key in keys)
    saved_backbone = config.get('backbone')
    saved_is_sam = saved_backbone == SAM_BACKBONE if saved_backbone else state_is_sam
    current_is_sam = is_sam_backbone(args)
    if current_is_sam != saved_is_sam:
        raise ValueError('Checkpoint backbone %r cannot load into --backbone %s; '
                         'SAM requires a same-architecture resume, not B0 warm-start'
                         % (saved_backbone or ('sam_vit_b' if state_is_sam else 'non-SAM'),
                            getattr(args, 'backbone', '')))
    if not current_is_sam:
        return
    if saved_backbone and not state_is_sam:
        raise ValueError('Checkpoint declares sam_vit_b but has no SAM image_encoder weights')
    for name, current in sam_resume_config(args).items():
        if name in config and config[name] != current:
            raise ValueError('SAM resume model setting %s differs: saved %r, requested %r'
                             % (name, config[name], current))
    saved_metadata = checkpoint.get('sam_encoder_metadata', {})
    current_metadata = getattr(args, 'sam_encoder_metadata', {})
    saved_sha = (saved_metadata.get('checkpoint_sha256')
                 if isinstance(saved_metadata, Mapping) else None)
    current_sha = (current_metadata.get('checkpoint_sha256')
                   if isinstance(current_metadata, Mapping) else None)
    verified_sha = bool(saved_sha and current_sha)
    if verified_sha and str(saved_sha).lower() != str(current_sha).lower():
        raise ValueError('SAM resume checkpoint source SHA256 differs from the '
                         'currently loaded frozen encoder; metadata would misstate its source')
    return {'scope': ('architecture_and_checkpoint_sha256' if verified_sha else 'architecture_only'),
            'saved_encoder_metadata_available': bool(saved_sha)}
