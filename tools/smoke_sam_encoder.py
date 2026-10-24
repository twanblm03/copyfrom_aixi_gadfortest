"""One image, real SAM weights and one GAD optimization step; CPU by default.

This is an engineering check with synthetic GAD labels, not a dataset score or
a full training run. An optional RGB image replaces the synthetic input image.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import torch

from models import build_model
from models.sam_backbone import file_sha256
from util.sam_encoder import (SAMRGBTransform, add_sam_encoder_arguments,
                              validate_sam_encoder_args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_sam_encoder_arguments(parser)
    parser.add_argument('--image', default='', help='optional local RGB image; labels remain synthetic')
    parser.add_argument('--device', default='cpu', help='cpu, or an explicitly allocated CUDA device')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', default=str(ROOT / 'results/sam_encoder/smoke.json'))
    cli = parser.parse_args()
    if cli.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(cli.threads)
    torch.manual_seed(1)
    config = json.loads((ROOT / 'configs/heatmap/b0.json').read_text())
    config.update(vars(cli))
    config.update(backbone='sam_vit_b', research_b0_repro=True, no_mae=True,
                  use_mae=False, criterion_on_cpu=True, train_mask_mode='disabled',
                  heatmap_variant='off', num_frame=1, unfreeze_blocks=0)
    args = argparse.Namespace(**config)
    validate_sam_encoder_args(args)
    device = torch.device(cli.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was explicitly requested but is unavailable')
    if cli.image:
        with Image.open(cli.image) as source:
            image = source.convert('RGB')
        image_source = str(Path(cli.image).resolve())
    else:
        y, x = np.mgrid[:720, :1280]
        image = Image.fromarray(np.stack((x % 256, y % 256, (x + y) % 256), -1).astype(np.uint8))
        image_source = 'synthetic RGB spatial ramp'
    pixels = SAMRGBTransform((args.image_height, args.image_width))(image)[None, None].to(device)
    # Four valid actors: a two-person group and two individuals; others padded.
    boxes = torch.zeros(1, 1, args.num_boxes, 4, device=device)
    boxes[0, 0, :4] = torch.tensor([[.25, .30, .12, .3], [.40, .35, .14, .3],
                                          [.65, .55, .15, .4], [.82, .45, .12, .35]], device=device)
    actions = torch.full((1, args.num_boxes), 7, dtype=torch.long, device=device)
    actions[0, :4] = torch.tensor([0, 0, 6, 6], device=device)
    member = torch.zeros(1, 1, args.num_boxes, device=device)
    member[0, 0, :2] = 1
    membership = torch.full((1, args.num_boxes), -1, dtype=torch.long, device=device)
    membership[0, :2] = 0
    valid = torch.zeros(1, 1, args.num_boxes, dtype=torch.bool, device=device)
    valid[0, 0, :4] = True
    target = dict(actions=actions, activities=torch.tensor([[0]], device=device),
                  members=member, membership=membership, boxes=boxes[0],
                  actor_frame_valid=valid[0])

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    print('Building the actual frozen SAM encoder and GAD...', flush=True)
    start = time.monotonic()
    model, criterion = build_model(args)
    model, criterion = model.to(device), criterion.to(device)
    build_seconds = time.monotonic() - start
    backbone = model.backbone[0]
    assert model.backbone.num_channels == 256
    assert tuple(model.fc_emb.weight.shape) == (256, 6400)
    assert tuple(model.input_proj.weight.shape) == (256, 256, 1, 1)
    assert not any('ResNet' in type(module).__name__ for module in model.modules())
    model.train()
    assert not backbone.image_encoder.training
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    before = model.input_proj.weight.detach().clone()
    print('Running one image through SAM, then the real GAD criterion/backward...', flush=True)
    sync()
    start = time.monotonic()
    outputs = model(pixels, boxes, actions.eq(7), actor_frame_valid=valid)
    sync()
    forward_seconds = time.monotonic() - start
    terms = criterion(outputs, [target], log=False)
    loss = sum(terms[name] * weight for name, weight in criterion.weight_dict.items())
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite smoke loss')
    start = time.monotonic()
    loss.backward()
    gradients = {}
    for prefix in ('fc_emb', 'input_proj', 'group_transformer'):
        grads = [p.grad for name, p in model.named_parameters()
                 if name.startswith(prefix) and p.grad is not None]
        if not grads or not all(torch.isfinite(grad).all() for grad in grads):
            raise RuntimeError('missing/nonfinite gradient in ' + prefix)
        gradients[prefix] = sum(float(grad.detach().square().sum()) for grad in grads) ** .5
        if gradients[prefix] <= 0:
            raise RuntimeError('zero gradient in ' + prefix)
    assert all(not p.requires_grad and p.grad is None for p in backbone.image_encoder.parameters())
    optimizer.step()
    sync()
    backward_seconds = time.monotonic() - start
    update = float((model.input_proj.weight.detach() - before).abs().max())
    assert update > 0
    result = dict(
        passed=True, scope='one-frame engineering optimization check; synthetic labels, no dataset metric',
        image_source=image_source, device=str(device), torch_version=torch.__version__,
        backbone=backbone.metadata, feature_shapes=backbone.last_feature_shapes,
        output_shapes={name: list(value.shape) for name, value in outputs.items()},
        loss=float(loss.detach()), loss_terms={name: float(value.detach()) for name, value in terms.items()},
        gradient_norms=gradients, input_proj_max_update=update, encoder_has_grad=False,
        encoder_trainable_parameters=sum(p.numel() for p in backbone.image_encoder.parameters() if p.requires_grad),
        gad_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        source_sha256={name: file_sha256(ROOT / name) for name in (
            'models/sam_backbone.py', 'models/backbone.py', 'models/b0_repro.py',
            'util/sam_encoder.py', 'tools/smoke_sam_encoder.py')},
        timings_seconds=dict(build=build_seconds, forward=forward_seconds, backward_step=backward_seconds))
    if device.type == 'cuda':
        result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, result=str(output), loss=result['loss'],
                          timings_seconds=result['timings_seconds'])), flush=True)


if __name__ == '__main__':
    main()
