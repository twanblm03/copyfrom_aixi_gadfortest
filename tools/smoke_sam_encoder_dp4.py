"""Two FP32 Adam steps with frozen SAM, T5 and batch16 on four Host DP GPUs.

This engineering check uses synthetic RGB and synthetic GAD labels with the
real model, Hungarian matcher and criterion. It produces no dataset score.
Running the script allocates GPUs; --help only describes the command.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEVICE_IDS = (0, 1, 2, 3)
SOURCE_FILES = (
    'tools/smoke_sam_encoder_dp4.py', 'train.py', 'models/__init__.py',
    'models/sam_backbone.py', 'models/backbone.py', 'models/b0_repro.py',
    'models/b0_transformer.py', 'models/criterion.py', 'models/group_matcher.py',
    'models/heatmap_criterion.py', 'util/sam_encoder.py',
    'util/host_data_parallel.py', 'dataloader/cafe.py', 'evaluation/cafe_eval.py',
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config):
    """Reject smaller or architecturally different smoke configurations."""
    fixed = dict(backbone='sam_vit_b', research_b0_repro=True,
                 no_mae=True, heatmap_variant='off', train_mask_mode='disabled',
                 num_frame=5, batch=16, num_boxes=14, num_class=6, hidden_dim=256,
                 crop_size=5, num_group_tokens=12, image_height=720,
                 image_width=1280, sam_output_height=23, sam_output_width=40,
                 parallel_transport='host', distributed=False, micro_batch=0,
                 optimizer='adam')
    for key, value in fixed.items():
        if key not in config or config[key] != value:
            raise ValueError('DP4 smoke requires %s=%r, got %r' %
                             (key, value, config.get(key)))
    if str(config.get('device', '')).replace(' ', '') != '0,1,2,3':
        raise ValueError('DP4 smoke requires --device 0,1,2,3 in frozen arguments')
    for key in ('lr', 'backbone_lr_scale'):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('DP4 smoke requires a finite positive %s' % key)
    if not math.isfinite(config.get('weight_decay', float('nan'))):
        raise ValueError('DP4 smoke requires finite weight_decay')


def synthetic_batch(args, device):
    """The full batch matches loader tensor shapes; labels cover six classes."""
    import torch
    from util.sam_encoder import SAMRGBTransform

    height, width = SAMRGBTransform((args.image_height, args.image_width)).input_hw
    yy = torch.arange(height, dtype=torch.int32)[:, None]
    xx = torch.arange(width, dtype=torch.int32)[None, :]
    ramp = torch.stack((xx.expand(height, width) % 256,
                        yy.expand(height, width) % 256,
                        (xx + yy) % 256)).to(torch.uint8)
    pixels = torch.empty(args.batch, args.num_frame, 3, height, width,
                         dtype=torch.uint8)
    for batch_index in range(args.batch):
        for frame_index in range(args.num_frame):
            # Spatial ramps vary across both clip and frame, deterministically.
            pixels[batch_index, frame_index] = ramp + 7 * batch_index + frame_index
    pixels = pixels.to(device)
    boxes = torch.zeros(args.batch, args.num_frame, args.num_boxes, 4, device=device)
    template = torch.tensor([[.25, .30, .12, .30], [.40, .35, .14, .30],
                             [.65, .55, .15, .40], [.82, .45, .12, .35]], device=device)
    boxes[:, :, :4] = template
    valid = torch.zeros(args.batch, args.num_frame, args.num_boxes,
                        dtype=torch.bool, device=device)
    valid[:, :, :4] = True
    all_actions = torch.full((args.batch, args.num_boxes), args.num_class + 1,
                             dtype=torch.long, device=device)
    targets = []
    for batch_index in range(args.batch):
        activity = batch_index % args.num_class
        all_actions[batch_index, :2] = activity
        all_actions[batch_index, 2:4] = args.num_class
        members = torch.zeros(1, 1, args.num_boxes, device=device)
        members[0, 0, :2] = 1
        membership = torch.full((1, args.num_boxes), -1, dtype=torch.long, device=device)
        membership[0, :2] = 0
        targets.append(dict(
            actions=all_actions[batch_index:batch_index + 1],
            activities=torch.tensor([[activity]], dtype=torch.long, device=device),
            members=members, membership=membership, boxes=boxes[batch_index],
            actor_frame_valid=valid[batch_index]))
    return pixels, boxes, all_actions.eq(args.num_class + 1), valid, targets


def gradient_report(module):
    import torch
    result = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError('Nonfinite gradient: ' + name)
    for prefix in ('fc_emb', 'input_proj', 'group_transformer'):
        gradients = [parameter.grad for name, parameter in module.named_parameters()
                     if name.startswith(prefix + '.') and parameter.grad is not None]
        if not gradients:
            raise RuntimeError('Missing gradient: ' + prefix)
        norm = math.sqrt(sum(float(grad.detach().double().square().sum())
                             for grad in gradients))
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError('Zero/nonfinite gradient: ' + prefix)
        result[prefix] = norm
    encoder = module.backbone[0].image_encoder
    if encoder.training or any(parameter.requires_grad or parameter.grad is not None
                               for parameter in encoder.parameters()):
        raise RuntimeError('SAM must remain eval, frozen and without gradients')
    return result


def run_smoke(config, report):
    # Set visibility before importing torch or querying CUDA.
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    os.environ['NCCL_P2P_DISABLE'] = '1'
    os.environ['NCCL_IB_DISABLE'] = '1'
    import numpy as np
    import torch
    from models import build_model
    from util.host_data_parallel import HostStagedDataParallel
    from util.sam_encoder import validate_sam_encoder_args

    args = argparse.Namespace(**config)
    validate_sam_encoder_args(args)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise RuntimeError('Exactly four allocated CUDA devices must be visible')
    torch.set_num_threads(4)
    seed = int(args.random_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    for index in DEVICE_IDS:
        torch.cuda.reset_peak_memory_stats(index)

    args.use_mae = False
    args.mae_dim = 0
    args.criterion_on_cpu = False
    started = time.monotonic()
    print('Building frozen SAM and GAD; four GPU batch16/T5 engineering smoke.', flush=True)
    module, criterion = build_model(args)
    module = module.to(device=device, dtype=torch.float32)
    criterion = criterion.to(device=device, dtype=torch.float32)
    if any('resnet' in type(item).__name__.lower() for item in module.modules()):
        raise RuntimeError('The SAM replacement must not instantiate ResNet')
    if tuple(module.fc_emb.weight.shape) != (256, 6400):
        raise RuntimeError('Unexpected SAM actor projection shape')
    if tuple(module.input_proj.weight.shape) != (256, 256, 1, 1):
        raise RuntimeError('Unexpected SAM scene projection shape')
    if any(parameter.dtype != torch.float32 for parameter in module.parameters()):
        raise RuntimeError('All model parameters must be FP32')
    model = HostStagedDataParallel(module, device_ids=list(DEVICE_IDS), output_device=0)
    model.train()
    criterion.train()
    if module.backbone[0].image_encoder.training:
        raise RuntimeError('model.train() enabled SAM training mode')
    head_params, backbone_params = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (backbone_params if 'backbone' in name else head_params).append(parameter)
    if backbone_params:
        raise RuntimeError('Frozen SAM parameters entered optimizer')
    # Match the trainer's Adam groups and first-epoch learning rates.
    optimizer = torch.optim.Adam([
        dict(params=head_params, lr=args.lr),
        dict(params=backbone_params, lr=args.lr * args.backbone_lr_scale),
    ], betas=(.9, .999), eps=1e-8, weight_decay=args.weight_decay)
    report.update(
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        devices=[dict(logical_index=index, name=torch.cuda.get_device_name(index),
                      total_memory_bytes=torch.cuda.get_device_properties(index).total_memory)
                 for index in DEVICE_IDS],
        backbone_metadata=module.backbone[0].metadata,
        projection_shapes=dict(fc_emb=list(module.fc_emb.weight.shape),
                               input_proj=list(module.input_proj.weight.shape)),
        parameters=dict(total=sum(parameter.numel() for parameter in module.parameters()),
                        trainable=sum(parameter.numel() for parameter in head_params)),
        optimizer=dict(name='Adam', lr=args.lr, betas=[.9, .999], eps=1e-8,
                       weight_decay=args.weight_decay, parameter_groups=2),
        build_seconds=time.monotonic() - started)
    inputs, boxes, dummy, valid, targets = synthetic_batch(args, device)
    report['input_shapes'] = dict(images=list(inputs.shape), boxes=list(boxes.shape),
                                  dummy_mask=list(dummy.shape), actor_frame_valid=list(valid.shape))
    report['input_dtype'] = str(inputs.dtype)
    # Hooks run on each replica; require the intended four clips/T5 on all GPUs.
    calls = {index: [] for index in DEVICE_IDS}

    def feature_hook(_backbone, arguments, output):
        index = output.device.index
        entry = dict(input=list(arguments[0].shape), output=list(output.shape))
        calls[index].append(entry)
        if (_backbone.image_encoder.training or output.requires_grad
                or any(parameter.requires_grad for parameter in _backbone.image_encoder.parameters())):
            raise RuntimeError('Replica SAM encoder must remain frozen and in eval mode')
        if entry['input'] != [20, 3, 576, 1024] or entry['output'] != [20, 256, 23, 40]:
            raise RuntimeError('Replica batch/frame/feature shape differs: %r' % entry)

    handle = module.backbone[0].register_forward_hook(feature_hook)

    def synchronize():
        for index in DEVICE_IDS:
            torch.cuda.synchronize(index)

    report['steps'] = []
    try:
        for step in (1, 2):
            print('DP4 smoke optimizer step %d/2, batch16/T5.' % step, flush=True)
            optimizer.zero_grad()
            before = module.input_proj.weight.detach().clone()
            synchronize()
            started = time.monotonic()
            outputs = model(inputs, boxes, dummy, actor_frame_valid=valid)
            if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                   for value in outputs.values()):
                raise FloatingPointError('Non-FP32/nonfinite smoke output')
            expected = dict(pred_actions=[16, 14, 7], pred_activities=[16, 12, 7],
                            membership=[16, 12, 14], actor_embeddings=[16, 14, 256])
            for name, shape in expected.items():
                if list(outputs[name].shape) != shape:
                    raise RuntimeError('Unexpected output shape: ' + name)
            terms = criterion(outputs, targets, log=False)
            if any(not bool(torch.isfinite(value).all()) for value in terms.values()):
                raise FloatingPointError('Nonfinite criterion term')
            loss = sum(terms[name] * weight for name, weight in criterion.weight_dict.items())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite smoke loss')
            loss.backward()
            gradients = gradient_report(module)
            optimizer.step()
            synchronize()
            update = float((module.input_proj.weight.detach() - before).abs().max())
            if not math.isfinite(update) or update <= 0:
                raise RuntimeError('No finite scene projection update')
            for index in DEVICE_IDS:
                if len(calls[index]) != step:
                    raise RuntimeError('GPU %d did not complete every smoke step' % index)
            report['steps'].append(dict(
                optimizer_step=step, loss=float(loss.detach()),
                loss_terms={name: float(value.detach()) for name, value in terms.items()},
                gradient_norms=gradients, input_proj_max_update=update,
                output_shapes={name: list(value.shape) for name, value in outputs.items()},
                elapsed_seconds=time.monotonic() - started))
            del outputs, terms, loss, before
    finally:
        handle.remove()
        synchronize()
        report['replica_feature_shapes'] = calls
        report['peak_memory_per_gpu'] = [dict(
            logical_index=index,
            allocated_bytes=torch.cuda.max_memory_allocated(index),
            reserved_bytes=torch.cuda.max_memory_reserved(index)) for index in DEVICE_IDS]
    report.update(encoder_has_grad=False, encoder_trainable_parameters=0,
                  verified_optimizer_steps=2, passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--args-json', required=True,
                        help='frozen complete train.py arguments for SAM encoder replacement')
    parser.add_argument('--output', required=True, help='engineering check result JSON')
    cli = parser.parse_args()
    args_path = Path(cli.args_json).resolve()
    output = Path(cli.output).resolve()
    if output == args_path or output.exists():
        parser.error('--output must be a fresh path distinct from --args-json')
    config = json.loads(args_path.read_text())
    validate_config(config)
    report = dict(
        passed=False,
        scope='four-GPU engineering optimization check; synthetic RGB/labels; no dataset metric',
        args_json=str(args_path), args_json_sha256=file_sha256(args_path),
        batch=16, num_frame=5, optimizer_steps=2, precision='fp32',
        parallel_transport='HostStagedDataParallel', device='0,1,2,3',
        source_sha256={name: file_sha256(ROOT / name) for name in SOURCE_FILES},
        synthetic_labels='one two-person group and two individuals per clip; six activity classes',
        started_unix_seconds=time.time())
    started = time.monotonic()
    try:
        run_smoke(config, report)
    except Exception as error:
        report['error'] = '%s: %s' % (type(error).__name__, error)
        report['traceback'] = traceback.format_exc()
        raise
    finally:
        report['total_seconds'] = time.monotonic() - started
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(passed=report['passed'], result=str(output),
                          optimizer_steps=report['verified_optimizer_steps'])), flush=True)


if __name__ == '__main__':
    main()
