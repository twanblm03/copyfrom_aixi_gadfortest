"""Epoch-boundary recovery for recorded full heatmap training."""
import os
from pathlib import Path
import random

import numpy as np
import torch


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def capture_rng(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), loader=generator.get_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def validate_resume(checkpoint, args, steps_per_epoch):
    required = {'state_dict', 'optimizer', 'epoch', 'metrics', 'args'}
    if not required.issubset(checkpoint):
        raise ValueError('Resume requires a complete epoch checkpoint: ' + str(required - set(checkpoint)))
    saved = checkpoint['args']
    # Probe output does not change training. All actual training settings must match.
    changed = {key: (saved.get(key), value) for key, value in vars(args).items()
               if key != 'attention_probe' and saved.get(key) != value}
    if changed:
        raise ValueError('Resume configuration differs (saved, requested): ' + str(changed))
    epoch = checkpoint['epoch']
    if not isinstance(epoch, int) or not 0 < epoch < args.epochs:
        raise ValueError('Resume epoch must precede the requested final epoch')
    step = checkpoint.get('step', epoch * steps_per_epoch)
    if step != epoch * steps_per_epoch:
        raise ValueError('Resume requires a full epoch boundary with unchanged TRAIN size and batch')
    return epoch, step


def restore_training_state(checkpoint, optimizer, args, steps_per_epoch, generator):
    epoch, step = validate_resume(checkpoint, args, steps_per_epoch)
    optimizer.load_state_dict(checkpoint['optimizer'])
    rng = checkpoint.get('rng_state')
    if rng:
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['torch'].cpu())
        generator.set_state(rng['loader'].cpu())
        if torch.cuda.is_available():
            if len(rng['cuda']) != torch.cuda.device_count():
                raise ValueError('Resume CUDA RNG allocation differs from saved allocation')
            torch.cuda.set_rng_state_all([state.cpu() for state in rng['cuda']])
    else:
        # Legacy checkpoints did not save random streams. Record this reset explicitly.
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_seed)
        generator.manual_seed(args.random_seed)
    best = float(checkpoint.get('best_metric', checkpoint['metrics'][args.selection_metric]))
    return epoch, step, best, bool(rng)


def training_checkpoint(core, optimizer, args, epoch, step, metrics, best, best_path, generator):
    return dict(state_dict=core.state_dict(), optimizer=optimizer.state_dict(),
                args=vars(args), epoch=epoch, step=step, metrics=metrics,
                best_metric=best, best_checkpoint=str(Path(best_path).resolve()),
                rng_state=capture_rng(generator), checkpoint_format=2)
