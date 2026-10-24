#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

# Generate the CLI from the portable, reviewed configuration. --dry-run uses
# only the Python standard library and never imports the trainer or a model.
exec "${PYTHON:-python}" - "$@" <<'PY'
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path.cwd()
parser = argparse.ArgumentParser(
    description='H1-sup: place TRAIN->VAL or view TRAIN+VAL->TEST (best-on-TEST).')
parser.add_argument('split', choices=['place', 'view'], nargs='?', default='place')
parser.add_argument('--data-path', default=os.environ.get('DATA_PATH', 'Dataset'))
parser.add_argument('--tracks-path', default=os.environ.get('TRACKS_PATH', ''))
parser.add_argument('--groundtruth', default=os.environ.get('GROUNDTRUTH', 'evaluation/gt_tracks.txt'))
parser.add_argument('--device', default=os.environ.get('DEVICE', '0,1,2,3'))
parser.add_argument('--output-root', default=os.environ.get('OUTPUT_ROOT', ''))
parser.add_argument('--tag', default='')
parser.add_argument('--target-mode', choices=['box', 'crowd'], default='box',
                    help='box is standard H1-sup; crowd is an explicit experimental variant')
parser.add_argument('--dry-run', action='store_true', help='print resolved config and argv; do not train')
options = parser.parse_args()

config = json.loads((root / 'configs/h1_sup' / (options.split + '.args.json')).read_text())
output = options.output_root or ('outputs/h1_sup/' + options.split)
config.update(data_path=options.data_path, tracks_path=options.tracks_path,
              groundtruth=options.groundtruth, device=options.device,
              result_path=output, checkpoint_root=str(Path(output) / 'checkpoints'),
              heatmap_target_mode=options.target_mode,
              experiment_tag=options.tag or
              'h1_sup_%s_%s_dp4_lr1_noclip_seed1' % (options.split, options.target_mode))

evaluator_sha256 = hashlib.sha256((root / 'evaluation/cafe_eval.py').read_bytes()).hexdigest()
expected = 'f1cd5f05b63ec769c149ad1b8926922eb03bb70ab3665fb7e6bf75cec28b40a0'
if evaluator_sha256 != expected:
    parser.error('evaluation/cafe_eval.py differs from the reviewed corrected evaluator')

derived = {'protocol', 'eval_split', 'use_mae', 'mae_dim'}
argv = []
for key, value in config.items():
    if key in derived:
        continue
    if key == 'val_mode':
        argv.append('--val_mode' if value else '--final_test')
    elif isinstance(value, bool):
        if value:
            argv.append('--' + key)
    else:
        argv.extend(['--' + key, str(value)])

if options.dry_run:
    print(json.dumps({'resolved_config': config, 'evaluator_sha256': evaluator_sha256,
                      'command': [sys.executable, 'train.py'] + argv}, indent=2))
    sys.exit(0)

# Validate required local assets before letting train.py initialize a GPU.
data = Path(config['data_path']) / 'cafe'
tracks = Path(config['tracks_path']) if config['tracks_path'] else data / 'gt_tracks.pkl'
for path in [data, tracks, Path(config['groundtruth']), Path(config['labelmap'])]:
    if not path.exists():
        parser.error('missing required dataset/evaluation asset: %s' % path)
os.execv(sys.executable, [sys.executable, str(root / 'train.py')] + argv)
PY
