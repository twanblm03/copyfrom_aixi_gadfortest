#!/usr/bin/env bash
set -euo pipefail
pilot_steps=8
pilot_seed=1
pilot_root=outputs/heatmap/place/paired_pilot8_rngfixed_seed1_20261012
pilot_gpus=0,1,2
pilot_python=.miniconda3/envs/gad/bin/python
pilot_workers=2
pilot_persistent=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) pilot_steps="$2"; shift 2 ;;
    --seed) pilot_seed="$2"; shift 2 ;;
    --root) pilot_root="$2"; shift 2 ;;
    --gpus) pilot_gpus="$2"; shift 2 ;;
    --python) pilot_python="$2"; shift 2 ;;
    --workers) pilot_workers="$2"; shift 2 ;;
    --persistent-workers) pilot_persistent=(--persistent-workers); shift ;;
    *) printf 'Unknown option: %s\n' "$1"; exit 2 ;;
  esac
done
for pilot_variant in b0 h1_sup h1_match h2_s; do
  "$pilot_python" tools/run_heatmap_research.py \
    --stage pilot --config "configs/heatmap/${pilot_variant}.json" \
    --steps "$pilot_steps" --seed "$pilot_seed" --train-clips 64 --eval-clips 32 \
    --workers "$pilot_workers" "${pilot_persistent[@]}" --device "cuda:${pilot_gpus%%,*}" --parallel-gpus "$pilot_gpus" \
    --output "${pilot_root}/${pilot_variant}"
done
