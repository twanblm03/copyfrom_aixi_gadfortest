#!/usr/bin/env bash
set -euo pipefail

project_dir=/data2/mzy/GAD/CAFE_codebase
python_bin=/data2/mzy/GAD/.miniconda3/envs/gad/bin/python
data_path=/nas/mzy/dataset/Cafe_Dataset/Cafe_Dataset/Dataset
feature_dir="$project_dir/.local_data/videomae_v2_giant"
cache_dir="$project_dir/.local_data/huggingface"
smoke_feature="$feature_dir/1_0.npy"
smoke_unit=cafe-videomae-smoke-gpu2.service

cd "$project_dir"
mkdir -p "$feature_dir" "$cache_dir"

echo "Waiting for the GPU 2 smoke extraction to produce $smoke_feature"
while [[ ! -f "$smoke_feature" ]]; do
    smoke_state=$(systemctl --user is-active "$smoke_unit" || true)
    if [[ "$smoke_state" != "active" ]]; then
        echo "Smoke extraction stopped before producing a feature (state: $smoke_state)."
        exit 1
    fi
    sleep 30
done

"$python_bin" -c "import numpy as np; x = np.load('$smoke_feature'); assert x.shape == (1408,) and np.isfinite(x).all(), x.shape"
echo "Smoke feature validated. Starting two non-overlapping GPU workers."

common_args=(
    extract_videomae_v2_giant.py
    --data_path "$data_path"
    --dataset cafe
    --save_dir "$feature_dir"
    --device cuda:0
    --num_frames 16
    --log_every 25
)

HF_HOME="$cache_dir" TRANSFORMERS_CACHE="$cache_dir" HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=2 PYTHONUNBUFFERED=1 \
    "$python_bin" "${common_args[@]}" --videos 1,2,3,4,5,6,7,8,9,10,11,12 \
    > videomae_extract_gpu2.log 2>&1 &
gpu2_pid=$!

HF_HOME="$cache_dir" TRANSFORMERS_CACHE="$cache_dir" HF_ENDPOINT=https://hf-mirror.com \
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 \
    "$python_bin" "${common_args[@]}" --videos 13,14,15,16,17,18,19,20,21,22,23,24 \
    > videomae_extract_gpu3.log 2>&1 &
gpu3_pid=$!

set +e
wait "$gpu2_pid"
gpu2_status=$?
wait "$gpu3_pid"
gpu3_status=$?
set -e

echo "GPU 2 worker exit: $gpu2_status; GPU 3 worker exit: $gpu3_status"
[[ "$gpu2_status" -eq 0 && "$gpu3_status" -eq 0 ]]

"$python_bin" scripts/verify_videomae_cafe_features.py \
    --data_path "$data_path" \
    --dataset cafe \
    --features_path "$feature_dir"
