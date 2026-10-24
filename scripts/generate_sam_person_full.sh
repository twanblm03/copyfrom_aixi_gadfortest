#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data2/mzy/GAD/.miniconda3/envs/gad/bin/python}"
manifest="${SAM_MANIFEST:-${repo_root}/.local_data/sam/full_manifest.jsonl}"
cache_root="${SAM_CACHE_ROOT:-${repo_root}/.local_data/sam/cache}"
sam_repo="${SAM_REPO:-${repo_root}/.local_deps/segment-anything}"
checkpoint="${SAM_CHECKPOINT:-${repo_root}/.local_data/sam/checkpoints/sam_vit_b_01ec64.pth}"
gpus="${GPUS:-0,1,2,3}"
verify="${VERIFY:-1}"
log_dir="${SAM_LOG_DIR:-${repo_root}/.local_data/sam}"
verify_log="${SAM_VERIFY_LOG:-${log_dir}/full_verify.log}"

[[ -f "${manifest}" ]] || { echo "Missing manifest: ${manifest}" >&2; exit 2; }
IFS=',' read -r -a gpu_list <<< "${gpus}"
[[ "${#gpu_list[@]}" -eq 4 ]] || { echo "Exactly four GPUs are required" >&2; exit 2; }
mkdir -p "${log_dir}"
cd "${repo_root}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

pids=()
for shard in 0 1 2 3; do
  log="${log_dir}/generation_shard_${shard}.log"
  CUDA_VISIBLE_DEVICES="${gpu_list[shard]}" "${python_bin}" -u tools/sam_person_full.py generate \
    --manifest "${manifest}" --cache-root "${cache_root}" \
    --sam-repo "${sam_repo}" --checkpoint "${checkpoint}" \
    --shard "${shard}" --shards 4 > "${log}" 2>&1 &
  pids+=("$!")
  echo "Started shard ${shard} on GPU ${gpu_list[shard]}: PID ${pids[shard]}, log ${log}"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one SAM generation shard failed; inspect generation_shard_*.log" >&2
  exit 1
fi

pids=()
for shard in 0 1 2 3; do
  log="${log_dir}/projection_shard_${shard}.log"
  "${python_bin}" -u tools/sam_person_full.py project \
    --manifest "${manifest}" --cache-root "${cache_root}" \
    --sam-repo "${sam_repo}" --checkpoint "${checkpoint}" \
    --shard "${shard}" --shards 4 > "${log}" 2>&1 &
  pids+=("$!")
  echo "Started feature projection shard ${shard}: PID ${pids[shard]}, log ${log}"
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one feature projection shard failed; inspect projection_shard_*.log" >&2
  exit 1
fi

if [[ "${verify}" == "1" ]]; then
  "${python_bin}" -u tools/sam_person_full.py verify \
    --manifest "${manifest}" --cache-root "${cache_root}" \
    --sam-repo "${sam_repo}" --checkpoint "${checkpoint}" \
    --projected \
    > "${verify_log}" 2>&1
  echo "Full cache verified: ${verify_log}"
fi
