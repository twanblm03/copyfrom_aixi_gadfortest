#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-/data2/mzy/GAD/.miniconda3/envs/gad/bin/python}"
run_root="${repo_root}/.local_data/sam/dp4_randomsegments_lr1"
manifest="${repo_root}/.local_data/sam/randomsegments_manifest.jsonl"
status="${run_root}/queue.status"
mkdir -p "${run_root}/cache_build"
if [[ -e "${status}" ]]; then
  echo "Existing SAM queue status; refusing duplicate launch: ${status}" >&2
  exit 2
fi
printf '%s\n' "$$" > "${run_root}/queue.pid"
phase=manifest
printf 'manifest_started %s\n' "$(date -Is)" > "${status}"
on_exit() {
  code=$?
  if (( code != 0 )); then
    printf 'pipeline_failed phase=%s code=%s %s\n' \
      "${phase}" "${code}" "$(date -Is)" > "${status}"
  fi
}
trap on_exit EXIT

if [[ ! -f "${manifest}" ]]; then
  nice -n 10 "${python_bin}" -u tools/sam_person_full.py manifest \
    --train-all-frames --output "${manifest}" \
    --tracks-path /data2/mzy/GAD/CAFE_codebase/.local_data/cafe/gt_tracks.pkl \
    > "${run_root}/manifest.log" 2>&1
fi
"${python_bin}" - "${manifest}.summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
assert summary['clips'] == {'train': 6444, 'val': 1386}, summary['clips']
assert summary['frames'] == 223670, summary['frames']
assert summary['sampling'] == 'all TRAIN frame candidates; fixed VAL segment centers'
PY
printf 'manifest_complete %s\n' "$(date -Is)" > "${status}"

phase=waiting_for_gpus
printf 'waiting_for_gpus %s\n' "$(date -Is)" > "${status}"
while true; do
  mapfile -t gpu_free < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if [[ "${#gpu_free[@]}" -eq 4 ]]; then
    enough=1
    for free in "${gpu_free[@]}"; do
      if (( free < 30000 )); then enough=0; fi
    done
    if (( enough )); then break; fi
  fi
  sleep 60
done

phase=cache_generation
printf 'cache_generation_started %s\n' "$(date -Is)" > "${status}"
SAM_MANIFEST="${manifest}" \
SAM_LOG_DIR="${run_root}/cache_build" \
VERIFY=0 bash scripts/generate_sam_person_full.sh \
  > "${run_root}/cache_build/generate_and_project.log" 2>&1
printf 'cache_generation_complete %s\n' "$(date -Is)" > "${status}"

phase=cache_verification
printf 'cache_verification_started %s\n' "$(date -Is)" > "${status}"
"${python_bin}" -u tools/sam_person_full.py verify \
  --manifest "${manifest}" --projected \
  > "${run_root}/cache_build/verify.log" 2>&1
printf 'cache_verification_complete %s\n' "$(date -Is)" > "${status}"

phase=smoke
printf 'smoke_started %s\n' "$(date -Is)" > "${status}"
SMOKE_STEPS=2 SAM_TRAIN_SAMPLING=randomsegments \
  bash scripts/run_sam_dp4_full.sh
grep -q '^training_complete ' "${run_root}/smoke2/training.status"
printf 'smoke_complete %s\n' "$(date -Is)" > "${status}"

phase=full_training
printf 'full_training_started %s\n' "$(date -Is)" > "${status}"
SAM_TRAIN_SAMPLING=randomsegments bash scripts/run_sam_dp4_full.sh
grep -q '^training_complete ' "${run_root}/full/training.status"
printf 'full_training_complete %s\n' "$(date -Is)" > "${status}"
