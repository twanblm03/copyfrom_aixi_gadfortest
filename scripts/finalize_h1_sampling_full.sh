#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
run_root="${repo_root}/.local_data/historical_lr_randomsegments"
status="${run_root}/comparison.status"
log="${run_root}/comparison.log"
python_bin="${PYTHON_BIN:-/data2/mzy/GAD/.miniconda3/envs/gad/bin/python}"
deadline=$(( $(date +%s) + 48 * 3600 ))

printf 'comparison_waiting %s\n' "$(date -Is)" > "${status}"
while (( $(date +%s) < deadline )); do
  b0_status="$(cat "${run_root}/b0/full_training.status")"
  h1_status="$(cat "${run_root}/h1_sup/full_training.status")"
  if [[ "${b0_status}" == training_failed* || "${h1_status}" == training_failed* ||
        "${b0_status}" == training_interrupted* || "${h1_status}" == training_interrupted* ]]; then
    printf 'comparison_blocked training_failed_or_interrupted %s\n' "$(date -Is)" > "${status}"
    exit 1
  fi
  if [[ "${b0_status}" == training_complete* && "${h1_status}" == training_complete* ]]; then
    break
  fi
  b0_pid="$(cat "${run_root}/b0/full_training.pid")"
  h1_pid="$(cat "${run_root}/h1_sup/full_training.pid")"
  if ! kill -0 "${b0_pid}" 2>/dev/null && ! kill -0 "${h1_pid}" 2>/dev/null; then
    # The comparison script validates full 30-epoch artifacts before reporting.
    break
  fi
  sleep 60
done

if (( $(date +%s) >= deadline )); then
  printf 'comparison_timeout %s\n' "$(date -Is)" > "${status}"
  exit 1
fi

printf 'comparison_running %s\n' "$(date -Is)" > "${status}"
if "${python_bin}" tools/compare_h1_sampling_full.py > "${log}" 2>&1; then
  printf 'comparison_complete %s\n' "$(date -Is)" > "${status}"
else
  code=$?
  printf 'comparison_failed code=%s %s\n' "${code}" "$(date -Is)" > "${status}"
  exit "${code}"
fi
