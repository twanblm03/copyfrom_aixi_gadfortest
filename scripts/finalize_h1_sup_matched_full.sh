#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
run_root="${repo_root}/.local_data/h1_sup_matched"
mkdir -p "${run_root}"
report_status="${run_root}/comparison.status"
report_log="${run_root}/comparison.log"
python_bin="${PYTHON_BIN:-/data2/mzy/GAD/.miniconda3/envs/gad/bin/python}"

while true; do
  all_complete=1
  for run in b0_matched h1_sup_matched; do
    status="${repo_root}/.local_data/${run}/full_training.status"
    [[ -f "${status}" ]] || { echo "Missing ${status}" >&2; exit 2; }
    read -r state _ < "${status}"
    if [[ "${state}" == training_started ]]; then
      all_complete=0
    elif [[ "${state}" != training_complete ]]; then
      printf 'comparison_blocked run=%s training_state=%s %s\n' \
        "${run}" "${state}" "$(date -Is)" > "${report_status}"
      exit 1
    fi
  done
  if (( all_complete )); then
    break
  fi
  sleep 30
done

printf 'comparison_started %s\n' "$(date -Is)" > "${report_status}"
if "${python_bin}" tools/compare_b0_sam_h1_full.py > "${report_log}" 2>&1; then
  printf 'comparison_complete %s\n' "$(date -Is)" > "${report_status}"
else
  code=$?
  printf 'comparison_failed code=%s %s\n' "${code}" "$(date -Is)" > "${report_status}"
  exit "${code}"
fi
