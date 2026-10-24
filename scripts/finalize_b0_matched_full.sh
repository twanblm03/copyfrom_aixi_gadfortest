#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
run_root="${repo_root}/.local_data/b0_matched"
status="${run_root}/full_training.status"
report_status="${run_root}/comparison.status"
report_log="${run_root}/comparison.log"
python_bin="${PYTHON_BIN:-/data2/mzy/GAD/.miniconda3/envs/gad/bin/python}"

while true; do
  [[ -f "${status}" ]] || { echo "Missing training status" >&2; exit 2; }
  read -r state _ < "${status}"
  if [[ "${state}" == training_complete ]]; then
    break
  fi
  if [[ "${state}" != training_started ]]; then
    printf 'comparison_blocked training_state=%s %s\n' "${state}" "$(date -Is)" > "${report_status}"
    exit 1
  fi
  sleep 30
done

printf 'comparison_started %s\n' "$(date -Is)" > "${report_status}"
if "${python_bin}" tools/compare_b0_sam_full.py > "${report_log}" 2>&1; then
  printf 'comparison_complete %s\n' "$(date -Is)" > "${report_status}"
else
  code=$?
  printf 'comparison_failed code=%s %s\n' "${code}" "$(date -Is)" > "${report_status}"
  exit "${code}"
fi
