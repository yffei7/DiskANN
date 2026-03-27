#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAMP="${1:-$(date -u +%Y%m%d-%H%M%S)}"

run_one() {
  local dataset="$1"
  local profile="$2"
  local run_id="${dataset}-${STAMP}"
  echo "[$(date -u +%FT%TZ)] dataset=${dataset} profile=${profile} run_id=${run_id}"
  if "$ROOT/run_datasets.sh" \
    --profile "$profile" \
    --datasets "$dataset" \
    --workloads "query_only,update_only,query_update_round" \
    --run-id "$run_id"; then
    echo "[$(date -u +%FT%TZ)] completed dataset=${dataset} run_id=${run_id}"
    return 0
  fi
  echo "[$(date -u +%FT%TZ)] failed dataset=${dataset} run_id=${run_id}" >&2
  return 1
}

failures=()

run_one "arxiv" "real_standard" || failures+=("arxiv")
run_one "sift" "real_standard" || failures+=("sift")
run_one "ms_turing" "real_100m" || failures+=("ms_turing")
run_one "spacev" "real_100m" || failures+=("spacev")
run_one "sift-100M" "real_100m" || failures+=("sift-100M")

if ((${#failures[@]} > 0)); then
  echo "[$(date -u +%FT%TZ)] failed datasets: ${failures[*]}" >&2
  exit 1
fi
