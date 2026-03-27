#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="real_standard"
DATASETS=""
SYSTEMS=""
WORKLOADS=""
RUN_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --systems)
      SYSTEMS="$2"
      shift 2
      ;;
    --workloads)
      WORKLOADS="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
fi

ARGS=(python3 "$ROOT/run_comparison.py" --profile "$PROFILE" --run-id "$RUN_ID")
if [[ -n "${DATASETS}" ]]; then
  ARGS+=(--datasets "$DATASETS")
fi
if [[ -n "${SYSTEMS}" ]]; then
  ARGS+=(--systems "$SYSTEMS")
fi
if [[ -n "${WORKLOADS}" ]]; then
  ARGS+=(--workloads "$WORKLOADS")
fi

echo "profile=$PROFILE"
echo "run_id=$RUN_ID"
if [[ -n "${DATASETS}" ]]; then echo "datasets=$DATASETS"; fi
if [[ -n "${SYSTEMS}" ]]; then echo "systems=$SYSTEMS"; fi
if [[ -n "${WORKLOADS}" ]]; then echo "workloads=$WORKLOADS"; fi

"${ARGS[@]}"

echo
echo "Summary CSV: $ROOT/../runs/$RUN_ID/results/summary.csv"
echo "Summary JSON: $ROOT/../runs/$RUN_ID/results/summary.json"
echo "Latest artifacts: $ROOT/../latest"
