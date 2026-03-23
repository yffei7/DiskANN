#!/usr/bin/env bash
# Compute exact brute-force K-NN ground truth for ANNS datasets.
# Uses DiskANN's compute_groundtruth binary (L2 distance).
# Saves output to each dataset's own directory, then updates
# the gt_file paths in run_disk_experiments.sh.
#
# Usage: bash scripts/compute_groundtruth.sh [dataset ...]
#   With no arguments all four datasets are processed.
#   Examples:
#     bash scripts/compute_groundtruth.sh
#     bash scripts/compute_groundtruth.sh arxiv sift

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_SCRIPT="$SCRIPT_DIR/run_disk_experiments.sh"

GT_BIN="$SCRIPT_DIR/../build/tests/utils/compute_groundtruth"
[[ -f "$GT_BIN" ]] || GT_BIN="$SCRIPT_DIR/../tests/utils/compute_groundtruth"

if [[ ! -f "$GT_BIN" ]]; then
    echo "ERROR: compute_groundtruth binary not found." >&2
    echo "       Build the project first (cmake --build build)." >&2
    exit 1
fi

K=10

# ---------------------------------------------------------------------------
# Dataset configuration
# type | base_file | query_file | new_gt_file | old_gt_file_in_script
# ---------------------------------------------------------------------------
declare -A TYPE BASE_FILE QUERY_FILE NEW_GT OLD_GT

TYPE["arxiv"]="float"
BASE_FILE["arxiv"]="/home/yfei/data/ANNS/arxiv/arxiv-base.bin"
QUERY_FILE["arxiv"]="/home/yfei/data/ANNS/arxiv/arxiv-query.bin"
NEW_GT["arxiv"]="/home/yfei/data/ANNS/arxiv/arxiv_groundtruth_K${K}_exact.diskann.bin"
OLD_GT["arxiv"]="/home/yfei/data/ANNS/arxiv/arxiv_groundtruth_K10.diskann.bin"

TYPE["sift"]="uint8"
BASE_FILE["sift"]="/home/yfei/data/ANNS/sift/sift_base.bin"
QUERY_FILE["sift"]="/home/yfei/data/ANNS/sift/sift_query.bin"
NEW_GT["sift"]="/home/yfei/data/ANNS/sift/sift_groundtruth_K${K}_exact.diskann.bin"
OLD_GT["sift"]="/home/yfei/data/ANNS/sift/sift_groundtruth_K10.bin"

TYPE["sift-100M"]="uint8"
BASE_FILE["sift-100M"]="/home/yfei/data/ANNS/sift-100M/learn.100M.u8bin"
QUERY_FILE["sift-100M"]="/home/yfei/data/ANNS/sift-100M/query.public.10K.u8bin"
NEW_GT["sift-100M"]="/home/yfei/data/ANNS/sift-100M/sift100m_groundtruth_K${K}_exact.diskann.bin"
OLD_GT["sift-100M"]=""   # not yet in run_disk_experiments.sh

TYPE["SPACEV"]="int8"
BASE_FILE["SPACEV"]="/home/yfei/data/ANNS/SPACEV/vectors_100m.bin"
QUERY_FILE["SPACEV"]="/home/yfei/data/ANNS/SPACEV/query.bin"
NEW_GT["SPACEV"]="/home/yfei/data/ANNS/SPACEV/spacev_groundtruth_K${K}_exact.diskann.bin"
OLD_GT["SPACEV"]="/home/yfei/data/ANNS/SPACEV/spacev100m-sliced-30K_groundtruth_K10.bin"

# Ordered list so runs happen in a deterministic order
ALL_DATASETS=("arxiv" "sift" "sift-100M" "SPACEV")

# If the user specified datasets on the command line, use those; otherwise all
if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
else
    TARGETS=("${ALL_DATASETS[@]}")
fi

# ---------------------------------------------------------------------------
compute_one() {
    local ds="$1"
    local dtype="${TYPE[$ds]}"
    local base="${BASE_FILE[$ds]}"
    local query="${QUERY_FILE[$ds]}"
    local gt="${NEW_GT[$ds]}"

    echo ""
    echo "================================================================"
    echo "Dataset : $ds"
    echo "Type    : $dtype   K=$K"
    echo "Base    : $base"
    echo "Query   : $query"
    echo "Output  : $gt"
    echo "================================================================"

    if [[ ! -f "$base" ]]; then
        echo "SKIP: base file not found: $base" >&2
        return 1
    fi
    if [[ ! -f "$query" ]]; then
        echo "SKIP: query file not found: $query" >&2
        return 1
    fi

    # Basic sanity check for DiskANN .bin header: [npts(int32), dim(int32)]
    if ! python3 - "$query" <<'PY'
import struct, sys
path = sys.argv[1]
with open(path, 'rb') as f:
    h = f.read(8)
if len(h) != 8:
    print(f"SKIP: invalid query header (file too small): {path}", file=sys.stderr)
    sys.exit(1)
n, d = struct.unpack('<ii', h)
if n <= 0 or d <= 0:
    print(f"SKIP: invalid query header n={n}, d={d}: {path}", file=sys.stderr)
    sys.exit(1)
PY
    then
        return 1
    fi

    if "$GT_BIN" "$dtype" "$base" "$query" "$K" "$gt"; then
        echo "Saved: $gt"
        return 0
    fi

    echo "FAIL: compute_groundtruth crashed or returned non-zero for dataset '$ds'." >&2
    return 1
}

# ---------------------------------------------------------------------------
# Update gt_file path in run_disk_experiments.sh for a dataset.
# Uses Python for reliable in-place JSON string replacement inside the shell var.
update_script_gt() {
    local ds="$1"
    local new_path="${NEW_GT[$ds]}"
    local old_path="${OLD_GT[$ds]}"

    if [[ -z "$old_path" ]]; then
        echo "NOTE: '$ds' has no existing entry in run_disk_experiments.sh — add it manually."
        return
    fi

    if grep -qF "\"gt_file\": \"$new_path\"" "$EXPERIMENT_SCRIPT"; then
        echo "Config already points to new GT for $ds."
        return
    fi

    sed -i "s|\"gt_file\": \"${old_path}\"|\"gt_file\": \"${new_path}\"|" "$EXPERIMENT_SCRIPT"
    echo "Updated run_disk_experiments.sh: gt_file for '$ds' -> $new_path"
}

# ---------------------------------------------------------------------------
FAILED=()
for ds in "${TARGETS[@]}"; do
    if [[ -z "${TYPE[$ds]+x}" ]]; then
        echo "Unknown dataset '$ds'. Known: ${ALL_DATASETS[*]}" >&2
        continue
    fi
    if compute_one "$ds"; then
        update_script_gt "$ds"
    else
        FAILED+=("$ds")
    fi
done

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "The following datasets were skipped due to missing files: ${FAILED[*]}"
fi
echo "Done."
