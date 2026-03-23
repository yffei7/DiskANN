#!/usr/bin/env bash
# Unified DiskANN disk-index experiment runner with JSON configuration.
#
# Loops over every (dataset, cache_size) combination and runs search_disk_index,
# printing the full metric block for each run so results can be compared easily.
#
# Usage: bash scripts/run_disk_experiments.sh [--build]
#   --build   also (re)build the disk index for each dataset before searching

set -euo pipefail

BUILD_INDEX=0
if [[ "${1:-}" == "--build" ]]; then
    BUILD_INDEX=1
fi

SEARCH_BIN="./build/tests/search_disk_index"
BUILD_BIN="./build/tests/build_disk_index"
[[ -f "$SEARCH_BIN" ]] || SEARCH_BIN="./tests/search_disk_index"
[[ -f "$BUILD_BIN"  ]] || BUILD_BIN="./tests/build_disk_index"

# ---------------------------------------------------------------------------
# JSON Configuration for datasets (loaded from external file)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASETS_CONFIG_FILE="${DATASETS_CONFIG_FILE:-$SCRIPT_DIR/datasets_config.json}"
if [[ ! -f "$DATASETS_CONFIG_FILE" ]]; then
    echo "Error: datasets config not found: $DATASETS_CONFIG_FILE" >&2
    exit 1
fi
DATASETS_CONFIG=$(cat "$DATASETS_CONFIG_FILE")

# Allow caller to override via env: DATASETS="spacev sift" ./run_disk_experiments.sh
if [[ -z "${DATASETS:-}" ]]; then
    # DATASETS=("spacev" "ms_turing" "sift" "sift-100M" "gist")
    # DATASETS=($(echo "$DATASETS_CONFIG" | jq -r 'keys[]'))
    # DATASETS=("arxiv" "spacev" "ms_turing" "sift" "sift-100M")
    DATASETS=("sift-100M")
else
    IFS=' ' read -ra DATASETS <<< "$DATASETS"
fi

# Validate dataset names
for DS in "${DATASETS[@]}"; do
    if ! echo "$DATASETS_CONFIG" | jq -e ".\"$DS\"" >/dev/null; then
        echo "Error: dataset '$DS' not found in DATASETS_CONFIG" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Helper function to get config value for a dataset
# ---------------------------------------------------------------------------
get_config() {
    local dataset="$1"
    local field="$2"
    echo "$DATASETS_CONFIG" | jq -r ".\"$dataset\".\"$field\""
}

get_build_config() {
    local dataset="$1"
    local field="$2"
    echo "$DATASETS_CONFIG" | jq -r ".\"$dataset\".build.\"$field\""
}

get_cache_sizes() {
    local dataset="$1"
    local budgets
    budgets=$(echo "$DATASETS_CONFIG" | jq -r ".\"$dataset\".cache_budgets_mb // empty | .[]" 2>/dev/null)
    if [[ -n "$budgets" ]]; then
        while IFS= read -r mb; do
            [[ -n "$mb" ]] && mb_to_cache_nodes "$mb" "$dataset"
        done <<< "$budgets"
    else
        echo "$DATASETS_CONFIG" | jq -r ".\"$dataset\".cache_sizes[]"
    fi
}

get_beamwidths() {
    local dataset="$1"
    echo "$DATASETS_CONFIG" | jq -r ".\"$dataset\".beamwidth | if type == \"array\" then .[] else . end"
}

get_l_values() {
    local dataset="$1"
    echo "$DATASETS_CONFIG" | jq -r ".\"$dataset\".l_values | if type == \"array\" then .[] else . end"
}

# ---------------------------------------------------------------------------
# Read a DiskANN binary file header (.bin / .fbin).
# Outputs: "num_points dim"  (both int32, little-endian at bytes 0-7)
# ---------------------------------------------------------------------------
read_bin_header() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo "Error: data file not found: $file" >&2
        return 1
    fi
    local num_points dim
    num_points=$(od -An -td4 -N4     "$file" | tr -d ' \n')
    dim=$(        od -An -td4 -j4 -N4 "$file" | tr -d ' \n')
    echo "$num_points $dim"
}

# Return sizeof(T) for a DiskANN data type string.
get_type_size() {
    case "$1" in
        float) echo 4 ;;
        uint8) echo 1 ;;
        int8)  echo 1 ;;
        *) echo "Error: unknown type '$1'" >&2; return 1 ;;
    esac
}

# Convert a memory budget (MB) to num_nodes_to_cache.
#
# DiskANN RAM cost per cached node (verified from source):
#   bytes_per_node = (R + 1) * 4          -- neighbor list (unsigned[])
#                  + aligned_dim * sizeof(T)  -- coordinate vector
#   aligned_dim = ceil(data_dim / 8) * 8
#   R = max_degree == build param R (after final pruning all node degrees <= R)
#
# num_nodes = floor(budget_bytes / bytes_per_node)
mb_to_cache_nodes() {
    local budget_mb="$1"
    local dataset="$2"

    local type data_file R
    type=$(get_config      "$dataset" "type")
    data_file=$(get_config "$dataset" "data_file")
    R=$(get_build_config   "$dataset" "R")

    local type_size
    type_size=$(get_type_size "$type")

    local header dim
    header=$(read_bin_header "$data_file")
    dim=$(echo "$header" | awk '{print $2}')

    local aligned_dim
    aligned_dim=$(( (dim + 7) / 8 * 8 ))

    local bytes_per_node budget_bytes
    bytes_per_node=$(( (R + 1) * 4 + aligned_dim * type_size ))
    budget_bytes=$(( budget_mb * 1024 * 1024 ))

    echo $(( budget_bytes / bytes_per_node ))
}

# ---------------------------------------------------------------------------
run_search() {
    local ds="$1" cache="$2" bw="$3" l_value="$4"
    local result_dir
    result_dir=$(get_config "$ds" "result_dir")
    local result="$result_dir/cache${cache}_bw${bw}_l${l_value}"
    mkdir -p "$(dirname "$result")"
    echo ""
    echo ">>> dataset=$ds  num_nodes_to_cache=$cache  beamwidth=$bw  l_value=$l_value <<<"

    if ! "$SEARCH_BIN" \
        "$(get_config "$ds" "type")" \
        "$(get_config "$ds" "index_prefix")" \
        "$(get_config "$ds" "single_file_index")" \
        "$(get_config "$ds" "tags")" \
        "$cache" \
        "$(get_config "$ds" "threads")" \
        "$bw" \
        "$(get_config "$ds" "query_file")" \
        "$(get_config "$ds" "gt_file")" \
        "$(get_config "$ds" "k")" \
        "$result" \
        "$(get_config "$ds" "similarity")" \
        "$l_value"; then
        echo "WARNING: search failed for '$ds' cache=$cache bw=$bw l_value=$l_value" >&2
    fi
    echo "----------------------------------------------------------------"
}

# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------
for DS in "${DATASETS[@]}"; do
    if (( BUILD_INDEX )); then
        echo "================================================================"
        echo "Building index: $DS  (type=$(get_config "$DS" "type"))"
        echo "================================================================"
        if ! "$BUILD_BIN" \
            "$(get_config "$DS" "type")" \
            "$(get_config "$DS" "data_file")" \
            "$(get_config "$DS" "index_prefix")" \
            "$(get_build_config "$DS" "R")" \
            "$(get_build_config "$DS" "L")" \
            "$(get_build_config "$DS" "B")" \
            "$(get_build_config "$DS" "M")" \
            "$(get_build_config "$DS" "T")" \
            "$(get_config "$DS" "similarity")" \
            "$(get_config "$DS" "tags")"; then
            echo "ERROR: index build failed for '$DS', skipping." >&2
            continue
        fi
    fi

    echo "================================================================"
    echo "Cache sweep: dataset=$DS  type=$(get_config "$DS" "type")  K=$(get_config "$DS" "k")"
    echo "query=$(get_config "$DS" "query_file")  gt=$(get_config "$DS" "gt_file")"
    echo "L values: $(get_config "$DS" "l_values")"
    echo "================================================================"

    # Sweep cache budgets (MB→nodes) × beamwidths × l_values
    while IFS= read -r CACHE; do
        [[ -n "$CACHE" ]] || continue
        while IFS= read -r BW; do
            [[ -n "$BW" ]] || continue
            while IFS= read -r LVAL; do
                [[ -n "$LVAL" ]] || continue
                run_search "$DS" "$CACHE" "$BW" "$LVAL"
            done < <(get_l_values "$DS")
        done < <(get_beamwidths "$DS")
    done < <(get_cache_sizes "$DS")
done