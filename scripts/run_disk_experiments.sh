#!/usr/bin/env bash
# Unified DiskANN disk-index experiment runner.
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
# Cache sizes to sweep
# ---------------------------------------------------------------------------
CACHE_SIZES=(10000 50000 100000 500000 100000000)

# ---------------------------------------------------------------------------
# Dataset configs
# Each dataset is one entry in the DATASETS array; fields are set via
# parallel arrays below.  Add/remove datasets by editing these arrays.
# ---------------------------------------------------------------------------
DATASETS=(arxiv spacev turing)

# index data type (float / int8 / uint8)
declare -A TYPE=( [arxiv]=float   [spacev]=int8   [turing]=float  )

# similarity metric (l2 / cosine)
declare -A SIM=(  [arxiv]=l2      [spacev]=l2     [turing]=l2     )

# paths — adjust to your local file layout
declare -A INDEX_PREFIX=( [arxiv]=/data/arxiv/index  [spacev]=/data/spacev/index  [turing]=/data/turing/index  )
declare -A DATA_FILE=(    [arxiv]=/data/arxiv/data.bin  [spacev]=/data/spacev/data.bin  [turing]=/data/turing/data.bin  )
declare -A QUERY_FILE=(   [arxiv]=/data/arxiv/queries.bin  [spacev]=/data/spacev/queries.bin  [turing]=/data/turing/queries.bin  )
declare -A GT_FILE=(      [arxiv]=/data/arxiv/gt.bin  [spacev]=/data/spacev/gt.bin  [turing]=/data/turing/gt.bin  )
declare -A RESULT_DIR=(   [arxiv]=/results/arxiv  [spacev]=/results/spacev  [turing]=/results/turing  )

# search params
declare -A K=(        [arxiv]=10   [spacev]=10    [turing]=10    )
declare -A BEAMWIDTH=([arxiv]=4    [spacev]=4     [turing]=4     )
declare -A THREADS=(  [arxiv]=1    [spacev]=1     [turing]=1     )
declare -A L_VALUES=( [arxiv]="50 75 100"  [spacev]="100 150 200"  [turing]="100 150 200"  )

# build params (used only with --build)
declare -A BUILD_R=(  [arxiv]=64   [spacev]=64    [turing]=64    )
declare -A BUILD_L=(  [arxiv]=100  [spacev]=100   [turing]=100   )
declare -A BUILD_B=(  [arxiv]=1    [spacev]=20    [turing]=100   )
declare -A BUILD_M=(  [arxiv]=8    [spacev]=64    [turing]=200   )
declare -A BUILD_T=(  [arxiv]=8    [spacev]=64    [turing]=64    )

# ---------------------------------------------------------------------------
run_search() {
    local ds="$1" cache="$2"
    local result="${RESULT_DIR[$ds]}/cache${cache}"
    mkdir -p "$(dirname "$result")"
    echo ""
    echo ">>> dataset=$ds  num_nodes_to_cache=$cache <<<"
    "$SEARCH_BIN" "${TYPE[$ds]}" "${INDEX_PREFIX[$ds]}" 0 0 \
        "$cache" "${THREADS[$ds]}" "${BEAMWIDTH[$ds]}" \
        "${QUERY_FILE[$ds]}" "${GT_FILE[$ds]}" "${K[$ds]}" \
        "$result" "${SIM[$ds]}" \
        ${L_VALUES[$ds]}
    echo "----------------------------------------------------------------"
}

for DS in "${DATASETS[@]}"; do
    if (( BUILD_INDEX )); then
        echo "================================================================"
        echo "Building index: $DS  (type=${TYPE[$DS]})"
        echo "================================================================"
        "$BUILD_BIN" "${TYPE[$DS]}" "${DATA_FILE[$DS]}" "${INDEX_PREFIX[$DS]}" \
            "${BUILD_R[$DS]}" "${BUILD_L[$DS]}" \
            "${BUILD_B[$DS]}" "${BUILD_M[$DS]}" "${BUILD_T[$DS]}" \
            "${SIM[$DS]}" 0
    fi

    echo "================================================================"
    echo "Cache sweep: dataset=$DS  type=${TYPE[$DS]}  K=${K[$DS]}"
    echo "query=${QUERY_FILE[$DS]}  gt=${GT_FILE[$DS]}"
    echo "L values: ${L_VALUES[$DS]}"
    echo "================================================================"

    for CACHE in "${CACHE_SIZES[@]}"; do
        run_search "$DS" "$CACHE"
    done
done
