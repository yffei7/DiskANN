#!/usr/bin/env bash
# Run search_disk_index with varying num_nodes_to_cache values to measure
# the effect of the in-memory cache on disk I/O and query latency.
#
# Usage:
#   bash scripts/run_cache_sweep.sh <index_type> <index_prefix> \
#       <query_file.bin> <result_prefix> \
#       [truthset.bin] [K] [beamwidth] [num_threads] [similarity] [L1] [L2] ...
#
# Example (float, L2, single-threaded, K=10, beamwidth=4, L=100 200 500):
#   bash scripts/run_cache_sweep.sh float my_index my_queries.bin results \
#       my_gt.bin 10 4 1 l2 100 200 500

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <index_type> <index_prefix> <query_file.bin> <result_prefix>" \
         "[truthset.bin] [K] [beamwidth] [num_threads] [similarity] [L1] [L2] ..."
    exit 1
fi

INDEX_TYPE="$1"
INDEX_PREFIX="$2"
QUERY_FILE="$3"
RESULT_PREFIX="$4"
GT_FILE="${5:-null}"
K="${6:-10}"
BEAMWIDTH="${7:-4}"
NUM_THREADS="${8:-1}"
SIMILARITY="${9:-l2}"
shift 9 || shift "$#"
L_VALUES="${*:-100}"

SEARCH_BIN="./build/tests/search_disk_index"
if [ ! -f "$SEARCH_BIN" ]; then
    SEARCH_BIN="./tests/search_disk_index"
fi

CACHE_SIZES=(10000 50000 100000 500000 100000000)

echo "================================================================"
echo "Cache sweep: index=$INDEX_PREFIX  type=$INDEX_TYPE  K=$K"
echo "query=$QUERY_FILE  gt=$GT_FILE  beamwidth=$BEAMWIDTH  threads=$NUM_THREADS"
echo "L values: $L_VALUES"
echo "================================================================"

for CACHE in "${CACHE_SIZES[@]}"; do
    echo ""
    echo ">>> num_nodes_to_cache=$CACHE <<<"
    "$SEARCH_BIN" "$INDEX_TYPE" "$INDEX_PREFIX" 0 0 \
        "$CACHE" "$NUM_THREADS" "$BEAMWIDTH" \
        "$QUERY_FILE" "$GT_FILE" "$K" \
        "${RESULT_PREFIX}_cache${CACHE}" "$SIMILARITY" \
        $L_VALUES
    echo "----------------------------------------------------------------"
done
