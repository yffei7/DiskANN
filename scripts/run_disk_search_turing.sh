#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
#
# Example script: build and search a disk-based DiskANN index for the
# Microsoft Turing-ANNS dataset (100M points, dimension 100, float32).
#
# Usage:
#   bash scripts/run_disk_search_turing.sh <data_file.bin> <query_file.bin> \
#       <index_prefix> <result_prefix> [groundtruth.bin]
#
# Parameters tuned for Turing-ANNS 100M (single-threaded search):
#   R=64, L=100, B=100 (100 GB in-memory budget), M=200 (200 GB build RAM), T=64
#   num_nodes_to_cache=500000, num_threads=1, beamwidth=4
#   L_search values: 100 150 200
#
# NOTE: B and M depend on the RAM available on your machine.
#   - B: the portion of the index kept in RAM during search (GB). Larger B
#        improves recall/speed. Set to at most (available RAM - OS overhead).
#   - M: RAM budget for building the index (GB). If M is less than what is
#        required to build in one pass, a divide-and-conquer approach is used
#        (up to ~1.5x slower). Provide as much as your machine allows.

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <data_file.bin> <query_file.bin> <index_prefix> <result_prefix> [groundtruth.bin]"
    exit 1
fi

DATA_FILE="$1"
QUERY_FILE="$2"
INDEX_PREFIX="$3"
RESULT_PREFIX="$4"
GT_FILE="${5:-null}"

# Paths to binaries (adjust if built in a different directory)
BUILD_BIN="./build/tests/build_disk_index"
SEARCH_BIN="./build/tests/search_disk_index"

if [ ! -f "$BUILD_BIN" ]; then
    BUILD_BIN="./tests/build_disk_index"
fi
if [ ! -f "$SEARCH_BIN" ]; then
    SEARCH_BIN="./tests/search_disk_index"
fi

# Memory parameters — adjust to your machine's available RAM
BUILD_RAM_GB=200   # M: RAM budget for building (GB)
SEARCH_RAM_GB=100  # B: in-memory budget for search index (GB)
BUILD_THREADS=64   # T: build threads

# --------------------------------------------------------------------------
# Build index
# --------------------------------------------------------------------------
# Arguments: <data_type> <data_file> <index_prefix> <R> <L> <B> <M> <T>
#            <similarity_metric> <single_file_index(0/1)>
echo "=== Building disk index for Turing-ANNS 100M ==="
"$BUILD_BIN" float "$DATA_FILE" "$INDEX_PREFIX" \
    64 100 "$SEARCH_RAM_GB" "$BUILD_RAM_GB" "$BUILD_THREADS" l2 0

# --------------------------------------------------------------------------
# Search index (single-threaded)
# --------------------------------------------------------------------------
# Arguments: <index_type> <index_prefix> <single_file_index(0/1)> <tags(0/1)>
#            <num_nodes_to_cache> <num_threads> <beamwidth>
#            <query_file> <truthset (or "null")> <K>
#            <result_prefix> <similarity_metric>
#            <L1> [L2] ...
#   num_nodes_to_cache=500000 : cache top BFS nodes in RAM (~a few GB)
#   num_threads=1             : single-threaded streaming search
#   beamwidth=4               : IO requests per search iteration
#   K=10                      : retrieve top-10 nearest neighbors
echo "=== Searching disk index for Turing-ANNS 100M (single-threaded) ==="
"$SEARCH_BIN" float "$INDEX_PREFIX" 0 0 500000 1 4 \
    "$QUERY_FILE" "$GT_FILE" 10 "$RESULT_PREFIX" l2 100 150 200
