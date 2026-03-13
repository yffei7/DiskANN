#!/usr/bin/env bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
#
# Example script: build and search a disk-based DiskANN index for the
# arXiv float32 dataset (10,000 points, dimension 768).
#
# Usage:
#   bash scripts/run_disk_search_arxiv.sh <data_file.bin> <query_file.bin> \
#       <index_prefix> <result_prefix> [groundtruth.bin]
#
# Parameters tuned for arXiv float32 (small dataset, single-threaded search):
#   R=64, L=100, B=1 (1 GB in-memory budget), M=8 (8 GB build RAM), T=8
#   num_nodes_to_cache=5000, num_threads=1, beamwidth=4
#   L_search values: 50 75 100

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

# --------------------------------------------------------------------------
# Build index
# --------------------------------------------------------------------------
# Arguments: <data_type> <data_file> <index_prefix> <R> <L> <B> <M> <T>
#            <similarity_metric> <single_file_index(0/1)>
#   R=64    : graph degree
#   L=100   : build search list size
#   B=1     : 1 GB in-memory budget at search time
#   M=8     : 8 GB RAM limit for building
#   T=8     : number of build threads
echo "=== Building disk index for arXiv float32 ==="
"$BUILD_BIN" float "$DATA_FILE" "$INDEX_PREFIX" 64 100 1 8 8 l2 0

# --------------------------------------------------------------------------
# Search index (single-threaded)
# --------------------------------------------------------------------------
# Arguments: <index_type> <index_prefix> <single_file_index(0/1)> <tags(0/1)>
#            <num_nodes_to_cache> <num_threads> <beamwidth>
#            <query_file> <truthset (or "null")> <K>
#            <result_prefix> <similarity_metric>
#            <L1> [L2] ...
#   num_nodes_to_cache=5000  : cache top BFS nodes in RAM
#   num_threads=1            : single-threaded streaming search
#   beamwidth=4              : IO requests per search iteration
#   K=10                     : retrieve top-10 nearest neighbors
echo "=== Searching disk index for arXiv float32 (single-threaded) ==="
"$SEARCH_BIN" float "$INDEX_PREFIX" 0 0 5000 1 4 \
    "$QUERY_FILE" "$GT_FILE" 10 "$RESULT_PREFIX" l2 50 75 100
