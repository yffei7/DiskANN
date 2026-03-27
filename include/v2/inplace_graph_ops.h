// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <string>
#include <vector>
#include "distance.h"
#include "neighbor.h"
#include "v2/inplace_backend.h"
#include "tsl/robin_set.h"

namespace diskann {
namespace inplace {

// Two-tier BFS search:
//   - frontier neighbors scored with PQ distances
//   - expanded nodes scored with full-precision coords (aligned scratch copy)
template<typename T>
std::pair<uint32_t, uint32_t>
graph_iterate_to_fixed_point(
    const T* query, unsigned L,
    const std::vector<unsigned>& init_ids,
    unsigned beamwidth,
    InPlaceGraphStore* store,
    unsigned aligned_dim,
    Distance<T>* dist_cmp,
    InPlaceSearchScratch* scratch,
    std::vector<Neighbor>& best_L_nodes,
    tsl::robin_set<unsigned>* visited_ids = nullptr,
    DistanceScope scope = DistanceScope::QUERY,
    unsigned pq_confirm_topk = 0,
    const std::string& pq_confirm_mode = "fixed",
    double pq_confirm_margin = 0.05,
    unsigned pq_confirm_delta = 4);

// PQ-based alpha-occlusion
template<typename T>
void graph_occlude_list_pq(
    std::vector<Neighbor>& pool,
    float alpha, unsigned degree, unsigned maxc,
    std::vector<Neighbor>& result,
    std::vector<float>& occlude_factor,
    InPlaceGraphStore* store,
    InPlaceSearchScratch* scratch,
    DistanceScope scope);

// Full-precision occlusion (for drain when coords are in cache)
template<typename T>
void graph_occlude_list(
    std::vector<Neighbor>& pool,
    float alpha, unsigned degree, unsigned maxc,
    std::vector<Neighbor>& result,
    std::vector<float>& occlude_factor,
    InPlaceGraphStore* store,
    InPlaceSearchScratch* scratch,
    unsigned aligned_dim,
    Distance<T>* distance,
    DistanceScope scope);

// Prune: uses PQ occlusion when available, full-precision fallback otherwise
template<typename T>
void graph_prune_neighbors_pq(
    unsigned location,
    std::vector<Neighbor>& pool,
    unsigned R, unsigned C, float alpha,
    std::vector<unsigned>& pruned_list,
    InPlaceGraphStore* store,
    InPlaceSearchScratch* scratch,
    unsigned aligned_dim = 0,
    Distance<T>* distance = nullptr,
    DistanceScope scope = DistanceScope::UPDATE);

// Defer-only reverse edge insert
void graph_inter_insert_deferred(
    unsigned new_node,
    std::vector<unsigned>& pruned_list,
    unsigned prune_R,
    InPlaceGraphStore* store);

}  // namespace inplace
}  // namespace diskann
