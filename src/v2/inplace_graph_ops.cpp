// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "v2/inplace_graph_ops.h"
#include "v2/inplace_backend.h"
#include "logger.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <limits>
#include <vector>
#include <immintrin.h>

namespace diskann {
namespace inplace {

// ===========================================================================
// PQ helpers (local copies from pq_flash_index.cpp anonymous namespace)
// ===========================================================================
static void aggregate_coords(const unsigned* ids, const _u64 n_ids,
                             const _u8* all_coords, const _u64 ndims,
                             _u8* out) {
    for (_u64 i = 0; i < n_ids; i++) {
        memcpy(out + i * ndims, all_coords + ids[i] * ndims,
               ndims * sizeof(_u8));
    }
}

static void pq_dist_lookup(const _u8* pq_ids, const _u64 n_pts,
                            const _u64 pq_nchunks, const float* pq_dists,
                            float* dists_out) {
    _mm_prefetch((char*)dists_out, _MM_HINT_T0);
    _mm_prefetch((char*)pq_ids, _MM_HINT_T0);
    memset(dists_out, 0, n_pts * sizeof(float));
    for (_u64 chunk = 0; chunk < pq_nchunks; chunk++) {
        const float* chunk_dists = pq_dists + 256 * chunk;
        if (chunk < pq_nchunks - 1) {
            _mm_prefetch((char*)(chunk_dists + 256), _MM_HINT_T0);
        }
        for (_u64 idx = 0; idx < n_pts; idx++) {
            _u8 pq_centerid = pq_ids[pq_nchunks * idx + chunk];
            dists_out[idx] += chunk_dists[pq_centerid];
        }
    }
}

static inline uint64_t elapsed_ns(
    const std::chrono::steady_clock::time_point& start) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - start)
        .count();
}

static uint32_t effective_confirm_topk(const std::string& mode,
                                       uint32_t configured_topk,
                                       uint32_t beamwidth,
                                       const std::vector<Neighbor>& pool,
                                       double margin,
                                       uint32_t delta) {
    uint32_t safe_floor = std::max<uint32_t>(1, beamwidth);
    uint32_t base = std::max<uint32_t>(safe_floor, configured_topk);
    if (mode != "adaptive" || pool.empty()) return base;
    uint32_t boundary = std::min<uint32_t>(base - 1, static_cast<uint32_t>(pool.size() - 1));
    uint32_t probe = std::min<uint32_t>(boundary + std::max<uint32_t>(1, delta),
                                        static_cast<uint32_t>(pool.size() - 1));
    if (probe <= boundary) return base;
    double gap = static_cast<double>(pool[probe].distance) - static_cast<double>(pool[boundary].distance);
    return gap >= margin ? safe_floor : base;
}

// ===========================================================================
// graph_iterate_to_fixed_point -- PQ two-tier search
// ===========================================================================
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
    tsl::robin_set<unsigned>* visited_ids,
    DistanceScope scope,
    unsigned pq_confirm_topk,
    const std::string& pq_confirm_mode,
    double pq_confirm_margin,
    unsigned pq_confirm_delta) {

    best_L_nodes.resize(L + 1);
    for (unsigned i = 0; i < L + 1; i++) {
        best_L_nodes[i].distance = std::numeric_limits<float>::max();
    }

    tsl::robin_set<unsigned> inserted_into_pool;
    inserted_into_pool.reserve(L * 20);

    // Pre-compute PQ chunk distances for this query
    uint32_t n_chunks = store->n_chunks();
    if (n_chunks > 0 && scratch->pq_dists) {
        auto start = std::chrono::steady_clock::now();
        store->pq_table_mut().populate_chunk_distances(
            (const float*)query, scratch->pq_dists);
        scratch->record_distance(scope, elapsed_ns(start), 0);
    }

    unsigned l = 0;
    for (auto id : init_ids) {
        if (inserted_into_pool.find(id) != inserted_into_pool.end()) continue;
        if (!store->is_active(id)) continue;

        // Full-precision distance for seed nodes
        auto view = store->pin_node(id, READ);
        if (view._page_id == INVALID_PAGE) continue;

        // Copy to aligned scratch for SIMD safety
        char* aligned = scratch->aligned_coord_scratch +
                        (scratch->coord_idx % InPlaceSearchScratch::MAX_SCRATCH_NODES) *
                        aligned_dim * sizeof(T);
        memcpy(aligned, view.coords, (size_t)aligned_dim * sizeof(T));
        scratch->coord_idx++;
        store->unpin_node(view);

        auto start = std::chrono::steady_clock::now();
        float dist = dist_cmp->compare(query, (const T*)aligned, aligned_dim);
        scratch->record_distance(scope, elapsed_ns(start), 1);
        inserted_into_pool.insert(id);
        best_L_nodes[l++] = Neighbor(id, dist, true);
        if (l == L) break;
    }

    std::sort(best_L_nodes.begin(), best_L_nodes.begin() + l);
    unsigned k = 0;
    uint32_t hops = 0;
    uint32_t cmps = 0;
    std::vector<unsigned> frontier_ids;
    frontier_ids.reserve(std::max<unsigned>(1, beamwidth));
    std::vector<unsigned> nbr_ids;
    nbr_ids.reserve(store->max_degree());

    while (k < l) {
        unsigned nk = l;
        frontier_ids.clear();
        unsigned beam = std::max<unsigned>(1, beamwidth);
        unsigned cursor = k;
        while (cursor < l && frontier_ids.size() < beam) {
            if (best_L_nodes[cursor].flag) {
                best_L_nodes[cursor].flag = false;
                frontier_ids.push_back(best_L_nodes[cursor].id);
            }
            cursor++;
        }
        if (frontier_ids.empty()) {
            ++k;
            continue;
        }

        for (auto n : frontier_ids) {
            hops++;
            if (visited_ids) visited_ids->insert(n);

            auto view = store->pin_node(n, READ);
            if (view._page_id == INVALID_PAGE) continue;
            if (view.flags & FLAG_DELETED) {
                store->unpin_node(view);
                continue;
            }

            nbr_ids.clear();
            nbr_ids.reserve(std::max<size_t>(nbr_ids.capacity(), view.degree));
            for (uint16_t m = 0; m < view.degree; m++) {
                unsigned nbr = view.neighbors[m];
                if (inserted_into_pool.find(nbr) == inserted_into_pool.end()) {
                    nbr_ids.push_back(nbr);
                }
            }
            store->unpin_node(view);

            if (nbr_ids.empty()) continue;

            if (n_chunks > 0 && scratch->pq_dists && !nbr_ids.empty()) {
                uint32_t batch = (uint32_t)nbr_ids.size();
                uint8_t* pq_scratch = scratch->pq_coord_scratch;
                const uint8_t* pq_data = store->pq_data();
                for (uint32_t m = 0; m < batch; ++m) {
                    _mm_prefetch((const char*)(pq_data + (uint64_t)nbr_ids[m] * n_chunks),
                                 _MM_HINT_T0);
                }
                auto start = std::chrono::steady_clock::now();
                aggregate_coords(nbr_ids.data(), batch,
                                 store->pq_data(), n_chunks, pq_scratch);
                pq_dist_lookup(pq_scratch, batch, n_chunks,
                               scratch->pq_dists, scratch->dist_scratch);
                scratch->record_distance(scope, elapsed_ns(start), batch);

                for (uint32_t m = 0; m < batch; m++) {
                    unsigned id = nbr_ids[m];
                    inserted_into_pool.insert(id);
                    cmps++;
                    float dist = scratch->dist_scratch[m];
                    if (dist >= best_L_nodes[l - 1].distance && l == L) continue;
                    Neighbor nn(id, dist, true);
                    unsigned r = InsertIntoPool(best_L_nodes.data(), l, nn);
                    if (l < L) ++l;
                    if (r < nk) nk = r;
                }
            } else {
                std::vector<uint8_t> found;
                std::vector<uint8_t> flags;
                char* batch_coords = scratch->aligned_coord_scratch +
                    (scratch->coord_idx % InPlaceSearchScratch::MAX_SCRATCH_NODES) *
                    aligned_dim * sizeof(T);
                store->batch_fetch_coords(nbr_ids, batch_coords, found, &flags, false);
                uint64_t local_cmps = 0;
                auto start = std::chrono::steady_clock::now();
                for (size_t m = 0; m < nbr_ids.size(); ++m) {
                    unsigned id = nbr_ids[m];
                    inserted_into_pool.insert(id);
                    if (!found[m] || (flags[m] & FLAG_DELETED)) continue;
                    const char* aligned = batch_coords + m * aligned_dim * sizeof(T);
                    scratch->coord_idx++;
                    cmps++;
                    local_cmps++;
                    float dist = dist_cmp->compare(query, reinterpret_cast<const T*>(aligned), aligned_dim);
                    if (dist >= best_L_nodes[l - 1].distance && l == L) continue;
                    Neighbor nn(id, dist, true);
                    unsigned r = InsertIntoPool(best_L_nodes.data(), l, nn);
                    if (l < L) ++l;
                    if (r < nk) nk = r;
                }
                scratch->record_distance(scope, elapsed_ns(start), local_cmps);
            }
        }

        if (pq_confirm_topk > 0 && l > 0) {
            uint32_t confirm_count = std::min<uint32_t>(
                effective_confirm_topk(pq_confirm_mode, pq_confirm_topk, beamwidth,
                                       best_L_nodes, pq_confirm_margin, pq_confirm_delta),
                l);
            std::vector<uint32_t> confirm_ids;
            std::vector<uint32_t> confirm_slots;
            confirm_ids.reserve(confirm_count);
            confirm_slots.reserve(confirm_count);
            for (uint32_t m = 0; m < confirm_count; ++m) {
                confirm_ids.push_back(best_L_nodes[m].id);
                confirm_slots.push_back(m);
            }

            std::vector<uint8_t> resident_found(confirm_ids.size(), 0);
            uint64_t local_cmps = 0;
            auto confirm_start = std::chrono::steady_clock::now();
            for (size_t idx = 0; idx < confirm_ids.size(); ++idx) {
                auto nview = store->try_pin_node_if_resident(confirm_ids[idx], READ);
                if (nview._page_id == INVALID_PAGE || (nview.flags & FLAG_DELETED)) {
                    if (nview._page_id != INVALID_PAGE) store->unpin_node(nview);
                    continue;
                }
                char* aligned = scratch->aligned_coord_scratch +
                    (scratch->coord_idx % InPlaceSearchScratch::MAX_SCRATCH_NODES) *
                    aligned_dim * sizeof(T);
                memcpy(aligned, nview.coords, (size_t)aligned_dim * sizeof(T));
                scratch->coord_idx++;
                store->unpin_node(nview);
                best_L_nodes[confirm_slots[idx]].distance =
                    dist_cmp->compare(query, reinterpret_cast<const T*>(aligned), aligned_dim);
                resident_found[idx] = 1;
                local_cmps++;
            }

            std::vector<uint32_t> miss_ids;
            std::vector<uint32_t> miss_slots;
            miss_ids.reserve(confirm_ids.size());
            miss_slots.reserve(confirm_ids.size());
            for (size_t idx = 0; idx < confirm_ids.size(); ++idx) {
                if (!resident_found[idx]) {
                    miss_ids.push_back(confirm_ids[idx]);
                    miss_slots.push_back(confirm_slots[idx]);
                }
            }
            if (!miss_ids.empty()) {
                std::vector<uint8_t> found;
                std::vector<uint8_t> flags;
                char* batch_coords = scratch->aligned_coord_scratch +
                    (scratch->coord_idx % InPlaceSearchScratch::MAX_SCRATCH_NODES) *
                    aligned_dim * sizeof(T);
                store->batch_fetch_coords(miss_ids, batch_coords, found, &flags, false);
                for (size_t idx = 0; idx < miss_ids.size(); ++idx) {
                    if (!found[idx] || (flags[idx] & FLAG_DELETED)) continue;
                    const char* aligned = batch_coords + idx * aligned_dim * sizeof(T);
                    scratch->coord_idx++;
                    best_L_nodes[miss_slots[idx]].distance =
                        dist_cmp->compare(query, reinterpret_cast<const T*>(aligned), aligned_dim);
                    local_cmps++;
                }
            }
            scratch->record_distance(scope, elapsed_ns(confirm_start), local_cmps);
            std::sort(best_L_nodes.begin(), best_L_nodes.begin() + l);
        }

        if (nk <= k) k = nk;
        else ++k;
    }

    // Trim deleted nodes from results
    std::vector<Neighbor> clean;
    clean.reserve(l);
    for (unsigned i = 0; i < l; i++) {
        if (store->is_active(best_L_nodes[i].id)) {
            clean.push_back(best_L_nodes[i]);
        }
    }
    if (n_chunks > 0 && !clean.empty()) {
        std::vector<uint32_t> rerank_ids;
        rerank_ids.reserve(clean.size());
        for (const auto& cand : clean) rerank_ids.push_back(cand.id);
        std::vector<uint8_t> resident_found(rerank_ids.size(), 0);
        uint64_t local_cmps = 0;
        auto rerank_start = std::chrono::steady_clock::now();
        for (size_t idx = 0; idx < rerank_ids.size(); ++idx) {
            auto nview = store->try_pin_node_if_resident(rerank_ids[idx], READ);
            if (nview._page_id == INVALID_PAGE || (nview.flags & FLAG_DELETED)) {
                if (nview._page_id != INVALID_PAGE) store->unpin_node(nview);
                continue;
            }
            char* aligned = scratch->aligned_coord_scratch +
                (scratch->coord_idx % InPlaceSearchScratch::MAX_SCRATCH_NODES) *
                aligned_dim * sizeof(T);
            memcpy(aligned, nview.coords, (size_t)aligned_dim * sizeof(T));
            scratch->coord_idx++;
            store->unpin_node(nview);
            clean[idx].distance = dist_cmp->compare(query, reinterpret_cast<const T*>(aligned), aligned_dim);
            resident_found[idx] = 1;
            local_cmps++;
        }
        std::vector<uint32_t> miss_ids;
        std::vector<uint32_t> miss_slots;
        for (size_t idx = 0; idx < rerank_ids.size(); ++idx) {
            if (!resident_found[idx]) {
                miss_ids.push_back(rerank_ids[idx]);
                miss_slots.push_back(static_cast<uint32_t>(idx));
            }
        }
        if (!miss_ids.empty()) {
            std::vector<uint8_t> found;
            std::vector<uint8_t> flags;
            char* batch_coords = scratch->aligned_coord_scratch +
                (scratch->coord_idx % InPlaceSearchScratch::MAX_SCRATCH_NODES) *
                aligned_dim * sizeof(T);
            store->batch_fetch_coords(miss_ids, batch_coords, found, &flags, false);
            for (size_t idx = 0; idx < miss_ids.size(); ++idx) {
                if (!found[idx] || (flags[idx] & FLAG_DELETED)) continue;
                const char* aligned = batch_coords + idx * aligned_dim * sizeof(T);
                scratch->coord_idx++;
                clean[miss_slots[idx]].distance =
                    dist_cmp->compare(query, reinterpret_cast<const T*>(aligned), aligned_dim);
                local_cmps++;
            }
        }
        scratch->record_distance(scope, elapsed_ns(rerank_start), local_cmps);
        std::sort(clean.begin(), clean.end());
    }
    best_L_nodes.resize(L + 1);
    for (unsigned i = 0; i < L + 1; i++) {
        best_L_nodes[i].distance = std::numeric_limits<float>::max();
    }
    for (unsigned i = 0; i < std::min((unsigned)clean.size(), L); i++) {
        best_L_nodes[i] = clean[i];
    }

    return std::make_pair(hops, cmps);
}

// ===========================================================================
// graph_occlude_list_pq
// ===========================================================================
template<typename T>
void graph_occlude_list_pq(
    std::vector<Neighbor>& pool,
    float alpha, unsigned degree, unsigned maxc,
    std::vector<Neighbor>& result,
    std::vector<float>& occlude_factor,
    InPlaceGraphStore* store,
    InPlaceSearchScratch* scratch,
    DistanceScope scope) {
    if (pool.empty()) return;
    assert(std::is_sorted(pool.begin(), pool.end()));

    std::vector<unsigned> batch_ids;
    batch_ids.reserve(maxc);

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
        uint32_t start = 0;
        while (result.size() < degree && start < pool.size() && start < maxc) {
            auto& p = pool[start];
            if (occlude_factor[start] > cur_alpha) {
                start++;
                continue;
            }
            occlude_factor[start] = std::numeric_limits<float>::max();
            result.push_back(p);

            batch_ids.clear();
            for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
                if (occlude_factor[t] <= alpha) batch_ids.push_back(pool[t].id);
            }
            if (!batch_ids.empty()) {
                auto dist_start = std::chrono::steady_clock::now();
                store->compute_pq_dists_src(
                    p.id, batch_ids.data(), scratch->dist_scratch,
                    (uint32_t)batch_ids.size(), scratch->pq_coord_scratch);
                scratch->record_distance(scope, elapsed_ns(dist_start),
                                         (uint64_t)batch_ids.size());
                uint32_t batch_idx = 0;
                for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
                    if (occlude_factor[t] > alpha) continue;
                    float djk = scratch->dist_scratch[batch_idx++];
                    occlude_factor[t] =
                        std::max(occlude_factor[t], pool[t].distance / djk);
                }
            }
            start++;
        }
        cur_alpha *= 1.2f;
    }
}

// ===========================================================================
// graph_occlude_list (full-precision)
// ===========================================================================
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
    DistanceScope scope) {
    if (pool.empty()) return;
    assert(std::is_sorted(pool.begin(), pool.end()));

    float cur_alpha = 1;
    while (cur_alpha <= alpha && result.size() < degree) {
        uint32_t start = 0;
        while (result.size() < degree && start < pool.size() && start < maxc) {
            auto& p = pool[start];
            if (occlude_factor[start] > cur_alpha) {
                start++;
                continue;
            }
            occlude_factor[start] = std::numeric_limits<float>::max();
            result.push_back(p);

            // Pin p once for the entire inner loop
            auto vp = store->pin_node(p.id, READ);
            T* aligned_p = reinterpret_cast<T*>(scratch->aux_coord_scratch);
            T* aligned_t = reinterpret_cast<T*>(
                scratch->aux_coord_scratch + aligned_dim * sizeof(T));
            bool p_valid = (vp._page_id != INVALID_PAGE);
            if (p_valid) {
                memcpy(aligned_p, vp.coords, aligned_dim * sizeof(T));
            }

            uint64_t local_cmps = 0;
            auto dist_start = std::chrono::steady_clock::now();
            for (uint32_t t = start + 1; t < pool.size() && t < maxc; t++) {
                if (occlude_factor[t] > alpha) continue;
                if (!p_valid) continue;
                auto vt = store->pin_node(pool[t].id, READ);
                float djk = std::numeric_limits<float>::max();
                if (vt._page_id != INVALID_PAGE) {
                    memcpy(aligned_t, vt.coords, aligned_dim * sizeof(T));
                    djk = distance->compare(aligned_p, aligned_t, aligned_dim);
                    local_cmps++;
                    store->unpin_node(vt);
                }
                occlude_factor[t] =
                    std::max(occlude_factor[t], pool[t].distance / djk);
            }
            scratch->record_distance(scope, elapsed_ns(dist_start),
                                     local_cmps);
            if (p_valid) store->unpin_node(vp);
            start++;
        }
        cur_alpha *= 1.2f;
    }
}

// ===========================================================================
// graph_prune_neighbors_pq
// ===========================================================================
template<typename T>
void graph_prune_neighbors_pq(
    unsigned location,
    std::vector<Neighbor>& pool,
    unsigned R, unsigned C, float alpha,
    std::vector<unsigned>& pruned_list,
    InPlaceGraphStore* store,
    InPlaceSearchScratch* scratch,
    unsigned aligned_dim,
    Distance<T>* distance,
    DistanceScope scope) {
    if (pool.empty()) return;
    std::sort(pool.begin(), pool.end());

    std::vector<Neighbor> result;
    result.reserve(R);
    std::vector<float> occlude_factor(pool.size(), 0);

    if (store->n_chunks() > 0 && scratch && scratch->pq_coord_scratch) {
        graph_occlude_list_pq<T>(pool, alpha, R, C, result, occlude_factor,
                                 store, scratch, scope);
    } else if (distance && aligned_dim > 0) {
        graph_occlude_list<T>(pool, alpha, R, C, result, occlude_factor,
                              store, scratch, aligned_dim, distance, scope);
    } else {
        // No PQ and no distance function: just take top-R by distance
        for (auto& p : pool) {
            if (p.id != location) {
                result.push_back(p);
                if (result.size() >= R) break;
            }
        }
    }

    pruned_list.clear();
    for (auto& iter : result) {
        if (iter.id != location)
            pruned_list.push_back(iter.id);
    }
}

// ===========================================================================
// graph_inter_insert_deferred
// ===========================================================================
void graph_inter_insert_deferred(
    unsigned new_node,
    std::vector<unsigned>& pruned_list,
    InPlaceGraphStore* store) {
    if (!store->is_active(new_node)) return;
    for (auto des : pruned_list) {
        if (des == new_node) continue;
        if (!store->is_active(des)) continue;
        auto view = store->try_pin_node_if_resident(des, WRITE);
        bool deferred = true;
        if (view._page_id != INVALID_PAGE) {
            if (view.flags & FLAG_DELETED) {
                deferred = false;
            } else {
                bool exists = false;
                for (uint16_t j = 0; j < view.degree; ++j) {
                    if (view.neighbors[j] == new_node) {
                        exists = true;
                        break;
                    }
                }
                if (exists) {
                    deferred = false;
                } else if (view.degree < view._Mmax) {
                    view.neighbors[view.degree++] = new_node;
                    store->commit_node(view);
                    deferred = false;
                }
            }
            store->unpin_node(view);
        }
        if (deferred) {
            store->defer_reverse_edge(des, new_node);
        }
    }
}

// ===========================================================================
// drain_deferred_edges -- template method on InPlaceGraphStore
// Page-local batching: sort by page, group by page, pin page once,
// process all targets within that page before unpinning.
// ===========================================================================
template<typename T>
void InPlaceGraphStore::drain_deferred_edges(
    unsigned R, unsigned C, float alpha,
    unsigned aligned_dim, diskann::Distance<T>* dist) {

    auto edges = _deferred_edges.drain();
    if (edges.empty()) return;

    // Build per-target source lists
    tsl::robin_map<uint32_t, std::vector<uint32_t>> target_to_sources;
    target_to_sources.reserve(edges.size());
    for (auto& e : edges) {
        if (!is_active(e.target)) continue;
        if (!is_active(e.src)) continue;
        target_to_sources[e.target].push_back(e.src);
    }

    // Collect targets and sort by page_id for locality
    std::vector<uint32_t> targets;
    targets.reserve(target_to_sources.size());
    for (auto& kv : target_to_sources) {
        targets.push_back(kv.first);
    }
    std::sort(targets.begin(), targets.end(),
              [this](uint32_t a, uint32_t b) {
                  return get_page_id(a) < get_page_id(b);
              });

    InPlaceSearchScratch local_scratch;
    local_scratch.init(aligned_dim, _n_chunks, sizeof(T));
    std::vector<unsigned> current_nbrs;
    current_nbrs.reserve(_Mmax + R);
    tsl::robin_set<unsigned> existing;
    existing.reserve(_Mmax + R);
    std::vector<Neighbor> pool;
    pool.reserve(_Mmax + R);
    std::vector<unsigned> pruned;
    pruned.reserve(_Mmax);
    std::vector<uint8_t> pq_scratch_buf;
    std::vector<float> dist_buf;

    // Group by page and process
    size_t ti = 0;
    while (ti < targets.size()) {
        uint32_t cur_page = get_page_id(targets[ti]);
        // Collect all targets on this page
        size_t page_start = ti;
        while (ti < targets.size() && get_page_id(targets[ti]) == cur_page) {
            ti++;
        }
        if (cur_page == INVALID_PAGE) continue;

        // Pin the page once for the entire batch
        auto& frame = _bp.pin(cur_page, FrameRegion::UPDATE);
        bool page_modified = false;

        for (size_t idx = page_start; idx < ti; idx++) {
            uint32_t target = targets[idx];
            auto& sources = target_to_sources[target];

            if (!is_active(target)) continue;

            // Locate the slot directly within the already-pinned page
            RID rid;
            {
                std::lock_guard<std::mutex> lk(_alloc_mtx);
                if (target >= _node_to_rid.size()) continue;
                rid = _node_to_rid[target];
            }
            if (rid.page_id != cur_page) {
                // Shouldn't happen, but guard against it
                continue;
            }

            char* slot_ptr = frame.data + header_bytes() +
                             rid.slot_idx * _slot_size;
            PackedSlotHeader hdr;
            memcpy(&hdr, slot_ptr, sizeof(hdr));

            uint32_t* nbrs_ptr = (uint32_t*)(slot_ptr +
                sizeof(PackedSlotHeader) + _aligned_dim * _elem_size);

            current_nbrs.clear();
            current_nbrs.reserve(std::max<size_t>(current_nbrs.capacity(),
                                                  hdr.degree + sources.size()));
            for (uint16_t j = 0; j < hdr.degree; j++) {
                if (is_active(nbrs_ptr[j])) current_nbrs.push_back(nbrs_ptr[j]);
            }

            existing.clear();
            existing.insert(current_nbrs.begin(), current_nbrs.end());
            for (auto src : sources) {
                if (!is_active(src)) continue;
                if (existing.find(src) == existing.end() && src != target) {
                    current_nbrs.push_back(src);
                    existing.insert(src);
                }
            }

            if (current_nbrs.size() <= (size_t)R) {
                hdr.degree = (uint16_t)current_nbrs.size();
                for (uint16_t j = 0; j < hdr.degree; j++) {
                    nbrs_ptr[j] = current_nbrs[j];
                }
            } else {
                // Prune with distance ranking
                pool.clear();
                pool.reserve(std::max<size_t>(pool.capacity(),
                                              current_nbrs.size()));

                if (_n_chunks > 0 && _pq_codes) {
                    pq_scratch_buf.resize(current_nbrs.size() * _n_chunks + 64);
                    dist_buf.resize(current_nbrs.size());
                    auto dist_start = std::chrono::steady_clock::now();
                    compute_pq_dists_src(target, current_nbrs.data(),
                                         dist_buf.data(),
                                         (uint32_t)current_nbrs.size(),
                                         pq_scratch_buf.data());
                    _stats.record_distance(DistanceScope::UPDATE,
                                           elapsed_ns(dist_start),
                                           (uint64_t)current_nbrs.size());
                    for (size_t j = 0; j < current_nbrs.size(); j++) {
                        pool.emplace_back(current_nbrs[j], dist_buf[j], true);
                    }
                } else {
                    char* target_coords = slot_ptr + sizeof(PackedSlotHeader);
                    T* aligned_tgt = reinterpret_cast<T*>(local_scratch.aux_coord_scratch);
                    T* aligned_buf = reinterpret_cast<T*>(
                        local_scratch.aux_coord_scratch + aligned_dim * sizeof(T));
                    memcpy(aligned_tgt, target_coords, aligned_dim * sizeof(T));
                    uint64_t local_cmps = 0;
                    auto dist_start = std::chrono::steady_clock::now();
                    for (auto nbr : current_nbrs) {
                        auto nview = pin_node(nbr, READ);
                        float d = std::numeric_limits<float>::max();
                        if (nview._page_id != INVALID_PAGE) {
                            memcpy(aligned_buf, nview.coords,
                                   aligned_dim * sizeof(T));
                            d = dist->compare(aligned_tgt, aligned_buf,
                                              aligned_dim);
                            local_cmps++;
                            unpin_node(nview);
                        }
                        pool.emplace_back(nbr, d, true);
                    }
                    _stats.record_distance(DistanceScope::UPDATE,
                                           elapsed_ns(dist_start),
                                           local_cmps);
                }

                pruned.clear();
                graph_prune_neighbors_pq<T>(target, pool, R, C, alpha,
                                            pruned, this, &local_scratch,
                                            aligned_dim, dist,
                                            DistanceScope::UPDATE);

                hdr.degree = (uint16_t)std::min((size_t)_Mmax, pruned.size());
                for (uint16_t j = 0; j < hdr.degree; j++) {
                    nbrs_ptr[j] = pruned[j];
                }
            }

            memcpy(slot_ptr, &hdr, sizeof(hdr));
            page_modified = true;
            _stats.deferred_edges_drained.fetch_add(sources.size());
        }

        _bp.unpin(cur_page, page_modified);
    }
}

// Explicit instantiations
template void InPlaceGraphStore::drain_deferred_edges<float>(
    unsigned, unsigned, float, unsigned, diskann::Distance<float>*);
template void InPlaceGraphStore::drain_deferred_edges<uint8_t>(
    unsigned, unsigned, float, unsigned, diskann::Distance<uint8_t>*);
template void InPlaceGraphStore::drain_deferred_edges<int8_t>(
    unsigned, unsigned, float, unsigned, diskann::Distance<int8_t>*);

// ===========================================================================
// sweep_repair_round -- distance-based pruning + generation rollover
// ===========================================================================
template<typename T>
uint32_t InPlaceGraphStore::sweep_repair_round(
    uint32_t budget, unsigned R, unsigned C, float alpha,
    unsigned aligned_dim, diskann::Distance<T>* dist) {

    uint32_t total_pages;
    {
        std::lock_guard<std::mutex> lk(_alloc_mtx);
        total_pages = (uint32_t)_page_dir.size();
    }
    if (total_pages == 0) return 0;

    InPlaceSearchScratch local_scratch;
    local_scratch.init(aligned_dim, _n_chunks, sizeof(T));
    std::vector<unsigned> candidates;
    candidates.reserve(_Mmax * 3);
    tsl::robin_set<unsigned> seen;
    seen.reserve(_Mmax * 3);
    std::vector<Neighbor> pool;
    pool.reserve(_Mmax * 3);
    std::vector<unsigned> pruned;
    pruned.reserve(_Mmax);
    std::vector<uint8_t> pq_scratch_buf;
    std::vector<float> dist_buf;

    uint32_t repaired = 0;
    uint32_t pages_visited = 0;
    bool completed_generation = false;

    while (pages_visited < budget && pages_visited < total_pages) {
        uint32_t pid = _sweep_cursor % total_pages;
        _sweep_cursor++;
        pages_visited++;
        if ((_sweep_cursor % total_pages) == 0) {
            completed_generation = true;
        }

        auto& frame = _bp.pin(pid, FrameRegion::UPDATE);
        uint16_t spp;
        memcpy(&spp, frame.data, 2);

        bool page_modified = false;
        for (uint16_t s = 0; s < spp; s++) {
            if (!test_bitmap(frame.data, s)) continue;

            char* slot_ptr = frame.data + header_bytes() + s * _slot_size;
            PackedSlotHeader hdr;
            memcpy(&hdr, slot_ptr, sizeof(hdr));

            if (hdr.flags & FLAG_DELETED) continue;
            if (hdr.degree == 0) continue;

            uint32_t* nbrs = (uint32_t*)(slot_ptr + sizeof(PackedSlotHeader) +
                                          _aligned_dim * _elem_size);

            bool has_dead = false;
            for (uint16_t j = 0; j < hdr.degree; j++) {
                if (!is_active(nbrs[j])) {
                    has_dead = true;
                    break;
                }
            }
            if (!has_dead) continue;

            // Collect all live candidates: existing live nbrs + replacements
            // from dead nbrs' neighborhoods
            candidates.clear();
            seen.clear();
            seen.insert(hdr.node_id);

            for (uint16_t j = 0; j < hdr.degree; j++) {
                if (is_active(nbrs[j])) {
                    if (seen.find(nbrs[j]) == seen.end()) {
                        candidates.push_back(nbrs[j]);
                        seen.insert(nbrs[j]);
                    }
                } else {
                    auto dead_view = pin_node(nbrs[j], READ);
                    if (dead_view._page_id != INVALID_PAGE) {
                        for (uint16_t dj = 0; dj < dead_view.degree; dj++) {
                            uint32_t cand = dead_view.neighbors[dj];
                            if (is_active(cand) &&
                                seen.find(cand) == seen.end()) {
                                candidates.push_back(cand);
                                seen.insert(cand);
                            }
                        }
                        unpin_node(dead_view);
                    }
                }
            }

            if (candidates.empty()) {
                hdr.degree = 0;
                memcpy(slot_ptr, &hdr, sizeof(hdr));
                page_modified = true;
                repaired++;
                continue;
            }

            // Build distance-ranked pool and prune properly
            pool.clear();
            pool.reserve(std::max<size_t>(pool.capacity(),
                                          candidates.size()));

            char* node_coords = slot_ptr + sizeof(PackedSlotHeader);

            if (_n_chunks > 0 && _pq_codes) {
                pq_scratch_buf.resize(candidates.size() * _n_chunks + 64);
                dist_buf.resize(candidates.size());
                auto dist_start = std::chrono::steady_clock::now();
                compute_pq_dists_src(hdr.node_id, candidates.data(),
                                     dist_buf.data(),
                                     (uint32_t)candidates.size(),
                                     pq_scratch_buf.data());
                _stats.record_distance(DistanceScope::UPDATE,
                                       elapsed_ns(dist_start),
                                       (uint64_t)candidates.size());
                for (size_t ci = 0; ci < candidates.size(); ci++) {
                    pool.emplace_back(candidates[ci], dist_buf[ci], true);
                }
            } else {
                T* aligned_src = reinterpret_cast<T*>(local_scratch.aux_coord_scratch);
                T* aligned_cand = reinterpret_cast<T*>(
                    local_scratch.aux_coord_scratch + aligned_dim * sizeof(T));
                memcpy(aligned_src, node_coords, aligned_dim * sizeof(T));
                uint64_t local_cmps = 0;
                auto dist_start = std::chrono::steady_clock::now();
                for (auto cand : candidates) {
                    auto cv = pin_node(cand, READ);
                    float d = std::numeric_limits<float>::max();
                    if (cv._page_id != INVALID_PAGE) {
                        memcpy(aligned_cand, cv.coords,
                               aligned_dim * sizeof(T));
                        d = dist->compare(aligned_src, aligned_cand,
                                          aligned_dim);
                        local_cmps++;
                        unpin_node(cv);
                    }
                    pool.emplace_back(cand, d, true);
                }
                _stats.record_distance(DistanceScope::UPDATE,
                                       elapsed_ns(dist_start),
                                       local_cmps);
            }

            pruned.clear();
            graph_prune_neighbors_pq<T>(hdr.node_id, pool, R, C, alpha,
                                        pruned, this, &local_scratch,
                                        aligned_dim, dist,
                                        DistanceScope::UPDATE);

            hdr.degree = (uint16_t)std::min((size_t)_Mmax, pruned.size());
            for (uint16_t j = 0; j < hdr.degree; j++) {
                nbrs[j] = pruned[j];
            }
            memcpy(slot_ptr, &hdr, sizeof(hdr));
            page_modified = true;
            repaired++;
        }

        _bp.unpin(pid, page_modified);
    }

    _stats.nodes_repaired.fetch_add(repaired);

    if (completed_generation) {
        reclaim_quarantined();
    }

    return repaired;
}

template uint32_t InPlaceGraphStore::sweep_repair_round<float>(
    uint32_t, unsigned, unsigned, float, unsigned, diskann::Distance<float>*);
template uint32_t InPlaceGraphStore::sweep_repair_round<uint8_t>(
    uint32_t, unsigned, unsigned, float, unsigned, diskann::Distance<uint8_t>*);
template uint32_t InPlaceGraphStore::sweep_repair_round<int8_t>(
    uint32_t, unsigned, unsigned, float, unsigned, diskann::Distance<int8_t>*);

// ===========================================================================
// Explicit template instantiations for graph ops
// ===========================================================================
template std::pair<uint32_t, uint32_t>
graph_iterate_to_fixed_point<float>(
    const float*, unsigned, const std::vector<unsigned>&, unsigned,
    InPlaceGraphStore*, unsigned, Distance<float>*,
    InPlaceSearchScratch*, std::vector<Neighbor>&,
    tsl::robin_set<unsigned>*, DistanceScope, unsigned,
    const std::string&, double, unsigned);

template std::pair<uint32_t, uint32_t>
graph_iterate_to_fixed_point<uint8_t>(
    const uint8_t*, unsigned, const std::vector<unsigned>&, unsigned,
    InPlaceGraphStore*, unsigned, Distance<uint8_t>*,
    InPlaceSearchScratch*, std::vector<Neighbor>&,
    tsl::robin_set<unsigned>*, DistanceScope, unsigned,
    const std::string&, double, unsigned);

template std::pair<uint32_t, uint32_t>
graph_iterate_to_fixed_point<int8_t>(
    const int8_t*, unsigned, const std::vector<unsigned>&, unsigned,
    InPlaceGraphStore*, unsigned, Distance<int8_t>*,
    InPlaceSearchScratch*, std::vector<Neighbor>&,
    tsl::robin_set<unsigned>*, DistanceScope, unsigned,
    const std::string&, double, unsigned);

template void graph_occlude_list_pq<float>(
    std::vector<Neighbor>&, float, unsigned, unsigned,
    std::vector<Neighbor>&, std::vector<float>&,
    InPlaceGraphStore*, InPlaceSearchScratch*, DistanceScope);
template void graph_occlude_list_pq<uint8_t>(
    std::vector<Neighbor>&, float, unsigned, unsigned,
    std::vector<Neighbor>&, std::vector<float>&,
    InPlaceGraphStore*, InPlaceSearchScratch*, DistanceScope);
template void graph_occlude_list_pq<int8_t>(
    std::vector<Neighbor>&, float, unsigned, unsigned,
    std::vector<Neighbor>&, std::vector<float>&,
    InPlaceGraphStore*, InPlaceSearchScratch*, DistanceScope);

template void graph_occlude_list<float>(
    std::vector<Neighbor>&, float, unsigned, unsigned,
    std::vector<Neighbor>&, std::vector<float>&,
    InPlaceGraphStore*, InPlaceSearchScratch*, unsigned, Distance<float>*,
    DistanceScope);

template void graph_prune_neighbors_pq<float>(
    unsigned, std::vector<Neighbor>&, unsigned, unsigned, float,
    std::vector<unsigned>&, InPlaceGraphStore*, InPlaceSearchScratch*,
    unsigned, Distance<float>*, DistanceScope);
template void graph_prune_neighbors_pq<uint8_t>(
    unsigned, std::vector<Neighbor>&, unsigned, unsigned, float,
    std::vector<unsigned>&, InPlaceGraphStore*, InPlaceSearchScratch*,
    unsigned, Distance<uint8_t>*, DistanceScope);
template void graph_prune_neighbors_pq<int8_t>(
    unsigned, std::vector<Neighbor>&, unsigned, unsigned, float,
    std::vector<unsigned>&, InPlaceGraphStore*, InPlaceSearchScratch*,
    unsigned, Distance<int8_t>*, DistanceScope);

}  // namespace inplace
}  // namespace diskann
