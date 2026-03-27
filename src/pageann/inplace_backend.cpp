// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "pageann/inplace_backend.h"
#include "index.h"
#include "logger.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>

namespace diskann {
namespace inplace {

// ===========================================================================
// InPlaceIOStats
// ===========================================================================
void InPlaceIOStats::reset() {
    physical_bytes_written.store(0);
    physical_bytes_read.store(0);
    logical_bytes_written.store(0);
    logical_bytes_read.store(0);
    cache_hits.store(0);
    cache_misses.store(0);
    evictions.store(0);
    dirty_evictions.store(0);
    pages_flushed.store(0);
    dirty_page_bytes_flushed.store(0);
    total_inserts.store(0);
    total_deletes.store(0);
    nodes_repaired.store(0);
    tombstone_count.store(0);
    repair_queue_length.store(0);
    deferred_edges_pushed.store(0);
    deferred_edges_drained.store(0);
    query_region_hits.store(0);
    update_region_hits.store(0);
    shared_region_hits.store(0);
    cross_region_spills.store(0);
    query_distance_ns.store(0);
    update_distance_ns.store(0);
    total_distance_ns.store(0);
    query_distance_ops.store(0);
    update_distance_ops.store(0);
    total_distance_ops.store(0);
}

std::string InPlaceIOStats::to_json() const {
    std::ostringstream ss;
    ss << "{"
       << "\"physical_bytes_written\":" << physical_bytes_written.load() << ","
       << "\"physical_bytes_read\":" << physical_bytes_read.load() << ","
       << "\"logical_bytes_written\":" << logical_bytes_written.load() << ","
       << "\"logical_bytes_read\":" << logical_bytes_read.load() << ","
       << "\"cache_hits\":" << cache_hits.load() << ","
       << "\"cache_misses\":" << cache_misses.load() << ","
       << "\"evictions\":" << evictions.load() << ","
       << "\"dirty_evictions\":" << dirty_evictions.load() << ","
       << "\"pages_flushed\":" << pages_flushed.load() << ","
       << "\"dirty_page_bytes_flushed\":" << dirty_page_bytes_flushed.load()
       << ","
       << "\"total_inserts\":" << total_inserts.load() << ","
       << "\"total_deletes\":" << total_deletes.load() << ","
       << "\"nodes_repaired\":" << nodes_repaired.load() << ","
       << "\"tombstone_count\":" << tombstone_count.load() << ","
       << "\"deferred_edges_pushed\":" << deferred_edges_pushed.load() << ","
       << "\"deferred_edges_drained\":" << deferred_edges_drained.load() << ","
       << "\"query_region_hits\":" << query_region_hits.load() << ","
       << "\"update_region_hits\":" << update_region_hits.load() << ","
       << "\"shared_region_hits\":" << shared_region_hits.load() << ","
       << "\"cross_region_spills\":" << cross_region_spills.load() << ","
       << "\"query_distance_ns\":" << query_distance_ns.load() << ","
       << "\"update_distance_ns\":" << update_distance_ns.load() << ","
       << "\"total_distance_ns\":" << total_distance_ns.load() << ","
       << "\"query_distance_ops\":" << query_distance_ops.load() << ","
       << "\"update_distance_ops\":" << update_distance_ops.load() << ","
       << "\"total_distance_ops\":" << total_distance_ops.load()
       << "}";
    return ss.str();
}

// ===========================================================================
// InPlaceSearchScratch
// ===========================================================================
void InPlaceSearchScratch::init(uint32_t aligned_dim, uint32_t n_chunks,
                                uint32_t elem_size) {
    size_t coord_bytes = (size_t)MAX_SCRATCH_NODES * aligned_dim * elem_size;
    if (coord_bytes > 0) {
        posix_memalign((void**)&aligned_coord_scratch, 32, coord_bytes);
        memset(aligned_coord_scratch, 0, coord_bytes);

        size_t aux_bytes = (size_t)aligned_dim * elem_size * 2;
        posix_memalign((void**)&aux_coord_scratch, 32, aux_bytes);
        memset(aux_coord_scratch, 0, aux_bytes);
    }

    size_t dist_bytes = (size_t)MAX_SCRATCH_NODES * sizeof(float);
    posix_memalign((void**)&dist_scratch, 32, dist_bytes);
    memset(dist_scratch, 0, dist_bytes);

    if (n_chunks > 0) {
        size_t pq_dist_bytes = (size_t)n_chunks * 256 * sizeof(float);
        posix_memalign((void**)&pq_dists, 32, pq_dist_bytes);
        memset(pq_dists, 0, pq_dist_bytes);

        size_t pq_coord_bytes = (size_t)n_chunks * MAX_SCRATCH_NODES;
        posix_memalign((void**)&pq_coord_scratch, 32, pq_coord_bytes);
        memset(pq_coord_scratch, 0, pq_coord_bytes);
    }
}

InPlaceSearchScratch::~InPlaceSearchScratch() {
    if (aligned_coord_scratch) free(aligned_coord_scratch);
    if (aux_coord_scratch)     free(aux_coord_scratch);
    if (pq_dists)              free(pq_dists);
    if (dist_scratch)          free(dist_scratch);
    if (pq_coord_scratch)      free(pq_coord_scratch);
}

void InPlaceSearchScratch::flush_distance_stats(InPlaceIOStats& stats) {
    if (local_total_distance_ns == 0 && local_total_distance_ops == 0) return;
    stats.total_distance_ns.fetch_add(local_total_distance_ns, std::memory_order_relaxed);
    stats.total_distance_ops.fetch_add(local_total_distance_ops, std::memory_order_relaxed);
    if (local_query_distance_ns || local_query_distance_ops) {
        stats.query_distance_ns.fetch_add(local_query_distance_ns, std::memory_order_relaxed);
        stats.query_distance_ops.fetch_add(local_query_distance_ops, std::memory_order_relaxed);
    }
    if (local_update_distance_ns || local_update_distance_ops) {
        stats.update_distance_ns.fetch_add(local_update_distance_ns, std::memory_order_relaxed);
        stats.update_distance_ops.fetch_add(local_update_distance_ops, std::memory_order_relaxed);
    }
    local_query_distance_ns = local_update_distance_ns = local_total_distance_ns = 0;
    local_query_distance_ops = local_update_distance_ops = local_total_distance_ops = 0;
}

// ===========================================================================
// DeferredEdgeBuffer
// ===========================================================================
DeferredEdgeBuffer::DeferredEdgeBuffer() {
    _shards.reserve(kShardCount);
    for (size_t i = 0; i < kShardCount; ++i) {
        _shards.emplace_back(std::make_unique<Shard>());
    }
}

size_t DeferredEdgeBuffer::shard_index(uint32_t target) const {
    return static_cast<size_t>(target) % kShardCount;
}

size_t DeferredEdgeBuffer::push(uint32_t target, uint32_t src) {
    auto& shard = *_shards[shard_index(target)];
    std::lock_guard<std::mutex> lk(shard.mtx);
    auto it = shard.target_to_state.find(target);
    bool inserted_target = (it == shard.target_to_state.end());
    auto& state = inserted_target ? shard.target_to_state[target] : it.value();
    auto& sources = state.sources;
    if (std::find(sources.begin(), sources.end(), src) == sources.end()) {
        sources.push_back(src);
        _edge_count.fetch_add(1, std::memory_order_relaxed);
        if (inserted_target) {
            _target_count.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return sources.size();
}

size_t DeferredEdgeBuffer::size() const {
    return _edge_count.load(std::memory_order_relaxed);
}

size_t DeferredEdgeBuffer::target_count() const {
    return _target_count.load(std::memory_order_relaxed);
}

size_t DeferredEdgeBuffer::aged_target_count() const {
    return _aged_target_count.load(std::memory_order_relaxed);
}

void DeferredEdgeBuffer::snapshot(uint32_t target, std::vector<uint32_t>& out) const {
    auto& shard = *_shards[shard_index(target)];
    std::lock_guard<std::mutex> lk(shard.mtx);
    auto it = shard.target_to_state.find(target);
    if (it == shard.target_to_state.end()) return;
    out.insert(out.end(), it->second.sources.begin(), it->second.sources.end());
}

std::vector<uint32_t> DeferredEdgeBuffer::select_targets_for_repair(
    uint32_t pending_threshold,
    size_t limit,
    bool force_all) {
    std::vector<uint32_t> targets;
    if (limit != 0) {
        targets.reserve(limit);
    } else {
        targets.reserve(target_count());
    }
    auto try_select = [&](bool aged_pass) {
        for (auto& shard_ptr : _shards) {
            auto& shard = *shard_ptr;
            std::lock_guard<std::mutex> lk(shard.mtx);
            for (auto it = shard.target_to_state.begin(); it != shard.target_to_state.end(); ++it) {
                auto& state = it.value();
                if (state.selected) continue;
                bool aged = state.scan_age_bit == 1;
                bool over_threshold =
                    state.sources.size() > static_cast<size_t>(pending_threshold);
                bool select = force_all || (aged_pass ? aged : (!aged && over_threshold));
                if (select && (limit == 0 || targets.size() < limit)) {
                    state.selected = true;
                    targets.push_back(it.key());
                    continue;
                }
                if (!force_all && !aged_pass && state.scan_age_bit == 0) {
                    state.scan_age_bit = 1;
                    _aged_target_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    };
    try_select(true);
    if (!force_all && limit != 0 && targets.size() >= limit) {
        return targets;
    }
    try_select(false);
    return targets;
}

size_t DeferredEdgeBuffer::finish_target(
    uint32_t target,
    const std::vector<uint32_t>& remove_sources,
    bool set_scan_age_bit) {
    auto& shard = *_shards[shard_index(target)];
    std::lock_guard<std::mutex> lk(shard.mtx);
    auto it = shard.target_to_state.find(target);
    if (it == shard.target_to_state.end()) return 0;

    auto& state = it.value();
    bool was_aged = state.scan_age_bit != 0;
    if (!remove_sources.empty()) {
        tsl::robin_set<uint32_t> remove_set;
        remove_set.reserve(remove_sources.size());
        for (uint32_t src : remove_sources) {
            remove_set.insert(src);
        }
        auto erase_begin = std::remove_if(
            state.sources.begin(), state.sources.end(),
            [&remove_set](uint32_t src) { return remove_set.find(src) != remove_set.end(); });
        size_t removed = static_cast<size_t>(state.sources.end() - erase_begin);
        if (removed > 0) {
            state.sources.erase(erase_begin, state.sources.end());
            _edge_count.fetch_sub(removed, std::memory_order_relaxed);
        }
    }

    state.selected = false;
    bool keep_age = set_scan_age_bit && !state.sources.empty();
    state.scan_age_bit = keep_age ? 1 : 0;
    if (was_aged && !keep_age) {
        _aged_target_count.fetch_sub(1, std::memory_order_relaxed);
    } else if (!was_aged && keep_age) {
        _aged_target_count.fetch_add(1, std::memory_order_relaxed);
    }

    if (state.sources.empty()) {
        shard.target_to_state.erase(it);
        _target_count.fetch_sub(1, std::memory_order_relaxed);
        return 0;
    }
    return state.sources.size();
}

OversizeNodeTable::OversizeNodeTable() {
    _shards.reserve(kShardCount);
    for (size_t i = 0; i < kShardCount; ++i) {
        _shards.emplace_back(std::make_unique<Shard>());
    }
}

size_t OversizeNodeTable::shard_index(uint32_t node_id) const {
    return static_cast<size_t>(node_id) % kShardCount;
}

void OversizeNodeTable::note(uint32_t node_id, uint32_t degree, uint32_t epoch) {
    auto& shard = *_shards[shard_index(node_id)];
    std::lock_guard<std::mutex> lk(shard.mtx);
    auto it = shard.nodes.find(node_id);
    if (it == shard.nodes.end()) {
        shard.nodes[node_id] = OversizeNodeRecord{epoch, epoch, degree};
        _size.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    auto& record = shard.nodes[node_id];
    record.last_epoch = epoch;
    record.last_degree = degree;
}

void OversizeNodeTable::erase(uint32_t node_id) {
    auto& shard = *_shards[shard_index(node_id)];
    std::lock_guard<std::mutex> lk(shard.mtx);
    auto it = shard.nodes.find(node_id);
    if (it == shard.nodes.end()) return;
    shard.nodes.erase(it);
    _size.fetch_sub(1, std::memory_order_relaxed);
}

size_t OversizeNodeTable::size() const {
    return _size.load(std::memory_order_relaxed);
}

bool OversizeNodeTable::has_aged(uint32_t current_epoch, uint32_t age_threshold_rounds) const {
    if (age_threshold_rounds == 0) return size() > 0;
    for (const auto& shard_ptr : _shards) {
        auto& shard = *shard_ptr;
        std::lock_guard<std::mutex> lk(shard.mtx);
        for (const auto& kv : shard.nodes) {
            if (current_epoch >= kv.second.first_epoch &&
                current_epoch - kv.second.first_epoch >= age_threshold_rounds) {
                return true;
            }
        }
    }
    return false;
}

std::vector<uint32_t> OversizeNodeTable::collect_candidates(
    uint32_t current_epoch,
    uint32_t age_threshold_rounds,
    size_t limit) const {
    std::vector<uint32_t> aged;
    std::vector<uint32_t> fallback;
    aged.reserve(limit == 0 ? size() : limit);
    fallback.reserve(limit == 0 ? size() : limit);
    for (const auto& shard_ptr : _shards) {
        auto& shard = *shard_ptr;
        std::lock_guard<std::mutex> lk(shard.mtx);
        for (const auto& kv : shard.nodes) {
            bool old_enough = age_threshold_rounds == 0 ||
                              (current_epoch >= kv.second.first_epoch &&
                               current_epoch - kv.second.first_epoch >= age_threshold_rounds);
            if (old_enough) {
                aged.push_back(kv.first);
                if (limit != 0 && aged.size() >= limit) return aged;
            } else if (limit == 0 || fallback.size() < limit) {
                fallback.push_back(kv.first);
            }
        }
    }
    if (limit != 0 && aged.size() + fallback.size() > limit) {
        fallback.resize(limit - aged.size());
    }
    aged.insert(aged.end(), fallback.begin(), fallback.end());
    return aged;
}

// ===========================================================================
// BufferPool
// ===========================================================================
BufferPool::~BufferPool() {
    stop_bg_flush();
    for (auto& f : _frames) {
        if (f.data) {
            free(f.data);
            f.data = nullptr;
        }
    }
    if (_heap_fd >= 0) {
        ::close(_heap_fd);
        _heap_fd = -1;
    }
}

void BufferPool::init(uint32_t page_size, uint32_t num_frames,
                      const std::string& heap_path, InPlaceIOStats* stats,
                      float region_query_frac, float region_update_frac,
                      uint32_t flush_budget_pages_per_cycle,
                      uint32_t flush_wakeup_ms,
                      bool truncate_heap) {
    _page_size  = page_size;
    _num_frames = num_frames;
    _stats      = stats;
    _flush_budget_pages_per_cycle = std::max<uint32_t>(1, flush_budget_pages_per_cycle);
    _flush_wakeup_ms = std::max<uint32_t>(1, flush_wakeup_ms);

    _frames.clear();
    _frame_waiters.clear();
    for (uint32_t i = 0; i < num_frames; ++i) {
        _frames.emplace_back();
        auto& f = _frames.back();
        posix_memalign((void**)&f.data, 4096, page_size);
        memset(f.data, 0, page_size);
        _frame_waiters.emplace_back(std::make_unique<FrameWaitState>());
    }

    uint32_t desired_shards = std::min<uint32_t>(16, std::max<uint32_t>(4, num_frames / 64));
    _num_shards = std::max<uint32_t>(1, std::min<uint32_t>(num_frames == 0 ? 1 : num_frames, desired_shards));
    _shards.clear();
    _shards.reserve(_num_shards);
    uint32_t frame_begin = 0;
    for (uint32_t shard_idx = 0; shard_idx < _num_shards; ++shard_idx) {
        auto shard = std::make_unique<ShardState>();
        uint32_t frame_end = ((shard_idx + 1) * num_frames) / _num_shards;
        shard->frame_begin = frame_begin;
        shard->frame_end = frame_end;
        shard->clock_hand = frame_begin;
        shard->flush_cursor = frame_begin;
        uint32_t shard_frames = frame_end - frame_begin;
        shard->query_budget = static_cast<uint32_t>(shard_frames * region_query_frac);
        shard->update_budget = static_cast<uint32_t>(shard_frames * region_update_frac);
        _shards.emplace_back(std::move(shard));
        frame_begin = frame_end;
    }

    int open_flags = O_RDWR | O_CREAT | O_DIRECT | O_LARGEFILE;
    if (truncate_heap) open_flags |= O_TRUNC;
    _heap_fd = ::open(heap_path.c_str(), open_flags, 0644);
    if (_heap_fd < 0) {
        open_flags = O_RDWR | O_CREAT;
        if (truncate_heap) open_flags |= O_TRUNC;
        _heap_fd = ::open(heap_path.c_str(), open_flags, 0644);
    }
    if (_heap_fd < 0) {
        throw ANNException("Failed to open heap file: " + heap_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    if (truncate_heap) {
        _next_page = 0;
    } else {
        struct stat st;
        if (::fstat(_heap_fd, &st) != 0) {
            throw ANNException("Failed to stat heap file: " + heap_path, -1,
                               __FUNCSIG__, __FILE__, __LINE__);
        }
        _next_page = static_cast<uint32_t>((st.st_size + _page_size - 1) / _page_size);
    }
}

uint32_t BufferPool::shard_index(uint32_t page_id) const {
    return _num_shards == 0 ? 0 : (page_id % _num_shards);
}

BufferPool::ShardState& BufferPool::shard_for_page(uint32_t page_id) {
    return *_shards[shard_index(page_id)];
}

const BufferPool::ShardState& BufferPool::shard_for_page(uint32_t page_id) const {
    return *_shards[shard_index(page_id)];
}

void BufferPool::read_page(uint32_t page_id, char* buf) {
    ssize_t n = ::pread(_heap_fd, buf, _page_size,
                        (off_t)page_id * _page_size);
    if (n < 0) n = 0;
    if ((size_t)n < _page_size) {
        memset(buf + n, 0, _page_size - n);
    }
    if (_stats) _stats->physical_bytes_read.fetch_add(_page_size);
}

void BufferPool::write_page(uint32_t page_id, const char* buf) {
    ssize_t n = ::pwrite(_heap_fd, buf, _page_size,
                         (off_t)page_id * _page_size);
    (void)n;
    if (_stats) _stats->physical_bytes_written.fetch_add(_page_size);
}

uint32_t BufferPool::evict_one(ShardState& shard, std::unique_lock<std::mutex>& lk, FrameRegion preferred) {
    auto evict_frame = [&](uint32_t idx) -> uint32_t {
        auto& f = _frames[idx];
        const uint32_t old_page_id = f.page_id;
        const bool was_dirty = f.dirty.exchange(false, std::memory_order_acq_rel);
        std::vector<char> flush_copy;
        if (was_dirty) {
            flush_copy.assign(f.data, f.data + _page_size);
            f.flushing.store(true, std::memory_order_release);
        }
        if (old_page_id != INVALID_PAGE) {
            shard.page_table.erase(old_page_id);
        }
        if (f.region == FrameRegion::QUERY && shard.query_used > 0) shard.query_used--;
        else if (f.region == FrameRegion::UPDATE && shard.update_used > 0) shard.update_used--;
        if (_stats) {
            _stats->evictions.fetch_add(1);
            if (was_dirty) _stats->dirty_evictions.fetch_add(1);
        }
        f.page_id = INVALID_PAGE;
        f.loading.store(true, std::memory_order_release);
        f.pin_count.store(0, std::memory_order_release);
        f.ref_bit.store(0, std::memory_order_release);
        f.protected_credit.store(0, std::memory_order_release);
        f.region = FrameRegion::SHARED;
        lk.unlock();
        if (was_dirty) {
            write_page(old_page_id, flush_copy.data());
            if (_stats) {
                _stats->pages_flushed.fetch_add(1);
                _stats->dirty_page_bytes_flushed.fetch_add(_page_size);
            }
            f.flushing.store(false, std::memory_order_release);
        }
        lk.lock();
        return idx;
    };

    auto try_evict_region = [&](FrameRegion r, bool only_clean) -> uint32_t {
        uint32_t shard_frames = shard.frame_end - shard.frame_begin;
        uint32_t budget = std::max<uint32_t>(8, std::min<uint32_t>(shard_frames * 2, 256));
        uint32_t scanned = 0;
        while (scanned < budget) {
            uint32_t idx = shard.clock_hand;
            auto& f = _frames[idx];
            shard.clock_hand++;
            if (shard.clock_hand >= shard.frame_end) shard.clock_hand = shard.frame_begin;
            scanned++;
            if (f.region != r) continue;
            if (f.loading.load(std::memory_order_acquire)) continue;
            if (f.flushing.load(std::memory_order_acquire)) continue;
            if (f.pin_count.load(std::memory_order_acquire) > 0) continue;
            if (only_clean && f.dirty.load(std::memory_order_acquire)) continue;
            uint8_t protect = f.protected_credit.load(std::memory_order_acquire);
            if (protect > 0) {
                f.protected_credit.fetch_sub(1, std::memory_order_acq_rel);
                f.ref_bit.store(1, std::memory_order_release);
                continue;
            }
            if (f.ref_bit.exchange(0, std::memory_order_acq_rel)) {
                continue;
            }
            return evict_frame(idx);
        }
        return INVALID_PAGE;
    };

    uint32_t idx;

    if (preferred == FrameRegion::QUERY) {
        idx = try_evict_region(FrameRegion::QUERY, true);
        if (idx != INVALID_PAGE) return idx;
        idx = try_evict_region(FrameRegion::SHARED, false);
        if (idx != INVALID_PAGE) return idx;
        idx = try_evict_region(FrameRegion::UPDATE, true);
        if (idx != INVALID_PAGE) return idx;
        idx = try_evict_region(FrameRegion::SHARED, false);
        if (idx != INVALID_PAGE) return idx;
    } else {
        idx = try_evict_region(FrameRegion::UPDATE, false);
        if (idx != INVALID_PAGE) return idx;
        idx = try_evict_region(FrameRegion::SHARED, false);
        if (idx != INVALID_PAGE) return idx;
        idx = try_evict_region(FrameRegion::QUERY, true);
        if (idx != INVALID_PAGE) return idx;
    }

    // last resort: evict anything we can
    for (uint32_t i = shard.frame_begin; i < shard.frame_end; i++) {
        auto& f = _frames[i];
        if (f.loading.load(std::memory_order_acquire)) continue;
        if (f.flushing.load(std::memory_order_acquire)) continue;
        if (f.pin_count.load(std::memory_order_acquire) == 0) {
            return evict_frame(i);
        }
    }
    throw ANNException("BufferPool: all frames pinned, cannot evict", -1,
                       __FUNCSIG__, __FILE__, __LINE__);
}

PageFrame& BufferPool::pin(uint32_t page_id, FrameRegion hint) {
    ShardState& shard = shard_for_page(page_id);
    while (true) {
        std::unique_lock<std::mutex> lk(shard.mtx);
        auto it = shard.page_table.find(page_id);
        if (it != shard.page_table.end()) {
            uint32_t frame_idx = it->second;
            auto& f = _frames[frame_idx];
            if (f.loading.load(std::memory_order_acquire)) {
                FrameWaitState* wait_state = _frame_waiters[frame_idx].get();
                lk.unlock();
                std::unique_lock<std::mutex> wait_lk(wait_state->mtx);
                wait_state->cv.wait(wait_lk, [&f] {
                    return !f.loading.load(std::memory_order_acquire);
                });
                continue;
            }
            f.pin_count.fetch_add(1, std::memory_order_acq_rel);
            f.ref_bit.store(1, std::memory_order_release);
            if (_stats) {
                _stats->cache_hits.fetch_add(1);
                if (f.region == FrameRegion::QUERY) _stats->query_region_hits.fetch_add(1);
                else if (f.region == FrameRegion::UPDATE) _stats->update_region_hits.fetch_add(1);
                else _stats->shared_region_hits.fetch_add(1);
            }
            return f;
        }

        if (_stats) _stats->cache_misses.fetch_add(1);

        uint32_t frame_idx = INVALID_PAGE;
        for (uint32_t i = shard.frame_begin; i < shard.frame_end; ++i) {
            auto& f = _frames[i];
            if (f.page_id == INVALID_PAGE && !f.loading.load(std::memory_order_acquire) &&
                !f.flushing.load(std::memory_order_acquire)) {
                frame_idx = i;
                f.loading.store(true, std::memory_order_release);
                break;
            }
        }
        if (frame_idx == INVALID_PAGE) {
            frame_idx = evict_one(shard, lk, hint);
        }

        auto& f = _frames[frame_idx];
        f.page_id = page_id;
        f.pin_count.store(1, std::memory_order_release);
        f.dirty.store(false, std::memory_order_release);
        f.ref_bit.store(1, std::memory_order_release);
        f.flushing.store(false, std::memory_order_release);
        f.protected_credit.store(0, std::memory_order_release);

        bool is_spill = false;
        if (hint == FrameRegion::QUERY) {
            if (shard.query_used < shard.query_budget) {
                f.region = FrameRegion::QUERY;
                shard.query_used++;
            } else {
                f.region = FrameRegion::SHARED;
                is_spill = true;
            }
        } else if (hint == FrameRegion::UPDATE) {
            if (shard.update_used < shard.update_budget) {
                f.region = FrameRegion::UPDATE;
                shard.update_used++;
            } else {
                f.region = FrameRegion::SHARED;
                is_spill = true;
            }
        } else {
            f.region = FrameRegion::SHARED;
        }

        if (is_spill && _stats) _stats->cross_region_spills.fetch_add(1);
        if (_stats) {
            if (f.region == FrameRegion::QUERY) _stats->query_region_hits.fetch_add(1);
            else if (f.region == FrameRegion::UPDATE) _stats->update_region_hits.fetch_add(1);
            else _stats->shared_region_hits.fetch_add(1);
        }

        shard.page_table[page_id] = frame_idx;
        lk.unlock();

        read_page(page_id, f.data);
        f.loading.store(false, std::memory_order_release);
        {
            std::lock_guard<std::mutex> wait_lk(_frame_waiters[frame_idx]->mtx);
        }
        _frame_waiters[frame_idx]->cv.notify_all();
        return f;
    }
}

PageFrame* BufferPool::try_pin_resident(uint32_t page_id, FrameRegion hint) {
    ShardState& shard = shard_for_page(page_id);
    std::unique_lock<std::mutex> lk(shard.mtx);
    auto it = shard.page_table.find(page_id);
    if (it == shard.page_table.end()) return nullptr;
    auto& f = _frames[it->second];
    if (f.loading.load(std::memory_order_acquire)) return nullptr;
    f.pin_count.fetch_add(1, std::memory_order_acq_rel);
    f.ref_bit.store(1, std::memory_order_release);
    if (_stats) {
        _stats->cache_hits.fetch_add(1);
        if (f.region == FrameRegion::QUERY) _stats->query_region_hits.fetch_add(1);
        else if (f.region == FrameRegion::UPDATE) _stats->update_region_hits.fetch_add(1);
        else _stats->shared_region_hits.fetch_add(1);
        if (hint != FrameRegion::SHARED && f.region != hint) {
            _stats->cross_region_spills.fetch_add(1);
        }
    }
    return &f;
}

void BufferPool::protect_page(uint32_t page_id, uint8_t credit, FrameRegion hint) {
    if (credit == 0) return;
    auto& f = pin(page_id, hint);
    uint8_t cur = f.protected_credit.load(std::memory_order_acquire);
    while (cur < credit &&
           !f.protected_credit.compare_exchange_weak(cur, credit, std::memory_order_acq_rel)) {
    }
    f.ref_bit.store(1, std::memory_order_release);
    unpin(page_id, false);
}

void BufferPool::unpin(uint32_t page_id, bool dirty) {
    ShardState& shard = shard_for_page(page_id);
    std::lock_guard<std::mutex> lk(shard.mtx);
    auto it = shard.page_table.find(page_id);
    if (it == shard.page_table.end()) return;
    auto& f = _frames[it->second];
    if (dirty) {
        f.dirty.store(true, std::memory_order_release);
    }
    uint32_t cur = f.pin_count.load(std::memory_order_acquire);
    if (cur > 0) {
        f.pin_count.fetch_sub(1, std::memory_order_acq_rel);
    }
}

void BufferPool::mark_dirty(uint32_t page_id) {
    ShardState& shard = shard_for_page(page_id);
    std::lock_guard<std::mutex> lk(shard.mtx);
    auto it = shard.page_table.find(page_id);
    if (it != shard.page_table.end()) {
        _frames[it->second].dirty.store(true, std::memory_order_release);
    }
}

void BufferPool::flush_all_dirty() {
    for (auto& shard_ptr : _shards) {
        auto flushes = collect_dirty_frames(*shard_ptr, std::numeric_limits<uint32_t>::max(), true);
        for (auto& flush_entry : flushes) {
            write_page(flush_entry.first, flush_entry.second.data());
            if (_stats) {
                _stats->pages_flushed.fetch_add(1);
                _stats->dirty_page_bytes_flushed.fetch_add(_page_size);
            }
            ShardState& shard = shard_for_page(flush_entry.first);
            std::lock_guard<std::mutex> lk(shard.mtx);
            auto it = shard.page_table.find(flush_entry.first);
            if (it != shard.page_table.end()) {
                _frames[it->second].flushing.store(false, std::memory_order_release);
            }
        }
    }
}

uint32_t BufferPool::dirty_page_count() const {
    uint32_t dirty_count = 0;
    for (const auto& f : _frames) {
        if (f.page_id != INVALID_PAGE && f.dirty.load(std::memory_order_acquire)) {
            dirty_count++;
        }
    }
    return dirty_count;
}

double BufferPool::dirty_ratio() const {
    if (_num_frames == 0) return 0.0;
    return static_cast<double>(dirty_page_count()) / static_cast<double>(_num_frames);
}

uint32_t BufferPool::allocate_page() {
    std::lock_guard<std::mutex> lk(_alloc_mtx);
    uint32_t pid = _next_page++;
    // extend the heap file
    if (::ftruncate(_heap_fd, (off_t)_next_page * _page_size) < 0) {
        // best effort
    }
    return pid;
}

std::vector<std::pair<uint32_t, std::vector<char>>> BufferPool::collect_dirty_frames(
    ShardState& shard, uint32_t max_frames, bool flush_all) {
    std::vector<std::pair<uint32_t, std::vector<char>>> flushes;
    std::lock_guard<std::mutex> lk(shard.mtx);
    if (shard.frame_begin >= shard.frame_end) return flushes;
    uint32_t shard_frames = shard.frame_end - shard.frame_begin;
    uint32_t scanned = 0;
    uint32_t budget = flush_all ? shard_frames : std::max<uint32_t>(1, std::min<uint32_t>(max_frames, shard_frames));
    while (scanned < shard_frames && flushes.size() < budget) {
        uint32_t idx = shard.flush_cursor;
        auto& f = _frames[idx];
        shard.flush_cursor++;
        if (shard.flush_cursor >= shard.frame_end) shard.flush_cursor = shard.frame_begin;
        scanned++;
        if (f.page_id == INVALID_PAGE) continue;
        if (f.loading.load(std::memory_order_acquire)) continue;
        if (f.flushing.load(std::memory_order_acquire)) continue;
        if (f.pin_count.load(std::memory_order_acquire) > 0) continue;
        if (!f.dirty.load(std::memory_order_acquire)) continue;
        std::vector<char> copy(f.data, f.data + _page_size);
        f.dirty.store(false, std::memory_order_release);
        f.flushing.store(true, std::memory_order_release);
        flushes.emplace_back(f.page_id, std::move(copy));
    }
    return flushes;
}

void BufferPool::bg_flush_loop(float high_wm, float low_wm) {
    while (_flush_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(_flush_wakeup_ms));
        uint32_t dirty_count = 0;
        for (uint32_t i = 0; i < _num_frames; ++i) {
            const auto& f = _frames[i];
            if (f.page_id != INVALID_PAGE && f.dirty.load(std::memory_order_acquire)) {
                dirty_count++;
            }
        }
        float ratio = (float)dirty_count / (float)(_num_frames ? _num_frames : 1);
        if (ratio <= high_wm) continue;
        for (auto& shard_ptr : _shards) {
            if (dirty_count <= static_cast<uint32_t>(low_wm * _num_frames)) break;
            auto flushes = collect_dirty_frames(*shard_ptr, _flush_budget_pages_per_cycle, false);
            for (auto& flush_entry : flushes) {
                write_page(flush_entry.first, flush_entry.second.data());
                if (_stats) {
                    _stats->pages_flushed.fetch_add(1);
                    _stats->dirty_page_bytes_flushed.fetch_add(_page_size);
                }
                dirty_count--;
                ShardState& shard = shard_for_page(flush_entry.first);
                std::lock_guard<std::mutex> lk(shard.mtx);
                auto it = shard.page_table.find(flush_entry.first);
                if (it != shard.page_table.end()) {
                    _frames[it->second].flushing.store(false, std::memory_order_release);
                }
            }
        }
    }
}

void BufferPool::start_bg_flush(float high_wm, float low_wm) {
    if (_flush_running.load()) return;
    _flush_running.store(true);
    _flush_thread = std::thread(&BufferPool::bg_flush_loop, this,
                                high_wm, low_wm);
}

void BufferPool::stop_bg_flush() {
    _flush_running.store(false);
    if (_flush_thread.joinable()) _flush_thread.join();
}

uint32_t BufferPool::pin_count_of(uint32_t page_id) const {
    const ShardState& shard = shard_for_page(page_id);
    std::lock_guard<std::mutex> lk(shard.mtx);
    auto it = shard.page_table.find(page_id);
    if (it != shard.page_table.end()) {
        return _frames[it->second].pin_count.load(std::memory_order_acquire);
    }
    return 0;  // not in buffer pool => not pinned
}

void BufferPool::reset() {
    stop_bg_flush();
    for (auto& f : _frames) {
        f.page_id = INVALID_PAGE;
        f.dirty.store(false, std::memory_order_release);
        f.loading.store(false, std::memory_order_release);
        f.flushing.store(false, std::memory_order_release);
        f.pin_count.store(0, std::memory_order_release);
        f.ref_bit.store(0, std::memory_order_release);
        f.protected_credit.store(0, std::memory_order_release);
        f.region = FrameRegion::SHARED;
    }
    for (auto& shard_ptr : _shards) {
        std::lock_guard<std::mutex> lk(shard_ptr->mtx);
        shard_ptr->page_table.clear();
        shard_ptr->clock_hand = shard_ptr->frame_begin;
        shard_ptr->flush_cursor = shard_ptr->frame_begin;
        shard_ptr->query_used = 0;
        shard_ptr->update_used = 0;
    }
}

// ===========================================================================
// Standalone PQ helpers (copied from pq_flash_index.cpp anonymous namespace)
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

// ===========================================================================
// InPlaceGraphStore
// ===========================================================================
InPlaceGraphStore::~InPlaceGraphStore() {
    stop_bg_flush();
    if (_pq_codes) {
        aligned_free(_pq_codes);
        _pq_codes = nullptr;
    }
    delete[] _node_active_flags;
    _node_active_flags = nullptr;
}

void InPlaceGraphStore::init(uint32_t dim, uint32_t Mmax,
                             uint32_t elem_size_bytes, uint32_t page_size,
                             uint32_t buffer_pool_frames,
                             const std::string& heap_path,
                             float bp_query_frac, float bp_update_frac,
                             uint32_t flush_budget_pages_per_cycle,
                             uint32_t flush_wakeup_ms,
                             bool truncate_heap) {
    _dim       = dim;
    _aligned_dim = (uint32_t)ROUND_UP(dim, 8);
    _Mmax      = Mmax;
    _elem_size = elem_size_bytes;
    _page_size = page_size;

    _slot_size = sizeof(PackedSlotHeader) +
                 _aligned_dim * _elem_size +
                 Mmax * sizeof(uint32_t);

    // Compute slots per page iteratively (page header + bitmap + slots)
    uint32_t avail = page_size - 8;  // page header is 8 bytes
    _slots_per_page = 0;
    for (uint32_t s = 1; s <= avail / _slot_size; s++) {
        uint32_t bmap = (s + 7) / 8;
        if (8 + bmap + s * _slot_size <= page_size) {
            _slots_per_page = s;
        }
    }
    if (_slots_per_page == 0) {
        throw ANNException("Slot size exceeds page size", -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    _bp.init(page_size, buffer_pool_frames, heap_path, &_stats,
             bp_query_frac, bp_update_frac,
             flush_budget_pages_per_cycle, flush_wakeup_ms, truncate_heap);
}

uint32_t InPlaceGraphStore::bitmap_bytes() const {
    return (_slots_per_page + 7) / 8;
}

uint32_t InPlaceGraphStore::header_bytes() const {
    return 8 + bitmap_bytes();
}

void InPlaceGraphStore::set_bitmap(char* page_data, uint16_t slot_idx) {
    uint8_t* bmap = (uint8_t*)(page_data + 8);
    bmap[slot_idx / 8] |= (1 << (slot_idx % 8));
}

void InPlaceGraphStore::clear_bitmap(char* page_data, uint16_t slot_idx) {
    uint8_t* bmap = (uint8_t*)(page_data + 8);
    bmap[slot_idx / 8] &= ~(1 << (slot_idx % 8));
}

bool InPlaceGraphStore::test_bitmap(const char* page_data,
                                    uint16_t slot_idx) const {
    const uint8_t* bmap = (const uint8_t*)(page_data + 8);
    return (bmap[slot_idx / 8] >> (slot_idx % 8)) & 1;
}

InPlaceGraphStore::RID InPlaceGraphStore::allocate_slot() {
    // Find a page with space
    if (!_pages_with_space.empty()) {
        uint32_t pid = *_pages_with_space.begin();
        auto& dir = _page_dir[pid];
        auto& frame = _bp.pin(pid, FrameRegion::UPDATE);
        char* page_data = frame.data;

        for (uint16_t s = 0; s < dir.slots_per_page; s++) {
            if (!test_bitmap(page_data, s)) {
                set_bitmap(page_data, s);
                dir.num_occupied++;
                if (dir.num_occupied >= dir.slots_per_page) {
                    _pages_with_space.erase(pid);
                }
                // Write updated header
                memcpy(page_data + 2, &dir.num_occupied, 2);
                _bp.unpin(pid, true);
                return RID{pid, s};
            }
        }
        _bp.unpin(pid);
    }

    // Allocate new page
    uint32_t pid = _bp.allocate_page();
    _page_dir.push_back({(uint16_t)_slots_per_page, 1});
    auto& frame = _bp.pin(pid, FrameRegion::UPDATE);
    char* page_data = frame.data;
    memset(page_data, 0, _page_size);

    // Write page header
    uint16_t spp = (uint16_t)_slots_per_page;
    uint16_t occ = 1;
    memcpy(page_data, &spp, 2);
    memcpy(page_data + 2, &occ, 2);

    set_bitmap(page_data, 0);
    _bp.unpin(pid, true);

    if (_slots_per_page > 1) {
        _pages_with_space.insert(pid);
    }
    return RID{pid, 0};
}

static void grow_active_flags(std::atomic<uint8_t>*& arr, uint32_t& cap,
                              uint32_t needed) {
    uint32_t new_cap = std::max(needed, cap * 2 + 1);
    auto* new_arr = new std::atomic<uint8_t>[new_cap];
    for (uint32_t i = 0; i < new_cap; i++) {
        new_arr[i].store(i < cap ? arr[i].load(std::memory_order_relaxed) : 0,
                         std::memory_order_relaxed);
    }
    delete[] arr;
    arr = new_arr;
    cap = new_cap;
}

void InPlaceGraphStore::allocate_node(uint32_t node_id) {
    std::lock_guard<std::mutex> lk(_alloc_mtx);
    auto allocate_one_locked = [&](uint32_t current_node_id) {
        if (current_node_id >= _node_to_rid.size()) {
            size_t new_sz = (size_t)current_node_id + 1;
            _node_to_rid.resize(new_sz, RID{INVALID_PAGE, 0});
        }
        if (current_node_id >= _node_active_cap) {
            grow_active_flags(_node_active_flags, _node_active_cap,
                              (uint32_t)(current_node_id + 1));
        }
        if (current_node_id >= _max_nodes) {
            uint32_t new_max = std::max((uint32_t)(current_node_id + 1),
                                        _max_nodes * 2 + 1);
            if (_n_chunks > 0) {
                uint8_t* new_codes = nullptr;
                alloc_aligned((void**)&new_codes,
                              (size_t)new_max * _n_chunks, 32);
                memset(new_codes, 0, (size_t)new_max * _n_chunks);
                if (_pq_codes && _max_nodes > 0) {
                    memcpy(new_codes, _pq_codes, (size_t)_max_nodes * _n_chunks);
                    aligned_free(_pq_codes);
                }
                _pq_codes = new_codes;
            }
            _max_nodes = new_max;
        }

        RID rid = allocate_slot();
        _node_to_rid[current_node_id] = rid;

        auto& frame = _bp.pin(rid.page_id, FrameRegion::UPDATE);
        char* slot_ptr = frame.data + header_bytes() + rid.slot_idx * _slot_size;
        PackedSlotHeader hdr;
        hdr.degree  = 0;
        hdr.flags   = 0;
        hdr._pad    = 0;
        hdr.node_id = current_node_id;
        memcpy(slot_ptr, &hdr, sizeof(hdr));
        _bp.unpin(rid.page_id, true);
    };
    allocate_one_locked(node_id);
}

void InPlaceGraphStore::allocate_nodes_batch(const uint32_t* node_ids, size_t count) {
    if (node_ids == nullptr || count == 0) return;
    std::lock_guard<std::mutex> lk(_alloc_mtx);
    auto allocate_one_locked = [&](uint32_t current_node_id) {
        if (current_node_id >= _node_to_rid.size()) {
            size_t new_sz = (size_t)current_node_id + 1;
            _node_to_rid.resize(new_sz, RID{INVALID_PAGE, 0});
        }
        if (current_node_id >= _node_active_cap) {
            grow_active_flags(_node_active_flags, _node_active_cap,
                              (uint32_t)(current_node_id + 1));
        }
        if (current_node_id >= _max_nodes) {
            uint32_t new_max = std::max((uint32_t)(current_node_id + 1),
                                        _max_nodes * 2 + 1);
            if (_n_chunks > 0) {
                uint8_t* new_codes = nullptr;
                alloc_aligned((void**)&new_codes,
                              (size_t)new_max * _n_chunks, 32);
                memset(new_codes, 0, (size_t)new_max * _n_chunks);
                if (_pq_codes && _max_nodes > 0) {
                    memcpy(new_codes, _pq_codes, (size_t)_max_nodes * _n_chunks);
                    aligned_free(_pq_codes);
                }
                _pq_codes = new_codes;
            }
            _max_nodes = new_max;
        }

        RID rid = allocate_slot();
        _node_to_rid[current_node_id] = rid;

        auto& frame = _bp.pin(rid.page_id, FrameRegion::UPDATE);
        char* slot_ptr = frame.data + header_bytes() + rid.slot_idx * _slot_size;
        PackedSlotHeader hdr;
        hdr.degree  = 0;
        hdr.flags   = 0;
        hdr._pad    = 0;
        hdr.node_id = current_node_id;
        memcpy(slot_ptr, &hdr, sizeof(hdr));
        _bp.unpin(rid.page_id, true);
    };
    for (size_t i = 0; i < count; ++i) {
        allocate_one_locked(node_ids[i]);
    }
}

void InPlaceGraphStore::publish_node(uint32_t node_id) {
    if (node_id < _node_active_cap) {
        uint8_t prev = _node_active_flags[node_id].exchange(
            1, std::memory_order_release);
        if (prev == 0) _num_active.fetch_add(1, std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> lk(_entry_mtx);
    if (_entry_candidates.size() < 4096) {
        _entry_candidates.push_back(node_id);
    } else if (!_entry_candidates.empty()) {
        _entry_candidates[_entry_rr_cursor % _entry_candidates.size()] = node_id;
        _entry_rr_cursor++;
    }
}

void InPlaceGraphStore::publish_nodes_batch(const uint32_t* node_ids, size_t count) {
    if (node_ids == nullptr || count == 0) return;
    for (size_t i = 0; i < count; ++i) {
        uint32_t node_id = node_ids[i];
        if (node_id < _node_active_cap) {
            uint8_t prev = _node_active_flags[node_id].exchange(
                1, std::memory_order_release);
            if (prev == 0) _num_active.fetch_add(1, std::memory_order_relaxed);
        }
    }
    std::lock_guard<std::mutex> lk(_entry_mtx);
    for (size_t i = 0; i < count; ++i) {
        uint32_t node_id = node_ids[i];
        if (_entry_candidates.size() < 4096) {
            _entry_candidates.push_back(node_id);
        } else if (!_entry_candidates.empty()) {
            _entry_candidates[_entry_rr_cursor % _entry_candidates.size()] = node_id;
            _entry_rr_cursor++;
        }
    }
}

NodeView InPlaceGraphStore::pin_node(uint32_t node_id, AccessMode mode) {
    NodeView view;
    if (node_id >= _node_to_rid.size()) return view;
    RID rid = _node_to_rid[node_id];
    if (rid.page_id == INVALID_PAGE) return view;

    FrameRegion hint = (mode == WRITE) ? FrameRegion::UPDATE
                                       : FrameRegion::QUERY;
    auto& frame = _bp.pin(rid.page_id, hint);
    char* slot_ptr = frame.data + header_bytes() + rid.slot_idx * _slot_size;

    PackedSlotHeader hdr;
    memcpy(&hdr, slot_ptr, sizeof(hdr));

    view.degree     = hdr.degree;
    view.flags      = hdr.flags;
    view.node_id    = hdr.node_id;
    view._slot_ptr  = slot_ptr;
    view.coords     = slot_ptr + sizeof(PackedSlotHeader);
    view.neighbors  = (uint32_t*)(view.coords + _aligned_dim * _elem_size);
    view._page_id   = rid.page_id;
    view._slot_idx  = rid.slot_idx;
    view._Mmax      = _Mmax;
    view._write_mode = (mode == WRITE);
    _stats.logical_bytes_read.fetch_add(
        sizeof(PackedSlotHeader) + (uint64_t)_aligned_dim * _elem_size +
        (uint64_t)hdr.degree * sizeof(uint32_t), std::memory_order_relaxed);
    return view;
}

NodeView InPlaceGraphStore::try_pin_node_if_resident(uint32_t node_id, AccessMode mode) {
    NodeView view;
    if (node_id >= _node_to_rid.size()) return view;
    RID rid = _node_to_rid[node_id];
    if (rid.page_id == INVALID_PAGE) return view;

    FrameRegion hint = (mode == WRITE) ? FrameRegion::UPDATE
                                       : FrameRegion::QUERY;
    auto* frame = _bp.try_pin_resident(rid.page_id, hint);
    if (frame == nullptr) return view;
    char* slot_ptr = frame->data + header_bytes() + rid.slot_idx * _slot_size;

    PackedSlotHeader hdr;
    memcpy(&hdr, slot_ptr, sizeof(hdr));

    view.degree      = hdr.degree;
    view.flags       = hdr.flags;
    view.node_id     = hdr.node_id;
    view._slot_ptr   = slot_ptr;
    view.coords      = slot_ptr + sizeof(PackedSlotHeader);
    view.neighbors   = (uint32_t*)(view.coords + _aligned_dim * _elem_size);
    view._page_id    = rid.page_id;
    view._slot_idx   = rid.slot_idx;
    view._Mmax       = _Mmax;
    view._write_mode = (mode == WRITE);
    _stats.logical_bytes_read.fetch_add(
        sizeof(PackedSlotHeader) + (uint64_t)_aligned_dim * _elem_size +
        (uint64_t)hdr.degree * sizeof(uint32_t), std::memory_order_relaxed);
    return view;
}

void InPlaceGraphStore::commit_node(NodeView& view) {
    if (!view._write_mode || view._page_id == INVALID_PAGE || view._slot_ptr == nullptr) return;
    PackedSlotHeader hdr;
    hdr.degree  = view.degree;
    hdr.flags   = view.flags;
    hdr._pad    = 0;
    hdr.node_id = view.node_id;
    memcpy(view._slot_ptr, &hdr, sizeof(hdr));
    _stats.logical_bytes_written.fetch_add(
        sizeof(PackedSlotHeader) + (uint64_t)_aligned_dim * _elem_size +
        (uint64_t)hdr.degree * sizeof(uint32_t), std::memory_order_relaxed);
}

void InPlaceGraphStore::unpin_node(NodeView& view) {
    if (view._page_id == INVALID_PAGE) return;
    _bp.unpin(view._page_id, view._write_mode);
    view._page_id = INVALID_PAGE;
    view._slot_ptr = nullptr;
}

void InPlaceGraphStore::protect_seed_pages(const std::vector<uint32_t>& node_ids, uint8_t credit) {
    if (credit == 0) return;
    tsl::robin_set<uint32_t> seen_pages;
    seen_pages.reserve(node_ids.size());
    for (uint32_t node_id : node_ids) {
        if (node_id >= _node_to_rid.size()) continue;
        RID rid = _node_to_rid[node_id];
        if (rid.page_id == INVALID_PAGE) continue;
        if (!seen_pages.insert(rid.page_id).second) continue;
        _bp.protect_page(rid.page_id, credit, FrameRegion::QUERY);
    }
}

void InPlaceGraphStore::batch_fetch_coords(const std::vector<uint32_t>& ids,
                                           char* out_coords,
                                           std::vector<uint8_t>& found,
                                           std::vector<uint8_t>* flags_out,
                                           bool resident_only) {
    found.assign(ids.size(), 0);
    if (flags_out) flags_out->assign(ids.size(), 0);
    if (ids.empty() || out_coords == nullptr) return;

    struct BatchItem {
        uint32_t page_id;
        uint16_t slot_idx;
        uint32_t node_id;
        size_t out_idx;
    };

    std::vector<BatchItem> items;
    items.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        uint32_t node_id = ids[i];
        if (node_id >= _node_to_rid.size()) continue;
        RID rid = _node_to_rid[node_id];
        if (rid.page_id == INVALID_PAGE) continue;
        items.push_back(BatchItem{rid.page_id, rid.slot_idx, node_id, i});
    }
    std::sort(items.begin(), items.end(), [](const BatchItem& a, const BatchItem& b) {
        if (a.page_id != b.page_id) return a.page_id < b.page_id;
        return a.slot_idx < b.slot_idx;
    });

    size_t cursor = 0;
    while (cursor < items.size()) {
        uint32_t page_id = items[cursor].page_id;
        PageFrame* frame_ptr = nullptr;
        if (resident_only) {
            frame_ptr = _bp.try_pin_resident(page_id, FrameRegion::QUERY);
        } else {
            frame_ptr = &_bp.pin(page_id, FrameRegion::QUERY);
        }
        if (frame_ptr == nullptr) {
            while (cursor < items.size() && items[cursor].page_id == page_id) {
                ++cursor;
            }
            continue;
        }

        while (cursor < items.size() && items[cursor].page_id == page_id) {
            const auto& item = items[cursor];
            char* slot_ptr = frame_ptr->data + header_bytes() + item.slot_idx * _slot_size;
            PackedSlotHeader hdr;
            memcpy(&hdr, slot_ptr, sizeof(hdr));
            if (hdr.node_id == item.node_id && !(hdr.flags & FLAG_DELETED)) {
                char* coords_ptr = slot_ptr + sizeof(PackedSlotHeader);
                memcpy(out_coords + item.out_idx * static_cast<size_t>(_aligned_dim) * _elem_size,
                       coords_ptr, static_cast<size_t>(_aligned_dim) * _elem_size);
                found[item.out_idx] = 1;
                if (flags_out) {
                    (*flags_out)[item.out_idx] = hdr.flags;
                }
                _stats.logical_bytes_read.fetch_add(
                    sizeof(PackedSlotHeader) + static_cast<uint64_t>(_aligned_dim) * _elem_size +
                    static_cast<uint64_t>(hdr.degree) * sizeof(uint32_t),
                    std::memory_order_relaxed);
            }
            ++cursor;
        }
        _bp.unpin(page_id, false);
    }
}

void InPlaceGraphStore::batch_fetch_frontier_neighbors(
    const std::vector<unsigned>& ids,
    std::vector<std::vector<unsigned>>& neighbors,
    std::vector<uint8_t>& found) {
    neighbors.clear();
    neighbors.resize(ids.size());
    found.assign(ids.size(), 0);
    if (ids.empty()) return;

    struct BatchItem {
        uint32_t page_id;
        uint16_t slot_idx;
        uint32_t node_id;
        size_t out_idx;
    };

    std::vector<BatchItem> items;
    items.reserve(ids.size());
    {
        std::lock_guard<std::mutex> lk(_alloc_mtx);
        for (size_t i = 0; i < ids.size(); ++i) {
            uint32_t node_id = ids[i];
            if (node_id >= _node_to_rid.size()) continue;
            RID rid = _node_to_rid[node_id];
            if (rid.page_id == INVALID_PAGE) continue;
            items.push_back(BatchItem{rid.page_id, rid.slot_idx, node_id, i});
        }
    }
    std::sort(items.begin(), items.end(), [](const BatchItem& a, const BatchItem& b) {
        if (a.page_id != b.page_id) return a.page_id < b.page_id;
        return a.slot_idx < b.slot_idx;
    });

    size_t cursor = 0;
    while (cursor < items.size()) {
        uint32_t page_id = items[cursor].page_id;
        auto& frame = _bp.pin(page_id, FrameRegion::QUERY);
        while (cursor < items.size() && items[cursor].page_id == page_id) {
            const auto& item = items[cursor];
            char* slot_ptr = frame.data + header_bytes() + item.slot_idx * _slot_size;
            PackedSlotHeader hdr;
            memcpy(&hdr, slot_ptr, sizeof(hdr));
            if (hdr.node_id == item.node_id && !(hdr.flags & FLAG_DELETED)) {
                uint32_t* nbrs_ptr = reinterpret_cast<uint32_t*>(
                    slot_ptr + sizeof(PackedSlotHeader) + _aligned_dim * _elem_size);
                auto& out = neighbors[item.out_idx];
                out.reserve(hdr.degree);
                for (uint16_t j = 0; j < hdr.degree; ++j) {
                    out.push_back(nbrs_ptr[j]);
                }
                found[item.out_idx] = 1;
                _stats.logical_bytes_read.fetch_add(
                    sizeof(PackedSlotHeader) + static_cast<uint64_t>(_aligned_dim) * _elem_size +
                    static_cast<uint64_t>(hdr.degree) * sizeof(uint32_t),
                    std::memory_order_relaxed);
            }
            ++cursor;
        }
        _bp.unpin(page_id, false);
    }
}

void InPlaceGraphStore::mark_deleted(uint32_t node_id) {
    auto view = pin_node(node_id, WRITE);
    if (view._page_id == INVALID_PAGE) return;
    view.flags |= FLAG_DELETED;
    commit_node(view);
    unpin_node(view);

    if (node_id < _node_active_cap) {
        uint8_t prev = _node_active_flags[node_id].exchange(
            0, std::memory_order_release);
        if (prev != 0) _num_active.fetch_sub(1, std::memory_order_relaxed);
    }
    {
        std::lock_guard<std::mutex> lk(_sweep_mtx);
        _new_deleted_nodes.insert(node_id);
    }
    clear_oversized_node(node_id);
    if (node_id == _entry_point) {
        uint32_t replacement = INVALID_NODE;
        {
            std::lock_guard<std::mutex> lk(_entry_mtx);
            if (!_entry_candidates.empty()) {
                for (size_t step = 0; step < _entry_candidates.size(); ++step) {
                    size_t idx = (_entry_rr_cursor + step) % _entry_candidates.size();
                    uint32_t cand = _entry_candidates[idx];
                    if (cand != node_id && is_active(cand)) {
                        replacement = cand;
                        _entry_rr_cursor = idx;
                        break;
                    }
                }
            }
        }
        if (replacement == INVALID_NODE) {
            auto deleted_view = pin_node(node_id, READ);
            if (deleted_view._page_id != INVALID_PAGE) {
                for (uint16_t i = 0; i < deleted_view.degree; ++i) {
                    uint32_t cand = deleted_view.neighbors[i];
                    if (cand != node_id && is_active(cand)) {
                        replacement = cand;
                        break;
                    }
                }
                unpin_node(deleted_view);
            }
        }
        if (replacement == INVALID_NODE) {
            for (uint32_t i = 0; i < _node_active_cap; ++i) {
                if (_node_active_flags[i].load(std::memory_order_acquire) != 0) {
                    replacement = i;
                    break;
                }
            }
        }
        if (replacement != INVALID_NODE) _entry_point = replacement;
    }
    _stats.tombstone_count.fetch_add(1);
}

bool InPlaceGraphStore::is_active(uint32_t node_id) const {
    if (node_id >= _node_active_cap) return false;
    return _node_active_flags[node_id].load(std::memory_order_acquire) != 0;
}

uint32_t InPlaceGraphStore::get_page_id(uint32_t node_id) const {
    if (node_id >= _node_to_rid.size()) return INVALID_PAGE;
    return _node_to_rid[node_id].page_id;
}

uint32_t InPlaceGraphStore::page_pin_count(uint32_t page_id) const {
    return _bp.pin_count_of(page_id);
}

uint32_t InPlaceGraphStore::num_active() const {
    return _num_active.load(std::memory_order_relaxed);
}

// PQ support
void InPlaceGraphStore::load_pq_from_disk_index(const std::string& pq_prefix,
                                                 uint32_t num_chunks) {
    std::string pivots_path = pq_prefix + "_pq_pivots.bin";
    _pq_table.load_pq_centroid_bin(pivots_path.c_str(), num_chunks);
    _n_chunks = num_chunks;

    std::string codes_path = pq_prefix + "_pq_compressed.bin";
    _u64 nr, nc;
    uint8_t* raw = nullptr;
    diskann::load_bin<uint8_t>(codes_path.c_str(), raw, nr, nc);

    uint32_t needed = std::max(_max_nodes, (uint32_t)nr);
    if (_pq_codes) aligned_free(_pq_codes);
    alloc_aligned((void**)&_pq_codes, (size_t)needed * _n_chunks, 32);
    memset(_pq_codes, 0, (size_t)needed * _n_chunks);
    memcpy(_pq_codes, raw, (size_t)nr * nc);
    _max_nodes = needed;
    delete[] raw;

    diskann::cout << "InPlace: loaded PQ codes, n=" << nr
                  << ", n_chunks=" << _n_chunks << std::endl;
}

void InPlaceGraphStore::compute_pq_dists_query(const unsigned* ids,
                                                uint64_t n_ids,
                                                const float* pq_dists_in,
                                                float* dists_out) {
    if (!_pq_codes || _n_chunks == 0) return;
    std::vector<uint8_t> scratch((size_t)n_ids * _n_chunks);
    aggregate_coords(ids, n_ids, _pq_codes, _n_chunks, scratch.data());
    pq_dist_lookup(scratch.data(), n_ids, _n_chunks, pq_dists_in, dists_out);
}

void InPlaceGraphStore::compute_pq_dists_src(uint32_t src,
                                              const unsigned* ids,
                                              float* dists_out,
                                              uint32_t count,
                                              uint8_t* scratch) {
    if (!_pq_codes || _n_chunks == 0) return;
    const uint8_t* src_ptr = _pq_codes + (uint64_t)src * _n_chunks;
    aggregate_coords(ids, count, _pq_codes, _n_chunks, scratch);
    _pq_table.compute_distances(src_ptr, scratch, dists_out, count);
}

void InPlaceGraphStore::encode_pq(uint32_t node_id, const float* coords) {
    if (!_pq_codes || _n_chunks == 0) return;
    if (node_id >= _max_nodes) return;
    _pq_table.deflate_vec(coords, _pq_codes + (uint64_t)node_id * _n_chunks);
}

void InPlaceGraphStore::defer_reverse_edge(uint32_t target, uint32_t src) {
    _deferred_edges.push(target, src);
    _stats.deferred_edges_pushed.fetch_add(1);
}

void InPlaceGraphStore::append_pending_reverse_neighbors(uint32_t target, std::vector<unsigned>& out) const {
    std::vector<uint32_t> pending;
    pending.reserve(8);
    _deferred_edges.snapshot(target, pending);
    for (uint32_t src : pending) {
        out.push_back(src);
    }
}

void InPlaceGraphStore::note_oversized_node(uint32_t node_id, uint32_t degree) {
    _oversized_nodes.note(node_id, degree, maintenance_epoch());
}

void InPlaceGraphStore::clear_oversized_node(uint32_t node_id) {
    _oversized_nodes.erase(node_id);
}

void InPlaceGraphStore::set_maintenance_epoch(uint32_t epoch) {
    _maintenance_epoch.store(epoch, std::memory_order_relaxed);
}

uint32_t InPlaceGraphStore::maintenance_epoch() const {
    return _maintenance_epoch.load(std::memory_order_relaxed);
}

bool InPlaceGraphStore::has_aged_oversized_nodes(uint32_t age_threshold_rounds) const {
    return _oversized_nodes.has_aged(maintenance_epoch(), age_threshold_rounds);
}

size_t InPlaceGraphStore::pending_oversized_nodes() const {
    return _oversized_nodes.size();
}

size_t InPlaceGraphStore::repair_queue_backlog() const {
    return _deferred_edges.aged_target_count();
}

uint32_t InPlaceGraphStore::dirty_page_count() const {
    return _bp.dirty_page_count();
}

void InPlaceGraphStore::reclaim_quarantined() {
    tsl::robin_set<uint32_t> quarantine_snapshot;
    tsl::robin_set<uint32_t> new_deleted_snapshot;
    {
        std::lock_guard<std::mutex> lk(_sweep_mtx);
        quarantine_snapshot = _quarantine_deleted_nodes;
        new_deleted_snapshot = _new_deleted_nodes;
        _new_deleted_nodes.clear();
    }
    tsl::robin_set<uint32_t> deferred;
    std::vector<uint32_t> reclaimed_pages;
    for (auto node_id : quarantine_snapshot) {
        RID rid{INVALID_PAGE, 0};
        {
            std::lock_guard<std::mutex> lk(_alloc_mtx);
            if (node_id >= _node_to_rid.size()) continue;
            rid = _node_to_rid[node_id];
        }
        if (rid.page_id == INVALID_PAGE) continue;

        if (_bp.pin_count_of(rid.page_id) > 0) {
            deferred.insert(node_id);
            continue;
        }

        auto& frame = _bp.pin(rid.page_id, FrameRegion::UPDATE);
        clear_bitmap(frame.data, rid.slot_idx);
        uint16_t occ = 0;
        {
            std::lock_guard<std::mutex> lk(_alloc_mtx);
            if (rid.page_id >= _page_dir.size()) {
                _bp.unpin(rid.page_id);
                continue;
            }
            auto& dir = _page_dir[rid.page_id];
            if (dir.num_occupied > 0) dir.num_occupied--;
            occ = dir.num_occupied;
            if (node_id < _node_to_rid.size() &&
                _node_to_rid[node_id].page_id == rid.page_id &&
                _node_to_rid[node_id].slot_idx == rid.slot_idx) {
                _node_to_rid[node_id].page_id = INVALID_PAGE;
            }
        }
        memcpy(frame.data + 2, &occ, 2);
        _bp.unpin(rid.page_id, true);
        reclaimed_pages.push_back(rid.page_id);
    }

    {
        std::lock_guard<std::mutex> lk(_alloc_mtx);
        for (uint32_t pid : reclaimed_pages) {
            _pages_with_space.insert(pid);
        }
    }
    {
        std::lock_guard<std::mutex> lk(_sweep_mtx);
        _quarantine_deleted_nodes = std::move(deferred);
        for (auto node_id : new_deleted_snapshot) {
            _quarantine_deleted_nodes.insert(node_id);
        }
        _sweep_generation++;
    }
}

template<typename T, typename TagT>
void InPlaceGraphStore::bulk_load_from_index(diskann::Index<T, TagT>& mem_index,
                                              uint32_t n) {
    const auto* graph = mem_index.get_graph();
    T*          data  = mem_index.get_data();

    for (uint32_t i = 0; i < n; i++) {
        allocate_node(i);
        auto view = pin_node(i, WRITE);
        if (view._page_id == INVALID_PAGE) continue;

        memcpy(view.coords, data + (size_t)_aligned_dim * i,
               (size_t)_aligned_dim * _elem_size);

        const auto& nbrs = (*graph)[i];
        view.degree = (uint16_t)std::min((size_t)_Mmax, nbrs.size());
        for (uint16_t j = 0; j < view.degree; j++) {
            view.neighbors[j] = nbrs[j];
        }
        commit_node(view);
        unpin_node(view);
        publish_node(i);
    }
    _entry_point = 0;

    diskann::cout << "InPlace: bulk loaded " << n << " nodes" << std::endl;
}

// explicit instantiations
template void InPlaceGraphStore::bulk_load_from_index<float, uint32_t>(
    diskann::Index<float, uint32_t>&, uint32_t);
template void InPlaceGraphStore::bulk_load_from_index<uint8_t, uint32_t>(
    diskann::Index<uint8_t, uint32_t>&, uint32_t);
template void InPlaceGraphStore::bulk_load_from_index<int8_t, uint32_t>(
    diskann::Index<int8_t, uint32_t>&, uint32_t);

void InPlaceGraphStore::warmup_bfs(uint32_t entry_point, uint32_t num_nodes) {
    if (num_nodes == 0) return;
    tsl::robin_set<uint32_t> visited;
    std::queue<uint32_t> q;
    q.push(entry_point);
    visited.insert(entry_point);
    uint32_t count = 0;
    while (!q.empty() && count < num_nodes) {
        uint32_t cur = q.front();
        q.pop();
        auto view = pin_node(cur, READ);
        if (view._page_id == INVALID_PAGE) continue;
        for (uint16_t i = 0; i < view.degree; i++) {
            uint32_t nbr = view.neighbors[i];
            if (visited.find(nbr) == visited.end()) {
                visited.insert(nbr);
                q.push(nbr);
            }
        }
        unpin_node(view);
        count++;
    }
    diskann::cout << "InPlace: warmed up " << count << " nodes" << std::endl;
}

void InPlaceGraphStore::start_bg_flush() { _bp.start_bg_flush(); }
void InPlaceGraphStore::stop_bg_flush()  { _bp.stop_bg_flush(); }
void InPlaceGraphStore::flush()          { _bp.flush_all_dirty(); }
double InPlaceGraphStore::dirty_ratio() const { return _bp.dirty_ratio(); }
void InPlaceGraphStore::set_entry_point(uint32_t entry_point) {
    if (is_active(entry_point)) _entry_point = entry_point;
}

void InPlaceGraphStore::save_snapshot(const std::string& meta_path) const {
    struct SnapshotHeader {
        char magic[8];
        uint32_t version;
        uint32_t dim;
        uint32_t aligned_dim;
        uint32_t Mmax;
        uint32_t elem_size;
        uint32_t page_size;
        uint32_t slot_size;
        uint32_t slots_per_page;
        uint32_t entry_point;
        uint32_t max_nodes;
        uint32_t n_chunks;
        uint32_t num_active;
        uint32_t total_pages;
        uint64_t rid_size;
        uint64_t active_cap;
        uint64_t page_dir_size;
        uint64_t pages_with_space_size;
        uint64_t entry_candidates_size;
        uint64_t entry_rr_cursor;
    } hdr{};

    memcpy(hdr.magic, "IPGSNP1", 8);
    hdr.version = 1;
    hdr.dim = _dim;
    hdr.aligned_dim = _aligned_dim;
    hdr.Mmax = _Mmax;
    hdr.elem_size = _elem_size;
    hdr.page_size = _page_size;
    hdr.slot_size = _slot_size;
    hdr.slots_per_page = _slots_per_page;
    hdr.entry_point = _entry_point;
    hdr.max_nodes = _max_nodes;
    hdr.n_chunks = _n_chunks;
    hdr.num_active = _num_active.load(std::memory_order_relaxed);
    hdr.total_pages = _bp.next_page();
    hdr.rid_size = _node_to_rid.size();
    hdr.active_cap = _node_active_cap;
    hdr.page_dir_size = _page_dir.size();
    hdr.pages_with_space_size = _pages_with_space.size();
    hdr.entry_candidates_size = _entry_candidates.size();
    hdr.entry_rr_cursor = _entry_rr_cursor;

    std::ofstream out(meta_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw ANNException("Failed to open snapshot file: " + meta_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    out.write(reinterpret_cast<const char*>(_node_to_rid.data()),
              static_cast<std::streamsize>(_node_to_rid.size() * sizeof(RID)));
    std::vector<uint8_t> active_flags(_node_active_cap, 0);
    for (uint32_t i = 0; i < _node_active_cap; ++i) {
        active_flags[i] = _node_active_flags[i].load(std::memory_order_relaxed);
    }
    out.write(reinterpret_cast<const char*>(active_flags.data()),
              static_cast<std::streamsize>(active_flags.size()));
    out.write(reinterpret_cast<const char*>(_page_dir.data()),
              static_cast<std::streamsize>(_page_dir.size() * sizeof(PageDir)));
    for (uint32_t pid : _pages_with_space) {
        out.write(reinterpret_cast<const char*>(&pid), sizeof(pid));
    }
    out.write(reinterpret_cast<const char*>(_entry_candidates.data()),
              static_cast<std::streamsize>(_entry_candidates.size() * sizeof(uint32_t)));
}

void InPlaceGraphStore::load_snapshot(const std::string& meta_path) {
    struct SnapshotHeader {
        char magic[8];
        uint32_t version;
        uint32_t dim;
        uint32_t aligned_dim;
        uint32_t Mmax;
        uint32_t elem_size;
        uint32_t page_size;
        uint32_t slot_size;
        uint32_t slots_per_page;
        uint32_t entry_point;
        uint32_t max_nodes;
        uint32_t n_chunks;
        uint32_t num_active;
        uint32_t total_pages;
        uint64_t rid_size;
        uint64_t active_cap;
        uint64_t page_dir_size;
        uint64_t pages_with_space_size;
        uint64_t entry_candidates_size;
        uint64_t entry_rr_cursor;
    } hdr{};

    std::ifstream in(meta_path, std::ios::binary);
    if (!in) {
        throw ANNException("Failed to open snapshot file: " + meta_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    if (!in || memcmp(hdr.magic, "IPGSNP1", 8) != 0 || hdr.version != 1) {
        throw ANNException("Invalid snapshot header: " + meta_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }
    if (hdr.dim != _dim || hdr.page_size != _page_size || hdr.Mmax != _Mmax ||
        hdr.elem_size != _elem_size || hdr.slot_size != _slot_size ||
        hdr.slots_per_page != _slots_per_page) {
        throw ANNException("Snapshot layout mismatch: " + meta_path, -1,
                           __FUNCSIG__, __FILE__, __LINE__);
    }

    _entry_point = hdr.entry_point;
    _max_nodes = std::max(_max_nodes, hdr.max_nodes);
    _n_chunks = hdr.n_chunks;
    _num_active.store(hdr.num_active, std::memory_order_relaxed);
    _node_to_rid.assign(static_cast<size_t>(hdr.rid_size), RID{INVALID_PAGE, 0});
    if (!_node_to_rid.empty()) {
        in.read(reinterpret_cast<char*>(_node_to_rid.data()),
                static_cast<std::streamsize>(_node_to_rid.size() * sizeof(RID)));
    }

    delete[] _node_active_flags;
    _node_active_flags = nullptr;
    _node_active_cap = static_cast<uint32_t>(hdr.active_cap);
    _node_active_flags = new std::atomic<uint8_t>[_node_active_cap];
    std::vector<uint8_t> active_flags(_node_active_cap, 0);
    if (!active_flags.empty()) {
        in.read(reinterpret_cast<char*>(active_flags.data()),
                static_cast<std::streamsize>(active_flags.size()));
    }
    for (uint32_t i = 0; i < _node_active_cap; ++i) {
        _node_active_flags[i].store(active_flags[i], std::memory_order_relaxed);
    }

    _page_dir.assign(static_cast<size_t>(hdr.page_dir_size), PageDir{0, 0});
    if (!_page_dir.empty()) {
        in.read(reinterpret_cast<char*>(_page_dir.data()),
                static_cast<std::streamsize>(_page_dir.size() * sizeof(PageDir)));
    }

    _pages_with_space.clear();
    for (uint64_t i = 0; i < hdr.pages_with_space_size; ++i) {
        uint32_t pid = 0;
        in.read(reinterpret_cast<char*>(&pid), sizeof(pid));
        _pages_with_space.insert(pid);
    }

    _entry_candidates.assign(static_cast<size_t>(hdr.entry_candidates_size), 0);
    if (!_entry_candidates.empty()) {
        in.read(reinterpret_cast<char*>(_entry_candidates.data()),
                static_cast<std::streamsize>(_entry_candidates.size() * sizeof(uint32_t)));
    }
    _entry_rr_cursor = static_cast<size_t>(hdr.entry_rr_cursor);
    _sweep_cursor = 0;
    _sweep_generation = 0;
    _new_deleted_nodes.clear();
    _quarantine_deleted_nodes.clear();
}

size_t InPlaceGraphStore::memory_usage_bytes() const {
    size_t rss = 0;
    std::ifstream f("/proc/self/statm");
    if (f.is_open()) {
        size_t dummy;
        f >> dummy >> rss;
        rss *= sysconf(_SC_PAGESIZE);
    }
    return rss;
}

size_t InPlaceGraphStore::buffer_pool_bytes() const {
    return (size_t)_bp.num_frames() * _bp.page_size();
}

size_t InPlaceGraphStore::locator_bytes() const {
    return _node_to_rid.capacity() * sizeof(RID) +
           (size_t)_node_active_cap * sizeof(std::atomic<uint8_t>);
}

size_t InPlaceGraphStore::deferred_edge_bytes() const {
    return _deferred_edges.size() * sizeof(DeferredEdge);
}

size_t InPlaceGraphStore::deferred_edge_target_count() const {
    return _deferred_edges.target_count();
}

size_t InPlaceGraphStore::pq_data_bytes() const {
    return (size_t)_max_nodes * _n_chunks;
}

}
}
