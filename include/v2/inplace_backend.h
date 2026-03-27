// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <atomic>
#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "distance.h"
#include "neighbor.h"
#include "pq_table.h"
#include "utils.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"

namespace diskann {
  template<typename T, typename TagT>
  class Index;
}

namespace diskann {
namespace inplace {

static constexpr uint32_t INVALID_PAGE = 0xFFFFFFFFu;
static constexpr uint32_t INVALID_NODE = 0xFFFFFFFFu;
static constexpr uint8_t  FLAG_DELETED = 0x01;

enum class DistanceScope : uint8_t { QUERY = 0, UPDATE = 1 };

// ---------------------------------------------------------------------------
// InPlaceIOStats
// ---------------------------------------------------------------------------
struct InPlaceIOStats {
    std::atomic<uint64_t> physical_bytes_written{0};
    std::atomic<uint64_t> physical_bytes_read{0};
    std::atomic<uint64_t> logical_bytes_written{0};
    std::atomic<uint64_t> logical_bytes_read{0};

    std::atomic<uint64_t> cache_hits{0};
    std::atomic<uint64_t> cache_misses{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> dirty_evictions{0};

    std::atomic<uint64_t> pages_flushed{0};
    std::atomic<uint64_t> dirty_page_bytes_flushed{0};

    std::atomic<uint64_t> total_inserts{0};
    std::atomic<uint64_t> total_deletes{0};

    std::atomic<uint64_t> nodes_repaired{0};
    std::atomic<uint64_t> tombstone_count{0};
    std::atomic<uint64_t> repair_queue_length{0};

    std::atomic<uint64_t> deferred_edges_pushed{0};
    std::atomic<uint64_t> deferred_edges_drained{0};

    std::atomic<uint64_t> query_region_hits{0};
    std::atomic<uint64_t> update_region_hits{0};
    std::atomic<uint64_t> shared_region_hits{0};
    std::atomic<uint64_t> cross_region_spills{0};

    std::atomic<uint64_t> query_distance_ns{0};
    std::atomic<uint64_t> update_distance_ns{0};
    std::atomic<uint64_t> total_distance_ns{0};
    std::atomic<uint64_t> query_distance_ops{0};
    std::atomic<uint64_t> update_distance_ops{0};
    std::atomic<uint64_t> total_distance_ops{0};

    void   reset();
    std::string to_json() const;
    void record_distance(DistanceScope scope, uint64_t ns, uint64_t ops) {
        total_distance_ns.fetch_add(ns, std::memory_order_relaxed);
        total_distance_ops.fetch_add(ops, std::memory_order_relaxed);
        if (scope == DistanceScope::QUERY) {
            query_distance_ns.fetch_add(ns, std::memory_order_relaxed);
            query_distance_ops.fetch_add(ops, std::memory_order_relaxed);
        } else {
            update_distance_ns.fetch_add(ns, std::memory_order_relaxed);
            update_distance_ops.fetch_add(ops, std::memory_order_relaxed);
        }
    }
};

// ---------------------------------------------------------------------------
// PackedSlotHeader -- 8 bytes, on-disk
// ---------------------------------------------------------------------------
struct PackedSlotHeader {
    uint16_t degree;
    uint8_t  flags;
    uint8_t  _pad;
    uint32_t node_id;
};
static_assert(sizeof(PackedSlotHeader) == 8, "slot header must be 8 bytes");

// ---------------------------------------------------------------------------
// AccessMode / NodeView
// ---------------------------------------------------------------------------
enum AccessMode : uint8_t { READ = 0, WRITE = 1 };

struct NodeView {
    uint16_t  degree      = 0;
    uint8_t   flags       = 0;
    uint32_t  node_id     = INVALID_NODE;
    uint32_t* neighbors   = nullptr;
    char*     coords      = nullptr;   // NOT 32-byte aligned!
    char*     _slot_ptr   = nullptr;
    uint32_t  _page_id    = INVALID_PAGE;
    uint16_t  _slot_idx   = 0;
    uint32_t  _Mmax       = 0;
    bool      _write_mode = false;
};

// ---------------------------------------------------------------------------
// InPlaceSearchScratch -- per-thread aligned buffers
// ---------------------------------------------------------------------------
struct InPlaceSearchScratch {
    char*    aligned_coord_scratch = nullptr;
    char*    aux_coord_scratch     = nullptr;
    uint64_t coord_idx = 0;

    float*   pq_dists         = nullptr;
    float*   dist_scratch     = nullptr;
    uint8_t* pq_coord_scratch = nullptr;
    uint64_t local_query_distance_ns = 0;
    uint64_t local_update_distance_ns = 0;
    uint64_t local_total_distance_ns = 0;
    uint64_t local_query_distance_ops = 0;
    uint64_t local_update_distance_ops = 0;
    uint64_t local_total_distance_ops = 0;

    static constexpr uint32_t MAX_SCRATCH_NODES = 16384;

    void init(uint32_t aligned_dim, uint32_t n_chunks, uint32_t elem_size);
    void reset() {
        coord_idx = 0;
        local_query_distance_ns = 0;
        local_update_distance_ns = 0;
        local_total_distance_ns = 0;
        local_query_distance_ops = 0;
        local_update_distance_ops = 0;
        local_total_distance_ops = 0;
    }
    inline void record_distance(DistanceScope scope, uint64_t ns, uint64_t ops) {
        local_total_distance_ns += ns;
        local_total_distance_ops += ops;
        if (scope == DistanceScope::QUERY) {
            local_query_distance_ns += ns;
            local_query_distance_ops += ops;
        } else {
            local_update_distance_ns += ns;
            local_update_distance_ops += ops;
        }
    }
    void flush_distance_stats(InPlaceIOStats& stats);
    ~InPlaceSearchScratch();
};

// ---------------------------------------------------------------------------
// DeferredEdgeBuffer
// ---------------------------------------------------------------------------
struct DeferredEdge {
    uint32_t target;
    uint32_t src;
};

struct PendingTargetState {
    std::vector<uint32_t> sources;
    uint8_t scan_age_bit = 0;
    bool selected = false;
    uint32_t last_append_epoch = 0;
};

class DeferredEdgeBuffer {
 public:
    DeferredEdgeBuffer();
    size_t push(uint32_t target, uint32_t src, uint32_t append_epoch = 0);
    size_t size() const;
    size_t target_count() const;
    size_t aged_target_count() const;
    void snapshot(uint32_t target, std::vector<uint32_t>& out) const;
    std::vector<uint32_t> select_targets_for_repair(uint32_t pending_threshold,
                                                    size_t limit = 0,
                                                    bool force_all = false);
    size_t finish_target(uint32_t target,
                         const std::vector<uint32_t>& remove_sources,
                         bool set_scan_age_bit);
 private:
    struct Shard {
        mutable std::mutex mtx;
        tsl::robin_map<uint32_t, PendingTargetState> target_to_state;
    };

    size_t shard_index(uint32_t target) const;

    static constexpr size_t kShardCount = 64;
    std::vector<std::unique_ptr<Shard>> _shards;
    std::atomic<size_t> _edge_count{0};
    std::atomic<size_t> _target_count{0};
    std::atomic<size_t> _aged_target_count{0};
};

struct OversizeNodeRecord {
    uint32_t first_epoch = 0;
    uint32_t last_epoch = 0;
    uint32_t last_degree = 0;
};

class OversizeNodeTable {
 public:
    OversizeNodeTable();
    void note(uint32_t node_id, uint32_t degree, uint32_t epoch);
    void erase(uint32_t node_id);
    size_t size() const;
    bool has_aged(uint32_t current_epoch, uint32_t age_threshold_rounds) const;
    std::vector<uint32_t> collect_candidates(uint32_t current_epoch,
                                             uint32_t age_threshold_rounds,
                                             size_t limit) const;
 private:
    struct Shard {
        mutable std::mutex mtx;
        tsl::robin_map<uint32_t, OversizeNodeRecord> nodes;
    };

    size_t shard_index(uint32_t node_id) const;

    static constexpr size_t kShardCount = 64;
    std::vector<std::unique_ptr<Shard>> _shards;
    std::atomic<size_t> _size{0};
};

// ---------------------------------------------------------------------------
// PageFrame + FrameRegion
// ---------------------------------------------------------------------------
enum class FrameRegion : uint8_t { QUERY = 0, UPDATE = 1, SHARED = 2 };

struct PageFrame {
    uint32_t    page_id   = INVALID_PAGE;
    char*       data      = nullptr;
    std::atomic<bool> dirty{false};
    std::atomic<bool> loading{false};
    std::atomic<bool> flushing{false};
    std::atomic<uint32_t> pin_count{0};
    std::atomic<uint8_t> ref_bit{0};
    std::atomic<uint8_t> protected_credit{0};
    FrameRegion region    = FrameRegion::SHARED;
};

// ---------------------------------------------------------------------------
// BufferPool -- CLOCK eviction, background flush, partitioned regions
// ---------------------------------------------------------------------------
class BufferPool {
 public:
    BufferPool() = default;
    ~BufferPool();

    void init(uint32_t page_size, uint32_t num_frames,
              const std::string& heap_path, InPlaceIOStats* stats,
              float region_query_frac  = 0.65f,
              float region_update_frac = 0.25f,
              uint32_t flush_budget_pages_per_cycle = 32,
              uint32_t flush_wakeup_ms = 100,
              bool truncate_heap = true);

    PageFrame& pin(uint32_t page_id,
                   FrameRegion hint = FrameRegion::QUERY);
    PageFrame*  try_pin_resident(uint32_t page_id,
                                 FrameRegion hint = FrameRegion::QUERY);
    void       unpin(uint32_t page_id, bool dirty = false);
    void       protect_page(uint32_t page_id, uint8_t credit,
                            FrameRegion hint = FrameRegion::QUERY);
    void       mark_dirty(uint32_t page_id);
    void       flush_all_dirty();
    uint32_t   allocate_page();

    void start_bg_flush(float high_wm = 0.70f, float low_wm = 0.30f);
    void stop_bg_flush();
    void reset();
    uint32_t dirty_page_count() const;
    double dirty_ratio() const;

    uint32_t page_size() const { return _page_size; }
    uint32_t num_frames() const { return _num_frames; }
    uint32_t next_page() const { return _next_page; }
    uint32_t pin_count_of(uint32_t page_id) const;

 private:
    struct FrameWaitState {
        std::mutex mtx;
        std::condition_variable cv;
    };

    struct ShardState {
        mutable std::mutex mtx;
        tsl::robin_map<uint32_t, uint32_t> page_table;
        uint32_t frame_begin = 0;
        uint32_t frame_end = 0;
        uint32_t clock_hand = 0;
        uint32_t flush_cursor = 0;
        uint32_t query_budget = 0;
        uint32_t update_budget = 0;
        uint32_t query_used = 0;
        uint32_t update_used = 0;
    };

    uint32_t shard_index(uint32_t page_id) const;
    ShardState& shard_for_page(uint32_t page_id);
    const ShardState& shard_for_page(uint32_t page_id) const;
    uint32_t evict_one(ShardState& shard, std::unique_lock<std::mutex>& lk, FrameRegion preferred);
    std::vector<std::pair<uint32_t, std::vector<char>>> collect_dirty_frames(ShardState& shard, uint32_t max_frames, bool flush_all);
    void     read_page(uint32_t page_id, char* buf);
    void     write_page(uint32_t page_id, const char* buf);
    void     bg_flush_loop(float high_wm, float low_wm);

    uint32_t         _page_size   = 0;
    uint32_t         _num_frames  = 0;
    std::deque<PageFrame> _frames;
    std::vector<std::unique_ptr<FrameWaitState>> _frame_waiters;
    std::vector<std::unique_ptr<ShardState>> _shards;
    int              _heap_fd     = -1;
    uint32_t         _next_page   = 0;
    mutable std::mutex _alloc_mtx;
    uint32_t         _num_shards = 1;
    uint32_t         _flush_budget_pages_per_cycle = 32;
    uint32_t         _flush_wakeup_ms = 100;

    std::thread         _flush_thread;
    std::atomic<bool>   _flush_running{false};
    InPlaceIOStats*     _stats = nullptr;
};

// ---------------------------------------------------------------------------
// InPlaceGraphStore
// ---------------------------------------------------------------------------
class InPlaceGraphStore {
 public:
    InPlaceGraphStore() = default;
    ~InPlaceGraphStore();

    void init(uint32_t dim, uint32_t Mmax, uint32_t elem_size_bytes,
              uint32_t page_size, uint32_t buffer_pool_frames,
              const std::string& heap_path,
              float bp_query_frac  = 0.65f,
              float bp_update_frac = 0.25f,
              uint32_t flush_budget_pages_per_cycle = 32,
              uint32_t flush_wakeup_ms = 100,
              bool truncate_heap = true);

    // PQ support
    void load_pq_from_disk_index(const std::string& pq_prefix,
                                 uint32_t num_chunks);
    void compute_pq_dists_query(const unsigned* ids, uint64_t n_ids,
                                const float* pq_dists, float* dists_out);
    void compute_pq_dists_src(uint32_t src, const unsigned* ids,
                              float* dists_out, uint32_t count,
                              uint8_t* scratch);
    const FixedChunkPQTable<float>& pq_table() const { return _pq_table; }
    FixedChunkPQTable<float>& pq_table_mut() { return _pq_table; }
    const uint8_t* pq_data() const { return _pq_codes; }
    uint32_t n_chunks() const { return _n_chunks; }

    // Pin/unpin zero-copy
    NodeView pin_node(uint32_t node_id, AccessMode mode);
    NodeView try_pin_node_if_resident(uint32_t node_id, AccessMode mode);
    void     unpin_node(NodeView& view);
    void     commit_node(NodeView& view);
    void     protect_seed_pages(const std::vector<uint32_t>& node_ids,
                                uint8_t credit = 2);
    void     batch_fetch_coords(const std::vector<uint32_t>& ids,
                                char* out_coords,
                                std::vector<uint8_t>& found,
                                std::vector<uint8_t>* flags_out = nullptr,
                                bool resident_only = false);
    void     batch_fetch_frontier_neighbors(const std::vector<unsigned>& ids,
                                            std::vector<std::vector<unsigned>>& neighbors,
                                            std::vector<uint8_t>& found);

    // Allocation
    void     allocate_node(uint32_t node_id);
    void     publish_node(uint32_t node_id);
    void     mark_deleted(uint32_t node_id);
    bool     is_active(uint32_t node_id) const;
    uint32_t get_page_id(uint32_t node_id) const;

    // Page-level pin count query for safe reclamation
    uint32_t page_pin_count(uint32_t page_id) const;

    // PQ encode for new inserts
    void encode_pq(uint32_t node_id, const float* coords);

    // Deferred reverse edges
    void defer_reverse_edge(uint32_t target, uint32_t src);
    void append_pending_reverse_neighbors(uint32_t target, std::vector<unsigned>& out) const;
    void note_oversized_node(uint32_t node_id, uint32_t degree);
    void clear_oversized_node(uint32_t node_id);
    void set_maintenance_epoch(uint32_t epoch);
    uint32_t maintenance_epoch() const;
    bool has_aged_oversized_nodes(uint32_t age_threshold_rounds) const;
    size_t pending_oversized_nodes() const;
    size_t repair_queue_backlog() const;
    uint32_t dirty_page_count() const;
    template<typename T>
    void drain_deferred_edges(unsigned R, unsigned C, float alpha,
                              unsigned slack,
                              unsigned aligned_dim,
                              uint32_t max_targets,
                              diskann::Distance<T>* dist);
    template<typename T>
    uint32_t prune_oversized_nodes(uint32_t budget, unsigned R, unsigned C,
                                   float alpha, unsigned aligned_dim,
                                   uint32_t age_threshold_rounds,
                                   diskann::Distance<T>* dist);

    // Sweep-based delete repair
    template<typename T>
    uint32_t sweep_repair_round(uint32_t budget, unsigned R, unsigned C,
                                float alpha, unsigned aligned_dim,
                                diskann::Distance<T>* dist);
    void reclaim_quarantined();

    // Bulk load
    template<typename T, typename TagT>
    void bulk_load_from_index(diskann::Index<T, TagT>& mem_index,
                              uint32_t n);

    // Warmup + lifecycle
    void warmup_bfs(uint32_t entry_point, uint32_t num_nodes);
    void start_bg_flush();
    void stop_bg_flush();
    void flush();
    double dirty_ratio() const;
    void set_entry_point(uint32_t entry_point);
    void save_snapshot(const std::string& meta_path) const;
    void load_snapshot(const std::string& meta_path);

    // Stats + info
    InPlaceIOStats& stats() { return _stats; }
    const InPlaceIOStats& stats() const { return _stats; }
    uint32_t entry_point() const { return _entry_point; }
    uint32_t num_active() const;
    uint32_t aligned_dim() const { return _aligned_dim; }
    uint32_t slot_size() const { return _slot_size; }
    uint32_t page_size() const { return _page_size; }
    uint32_t max_degree() const { return _Mmax; }
    uint32_t total_pages() const { return _bp.next_page(); }
    uint64_t sweep_cursor() const { return _sweep_cursor; }
    size_t   memory_usage_bytes() const;

    // Memory breakdown for metrics
    size_t buffer_pool_bytes() const;
    size_t locator_bytes() const;
    size_t deferred_edge_bytes() const;
    size_t deferred_edge_target_count() const;
    size_t pq_data_bytes() const;

 private:
    struct RID {
        uint32_t page_id;
        uint16_t slot_idx;
    };

    struct PageDir {
        uint16_t slots_per_page;
        uint16_t num_occupied;
    };

    RID allocate_slot();

    uint32_t _dim         = 0;
    uint32_t _aligned_dim = 0;
    uint32_t _Mmax        = 0;
    uint32_t _elem_size   = 0;
    uint32_t _page_size   = 0;
    uint32_t _slot_size   = 0;
    uint32_t _slots_per_page = 0;
    uint32_t _entry_point = 0;

    BufferPool     _bp;
    InPlaceIOStats _stats;
    DeferredEdgeBuffer _deferred_edges;

    // PQ
    FixedChunkPQTable<float> _pq_table;
    uint8_t* _pq_codes  = nullptr;
    uint32_t _n_chunks   = 0;
    uint32_t _max_nodes  = 0;

    // RID locator
    std::vector<RID>     _node_to_rid;
    // Lock-free active flags: allocated as raw array so std::atomic is usable
    std::atomic<uint8_t>* _node_active_flags = nullptr;
    uint32_t              _node_active_cap   = 0;
    std::atomic<uint32_t> _num_active{0};
    mutable std::mutex   _alloc_mtx;

    // Page directory
    std::vector<PageDir>  _page_dir;
    std::set<uint32_t>    _pages_with_space;

    // Sweep repair state
    uint32_t _sweep_cursor     = 0;
    uint64_t _sweep_generation = 0;
    std::atomic<uint32_t> _maintenance_epoch{0};
    std::mutex _sweep_mtx;
    tsl::robin_set<uint32_t> _new_deleted_nodes;
    tsl::robin_set<uint32_t> _quarantine_deleted_nodes;
    OversizeNodeTable _oversized_nodes;
    mutable std::mutex _entry_mtx;
    std::vector<uint32_t> _entry_candidates;
    size_t _entry_rr_cursor = 0;

    // Bitmap helpers
    uint32_t bitmap_bytes() const;
    uint32_t header_bytes() const;
    void     set_bitmap(char* page_data, uint16_t slot_idx);
    void     clear_bitmap(char* page_data, uint16_t slot_idx);
    bool     test_bitmap(const char* page_data, uint16_t slot_idx) const;
};

}  // namespace inplace
}  // namespace diskann
