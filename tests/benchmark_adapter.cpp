// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <omp.h>

#include "ann_exception.h"
#include "aux_utils.h"
#include "distance.h"
#include "index.h"
#include "neighbor.h"
#include "parameters.h"
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"
#include "pageann/inplace_backend.h"
#include "pageann/inplace_graph_ops.h"

using namespace diskann::inplace;

namespace {

static bool path_exists(const std::string &path);

static bool workload_has_query(const std::string &workload) {
  return workload == "query_only" || workload == "query_update_round" ||
         workload == "query_insert_round" || workload == "query_delete_round";
}

static bool workload_has_inserts(const std::string &workload) {
  return workload == "update_only" || workload == "query_update_round" ||
         workload == "query_insert_round";
}

static bool workload_has_deletes(const std::string &workload) {
  return workload == "update_only" || workload == "query_update_round" ||
         workload == "query_delete_round";
}

static uint64_t file_size_if_exists(const std::string &path) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) return 0;
  return static_cast<uint64_t>(st.st_size);
}

static uint64_t inplace_aux_index_artifact_bytes(const std::string &pq_prefix) {
  uint64_t total = 0;
  const std::vector<std::string> suffixes = {
      "_pq_compressed.bin",
      "_pq_pivots.bin",
      "_pq_pivots.bin_centroid.bin",
      "_pq_pivots.bin_rearrangement_perm.bin",
      "_pq_pivots.bin_chunk_offsets.bin",
  };
  for (const auto &suffix : suffixes) total += file_size_if_exists(pq_prefix + suffix);
  return total;
}

struct ProcIo {
  uint64_t read_bytes = 0;
  uint64_t write_bytes = 0;
};

struct ProcMem {
  double rss_kb = 0;
  double peak_rss_kb = 0;
};

struct NativeIOSnapshot {
  uint64_t physical_bytes_read = 0;
  uint64_t physical_bytes_written = 0;
  uint64_t logical_bytes_read = 0;
  uint64_t logical_bytes_written = 0;
  uint64_t pages_flushed = 0;
  uint64_t dirty_page_bytes_flushed = 0;
};

struct CacheSnapshot {
  uint64_t hits = 0;
  uint64_t misses = 0;
};

struct SearchMetrics {
  double qps = 0;
  double query_wall_time_s = 0;
  double lat_avg_us = 0;
  double lat_p50_us = 0;
  double lat_p95_us = 0;
  double lat_p99_us = 0;
  double recall = -1;
  double disk_ios = 0;
  double query_distance_us = 0;
  double query_distance_ops = 0;
  uint64_t query_cache_hits = 0;
  uint64_t query_cache_misses = 0;
  double query_cache_hit_rate = 0;
  uint32_t candidate_pool_L_effective = 0;
  uint32_t pq_frontier_confirm_topk_effective = 0;
};
struct BatchMetrics {
  double insert_throughput = 0;
  double delete_throughput = 0;
  double update_throughput = 0;
  double round_update_wall_time_s = 0;
};

template <typename T>
static void build_mem_index(const std::string &base_bin, uint32_t npts, uint32_t dim,
                            uint32_t aligned_dim, uint32_t R, uint32_t L,
                            uint32_t C, float alpha, uint32_t build_threads,
                            bool saturate_graph, diskann::Index<T, uint32_t> &mem_index) {
  (void)dim;
  (void)aligned_dim;
  diskann::Parameters params;
  params.Set<unsigned>("L", L);
  params.Set<unsigned>("R", R);
  params.Set<float>("alpha", alpha);
  params.Set<unsigned>("C", C);
  params.Set<unsigned>("num_threads", build_threads);
  params.Set<bool>("saturate_graph", saturate_graph);
  mem_index.build(base_bin.c_str(), npts, params);
}

static std::vector<unsigned> load_medoids_file(const std::string &path) {
  std::vector<unsigned> medoids;
  if (!path_exists(path)) return medoids;
  uint32_t *raw = nullptr;
  size_t nr = 0, nc = 0;
  diskann::load_bin<uint32_t>(path, raw, nr, nc);
  medoids.assign(raw, raw + nr * nc);
  delete[] raw;
  return medoids;
}

static void save_medoids_file(const std::string &path,
                              const std::vector<unsigned> &medoids) {
  if (medoids.empty()) return;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    throw std::runtime_error("failed to open medoids file: " + path);
  }
  int32_t rows = static_cast<int32_t>(medoids.size());
  int32_t cols = 1;
  out.write(reinterpret_cast<const char *>(&rows), sizeof(rows));
  out.write(reinterpret_cast<const char *>(&cols), sizeof(cols));
  out.write(reinterpret_cast<const char *>(medoids.data()),
            static_cast<std::streamsize>(medoids.size() * sizeof(uint32_t)));
}

template <typename T>
static std::vector<unsigned> build_disk_stream_index(
    const std::string &base_bin, const std::string &build_temp_root,
    uint32_t npts, uint32_t dim, uint32_t aligned_dim, uint32_t R, uint32_t L,
    uint32_t build_threads, float memory_budget_gb, InPlaceGraphStore &store) {
  std::string mkdir_cmd = "mkdir -p " + build_temp_root;
  if (std::system(mkdir_cmd.c_str()) != 0) {
    throw std::runtime_error("failed to create build temp root: " + build_temp_root);
  }
  const std::string prefix = build_temp_root + "/inplace_vamana";
  const std::string mem_index_path = prefix + "_mem.index";
  const std::string medoids_path = prefix + "_medoids.bin";
  const std::string centroids_path = prefix + "_centroids.bin";
  size_t training_set_size =
      diskann::PQ_TRAINING_SET_FRACTION * npts > diskann::MAX_PQ_TRAINING_SET_SIZE
          ? diskann::MAX_PQ_TRAINING_SET_SIZE
          : static_cast<size_t>(std::round(diskann::PQ_TRAINING_SET_FRACTION * npts));
  training_set_size = std::max<size_t>(1, training_set_size);
  const double sampling_rate =
      npts > 0 ? static_cast<double>(training_set_size) / static_cast<double>(npts) : 1.0;
  (void)build_threads;
  if (diskann::build_merged_vamana_index<T>(base_bin, diskann::Metric::L2, false,
                                            L, R, sampling_rate,
                                            memory_budget_gb, mem_index_path,
                                            medoids_path, centroids_path, nullptr) != 0) {
    throw std::runtime_error("build_merged_vamana_index failed for in-place disk_stream mode");
  }

  std::vector<unsigned> medoids = load_medoids_file(medoids_path);
  std::ifstream graph_in(mem_index_path, std::ios::binary);
  if (!graph_in) throw std::runtime_error("failed to open stream-built mem graph: " + mem_index_path);
  uint64_t expected_size = 0;
  uint32_t width = 0;
  uint32_t entry = 0;
  uint64_t frozen_pts = 0;
  graph_in.read(reinterpret_cast<char *>(&expected_size), sizeof(uint64_t));
  graph_in.read(reinterpret_cast<char *>(&width), sizeof(uint32_t));
  graph_in.read(reinterpret_cast<char *>(&entry), sizeof(uint32_t));
  graph_in.read(reinterpret_cast<char *>(&frozen_pts), sizeof(uint64_t));
  (void)expected_size;
  (void)width;
  (void)frozen_pts;

  std::ifstream data_in(base_bin, std::ios::binary);
  if (!data_in) throw std::runtime_error("failed to open base data for stream conversion: " + base_bin);
  int32_t file_npts = 0, file_dim = 0;
  data_in.read(reinterpret_cast<char *>(&file_npts), sizeof(int32_t));
  data_in.read(reinterpret_cast<char *>(&file_dim), sizeof(int32_t));
  if (static_cast<uint32_t>(file_dim) != dim) {
    throw std::runtime_error("stream conversion dim mismatch");
  }

  std::vector<T> packed(dim);
  std::vector<T> aligned(aligned_dim, static_cast<T>(0));
  for (uint32_t i = 0; i < npts; ++i) {
    uint32_t degree = 0;
    graph_in.read(reinterpret_cast<char *>(&degree), sizeof(uint32_t));
    std::vector<uint32_t> nbrs(degree);
    if (degree > 0) {
      graph_in.read(reinterpret_cast<char *>(nbrs.data()), degree * sizeof(uint32_t));
    }
    data_in.read(reinterpret_cast<char *>(packed.data()), dim * sizeof(T));
    if (!graph_in || !data_in) {
      throw std::runtime_error("unexpected EOF while streaming in-place build conversion");
    }
    std::fill(aligned.begin(), aligned.end(), static_cast<T>(0));
    std::memcpy(aligned.data(), packed.data(), dim * sizeof(T));
    store.allocate_node(i);
    auto view = store.pin_node(i, WRITE);
    if (view._page_id == INVALID_PAGE) {
      throw std::runtime_error("failed to pin newly allocated node during stream conversion");
    }
    std::memcpy(view.coords, aligned.data(), aligned_dim * sizeof(T));
    view.degree = static_cast<uint16_t>(std::min<size_t>(store.max_degree(), nbrs.size()));
    for (uint16_t j = 0; j < view.degree; ++j) view.neighbors[j] = nbrs[j];
    store.commit_node(view);
    store.unpin_node(view);
    store.publish_node(i);
  }
  store.set_entry_point(entry);
  if (medoids.empty() && store.is_active(entry)) medoids.push_back(entry);
  return medoids;
}

template <typename T>
static uint32_t choose_sampled_entry_point(const T *base_data, uint32_t npts, uint32_t aligned_dim) {
  if (npts == 0) return 0;
  uint32_t sample_count = std::min<uint32_t>(npts, 4096);
  uint32_t stride = std::max<uint32_t>(1, npts / sample_count);
  std::vector<double> centroid(aligned_dim, 0.0);
  uint32_t actual = 0;
  for (uint32_t i = 0; i < npts && actual < sample_count; i += stride, ++actual) {
    const T *vec = base_data + static_cast<size_t>(i) * aligned_dim;
    for (uint32_t d = 0; d < aligned_dim; ++d) {
      centroid[d] += static_cast<double>(vec[d]);
    }
  }
  if (actual == 0) return 0;
  for (double &x : centroid) x /= static_cast<double>(actual);

  double best_dist = std::numeric_limits<double>::max();
  uint32_t best_id = 0;
  actual = 0;
  for (uint32_t i = 0; i < npts && actual < sample_count; i += stride, ++actual) {
    const T *vec = base_data + static_cast<size_t>(i) * aligned_dim;
    double dist = 0.0;
    for (uint32_t d = 0; d < aligned_dim; ++d) {
      double diff = static_cast<double>(vec[d]) - centroid[d];
      dist += diff * diff;
    }
    if (dist < best_dist) {
      best_dist = dist;
      best_id = i;
    }
  }
  return best_id;
}

static std::vector<unsigned> gather_start_ids(InPlaceGraphStore &store, uint32_t beamwidth,
                                              const std::vector<unsigned> &medoids,
                                              const std::string &entry_init_mode,
                                              bool static_query_fast_path = false) {
  std::vector<unsigned> init_ids;
  init_ids.reserve(std::max<uint32_t>(1, beamwidth));
  auto add_seed = [&](uint32_t id) {
    if (init_ids.size() >= beamwidth) return;
    if (!static_query_fast_path && !store.is_active(id)) return;
    if (std::find(init_ids.begin(), init_ids.end(), id) == init_ids.end()) init_ids.push_back(id);
  };
  uint32_t entry = store.entry_point();
  if (entry_init_mode == "medoid" || entry_init_mode == "medoid_only" ||
      entry_init_mode == "medoid_plus_neighbors") {
    for (uint32_t medoid : medoids) {
      add_seed(medoid);
      if (init_ids.size() >= beamwidth) break;
    }
  }
  if (entry_init_mode == "entry_neighbors" || entry_init_mode == "medoid_plus_neighbors" || init_ids.empty()) {
    add_seed(entry);
  }
  if (beamwidth <= 1 || init_ids.empty()) return init_ids;

  size_t seed_cursor = 0;
  while (seed_cursor < init_ids.size() && init_ids.size() < beamwidth) {
    auto view = store.pin_node(init_ids[seed_cursor], READ);
    if (view._page_id != INVALID_PAGE) {
      for (uint16_t i = 0; i < view.degree && init_ids.size() < beamwidth; ++i) {
        add_seed(view.neighbors[i]);
      }
      store.unpin_node(view);
    }
    ++seed_cursor;
  }
  return init_ids;
}

template <typename T>
static std::vector<T> build_seed_coord_cache(InPlaceGraphStore &store,
                                             const std::vector<unsigned> &init_ids,
                                             uint32_t aligned_dim) {
  std::vector<T> cached;
  if (init_ids.empty()) return cached;
  cached.assign(static_cast<size_t>(init_ids.size()) * aligned_dim, static_cast<T>(0));
  std::vector<uint8_t> found;
  store.batch_fetch_coords(
      init_ids, reinterpret_cast<char *>(cached.data()), found, nullptr, true);
  std::vector<uint32_t> miss_ids;
  std::vector<size_t> miss_slots;
  miss_ids.reserve(init_ids.size());
  miss_slots.reserve(init_ids.size());
  for (size_t i = 0; i < init_ids.size(); ++i) {
    if (!found[i]) {
      miss_ids.push_back(init_ids[i]);
      miss_slots.push_back(i);
    }
  }
  if (!miss_ids.empty()) {
    std::vector<T> miss_buf(static_cast<size_t>(miss_ids.size()) * aligned_dim,
                            static_cast<T>(0));
    std::vector<uint8_t> miss_found;
    store.batch_fetch_coords(miss_ids, reinterpret_cast<char *>(miss_buf.data()),
                             miss_found, nullptr, false);
    for (size_t i = 0; i < miss_ids.size(); ++i) {
      if (!miss_found[i]) continue;
      std::memcpy(cached.data() + miss_slots[i] * aligned_dim,
                  miss_buf.data() + i * aligned_dim,
                  static_cast<size_t>(aligned_dim) * sizeof(T));
    }
  }
  return cached;
}

static double percentile_sorted(const std::vector<double> &v, double p) {
  if (v.empty()) return 0.0;
  size_t idx = static_cast<size_t>(p * static_cast<double>(v.size()));
  if (idx >= v.size()) idx = v.size() - 1;
  return v[idx];
}

static ProcMem current_mem_kb() {
  ProcMem mem;
  std::ifstream in("/proc/self/status");
  std::string key;
  while (in >> key) {
    if (key == "VmRSS:") {
      in >> mem.rss_kb;
    } else if (key == "VmHWM:") {
      in >> mem.peak_rss_kb;
    } else {
      std::string rest;
      std::getline(in, rest);
    }
  }
  return mem;
}

static ProcIo read_proc_io() {
  ProcIo io;
  std::ifstream in("/proc/self/io");
  std::string key;
  uint64_t value = 0;
  while (in >> key >> value) {
    if (key == "read_bytes:") io.read_bytes = value;
    if (key == "write_bytes:") io.write_bytes = value;
  }
  return io;
}

static NativeIOSnapshot read_native_io(const InPlaceGraphStore &store) {
  const auto &stats = store.stats();
  NativeIOSnapshot io;
  io.physical_bytes_read = stats.physical_bytes_read.load();
  io.physical_bytes_written = stats.physical_bytes_written.load();
  io.logical_bytes_read = stats.logical_bytes_read.load();
  io.logical_bytes_written = stats.logical_bytes_written.load();
  io.pages_flushed = stats.pages_flushed.load();
  io.dirty_page_bytes_flushed = stats.dirty_page_bytes_flushed.load();
  return io;
}

static CacheSnapshot read_query_cache(InPlaceGraphStore &store) {
  CacheSnapshot s;
  s.hits = store.stats().cache_hits.load();
  s.misses = store.stats().cache_misses.load();
  return s;
}

static int parse_schedule(const std::string &schedule) {
  if (schedule == "dynamic") return omp_sched_dynamic;
  return omp_sched_static;
}

struct ParallelErrorState {
  std::atomic<bool> failed{false};
  std::mutex mtx;
  std::string message;

  void capture(const diskann::ANNException &e) {
    if (!failed.exchange(true, std::memory_order_acq_rel)) {
      std::lock_guard<std::mutex> lk(mtx);
      message = std::string("ANNException: ") + e.message() +
                " (error_code=" + std::to_string(e.errorCode()) + ")";
    }
  }

  void capture(const std::exception &e) {
    if (!failed.exchange(true, std::memory_order_acq_rel)) {
      std::lock_guard<std::mutex> lk(mtx);
      message = std::string("std::exception: ") + e.what();
    }
  }

  void capture_unknown() {
    if (!failed.exchange(true, std::memory_order_acq_rel)) {
      std::lock_guard<std::mutex> lk(mtx);
      message = "unknown exception";
    }
  }

  void rethrow_if_failed(const std::string &prefix) const {
    if (failed.load(std::memory_order_acquire)) {
      throw std::runtime_error(prefix + ": " + message);
    }
  }
};

template <typename T>
static inline void maybe_encode_pq(InPlaceGraphStore &, uint32_t, const T *) {}

template <>
inline void maybe_encode_pq<float>(InPlaceGraphStore &store, uint32_t node_id,
                                   const float *coords) {
  if (store.n_chunks() > 0) {
    store.encode_pq(node_id, coords);
  }
}

static bool path_exists(const std::string &path) {
  return access(path.c_str(), F_OK) == 0;
}

template <typename T>
static void ensure_diskann_family_pq(const std::string &data_bin,
                                     const std::string &pq_prefix,
                                     uint32_t pq_chunks,
                                     double pq_sample_rate) {
  if (pq_chunks == 0 || pq_prefix.empty()) return;
  std::string pivots_path = pq_prefix + "_pq_pivots.bin";
  std::string codes_path = pq_prefix + "_pq_compressed.bin";
  if (path_exists(pivots_path) && path_exists(codes_path)) return;

  float *train_data = nullptr;
  size_t train_size = 0, train_dim = 0;
  gen_random_slice<T>(data_bin, pq_sample_rate, train_data, train_size, train_dim);
  if (train_data == nullptr || train_size == 0) {
    throw std::runtime_error("failed to prepare PQ training sample for: " + pq_prefix);
  }
  generate_pq_pivots(train_data, train_size, static_cast<unsigned>(train_dim),
                     NUM_PQ_CENTERS, pq_chunks, NUM_K_MEANS_ITERS, pivots_path);
  generate_pq_data_from_pivots<T>(data_bin, NUM_PQ_CENTERS, pq_chunks,
                                  pivots_path, codes_path);
  delete[] train_data;
}

static void read_trace_file(const std::string &path, std::vector<uint32_t> &delete_ids,
                            std::vector<uint32_t> &insert_ids) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open trace file: " + path);
  int32_t n = 0;
  in.read(reinterpret_cast<char *>(&n), sizeof(int32_t));
  delete_ids.resize(static_cast<size_t>(n));
  insert_ids.resize(static_cast<size_t>(n));
  in.read(reinterpret_cast<char *>(delete_ids.data()), sizeof(uint32_t) * delete_ids.size());
  in.read(reinterpret_cast<char *>(insert_ids.data()), sizeof(uint32_t) * insert_ids.size());
}

template <typename T>
static void load_selected_vectors(const std::string &data_path, const std::vector<uint32_t> &ids,
                                  std::vector<T> &out, size_t &dim, size_t &aligned_dim) {
  std::ifstream in(data_path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open data file: " + data_path);
  int32_t npts = 0, raw_dim = 0;
  in.read(reinterpret_cast<char *>(&npts), sizeof(int32_t));
  in.read(reinterpret_cast<char *>(&raw_dim), sizeof(int32_t));
  dim = static_cast<size_t>(raw_dim);
  aligned_dim = ROUND_UP(dim, 8);
  out.assign(ids.size() * aligned_dim, static_cast<T>(0));
  if (ids.empty()) return;
  size_t run_start = 0;
  while (run_start < ids.size()) {
    size_t run_end = run_start + 1;
    while (run_end < ids.size() && ids[run_end] == ids[run_end - 1] + 1) {
      ++run_end;
    }
    size_t run_len = run_end - run_start;
    uint64_t offset = 2ULL * sizeof(int32_t) + static_cast<uint64_t>(ids[run_start]) * dim * sizeof(T);
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    std::vector<T> packed(run_len * dim);
    in.read(reinterpret_cast<char *>(packed.data()), packed.size() * sizeof(T));
    for (size_t j = 0; j < run_len; ++j) {
      std::memcpy(out.data() + (run_start + j) * aligned_dim,
                  packed.data() + j * dim, dim * sizeof(T));
    }
    run_start = run_end;
  }
}

static uint32_t compute_sweep_budget(const std::string &mode, uint32_t configured_value,
                                     uint32_t round_work, uint32_t active_points) {
  uint32_t fallback = std::max<uint32_t>(1, round_work);
  if (mode == "pages") return configured_value > 0 ? configured_value : fallback;
  if (mode == "time_ms") return configured_value > 0 ? configured_value : fallback;
  if (mode == "nodes") {
    if (configured_value > 0) return configured_value;
    return std::min<uint32_t>(std::max<uint32_t>(1, round_work), std::max<uint32_t>(1, active_points));
  }
  return fallback;
}

static uint32_t internal_query_pool_L(uint32_t search_L, uint32_t recall_at,
                                      uint32_t beamwidth,
                                      uint32_t pq_confirm_topk,
                                      const std::string &query_pool_mode,
                                      uint32_t query_pool_slack) {
  uint32_t safety_floor = std::max<uint32_t>(1, std::max<uint32_t>(recall_at, beamwidth));
  if (query_pool_mode == "legacy") {
    uint32_t recall_floor = std::max<uint32_t>(recall_at * 4, recall_at + beamwidth);
    uint32_t confirm_floor = search_L + pq_confirm_topk + beamwidth;
    return std::max<uint32_t>(search_L, std::max<uint32_t>(recall_floor, confirm_floor));
  }
  if (query_pool_mode == "slack") {
    return std::max<uint32_t>(safety_floor, search_L + query_pool_slack);
  }
  return std::max<uint32_t>(safety_floor, search_L);
}

static uint32_t query_confirm_topk(uint32_t configured_topk, uint32_t recall_at,
                                   uint32_t beamwidth) {
  (void) recall_at;
  (void) beamwidth;
  return configured_topk;
}

template <typename T>
static SearchMetrics run_checkpoint_search(InPlaceGraphStore &store, const std::string &query_file,
                                           const std::string &truthset_file, uint32_t recall_at,
                                           uint32_t search_L, uint32_t aligned_dim,
                                           uint32_t beamwidth,
                                           const std::vector<unsigned> &medoids,
                                           const std::string &entry_init_mode,
                                           bool static_query_fast_path,
                                           uint32_t query_threads, int query_schedule,
                                           bool warmup_enabled, uint32_t warmup_query_count,
                                           uint32_t pq_confirm_topk,
                                           const std::string &pq_confirm_mode,
                                           double pq_confirm_margin,
                                           uint32_t pq_confirm_delta,
                                           const std::string &query_pool_mode,
                                           uint32_t query_pool_slack) {
  T *query = nullptr;
  size_t query_num = 0, query_dim = 0, query_aligned_dim = 0;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

  diskann::DistanceL2 dist_cmp_l2;
  diskann::Distance<T> *dist_cmp = reinterpret_cast<diskann::Distance<T> *>(&dist_cmp_l2);
  std::vector<uint32_t> result_tags(query_num * recall_at, 0);
  std::vector<double> latency_us(query_num, 0.0);
  uint64_t q_ns0 = store.stats().query_distance_ns.load();
  uint64_t q_ops0 = store.stats().query_distance_ops.load();
  uint64_t p_read0 = store.stats().physical_bytes_read.load();
  CacheSnapshot cache0 = read_query_cache(store);
  uint32_t candidate_pool_L = internal_query_pool_L(search_L, recall_at, beamwidth,
                                                    pq_confirm_topk,
                                                    query_pool_mode,
                                                    query_pool_slack);
  uint32_t confirm_topk = query_confirm_topk(pq_confirm_topk, recall_at, beamwidth);
  std::vector<unsigned> init_ids = gather_start_ids(store, beamwidth, medoids, entry_init_mode,
                                                    static_query_fast_path);
  if (init_ids.empty() && (static_query_fast_path || store.is_active(store.entry_point()))) {
    init_ids.push_back(static_cast<unsigned>(store.entry_point()));
  }
  store.protect_seed_pages(init_ids, static_query_fast_path ? 2 : 1);
  std::vector<T> seed_coord_cache =
      static_query_fast_path ? build_seed_coord_cache<T>(store, init_ids, aligned_dim)
                             : std::vector<T>();
  uint32_t pq_chunks = store.n_chunks();
  omp_set_schedule(static_cast<omp_sched_t>(query_schedule), 1);

  if (warmup_enabled) {
    ParallelErrorState warmup_errors;
#pragma omp parallel num_threads(query_threads)
    {
      InPlaceSearchScratch scratch;
      scratch.init(aligned_dim, pq_chunks, sizeof(T));
#pragma omp for schedule(runtime)
      for (int64_t i = 0; i < static_cast<int64_t>(std::min<size_t>(query_num, warmup_query_count)); ++i) {
        if (warmup_errors.failed.load(std::memory_order_acquire)) continue;
        try {
          std::vector<diskann::Neighbor> results;
          scratch.reset();
          graph_iterate_to_fixed_point<T>(query + i * query_aligned_dim, candidate_pool_L,
                                          init_ids, seed_coord_cache.empty() ? nullptr : seed_coord_cache.data(),
                                          beamwidth, static_query_fast_path,
                                          &store, aligned_dim, dist_cmp, &scratch, results,
                                          nullptr, DistanceScope::QUERY, confirm_topk,
                                          pq_confirm_mode, pq_confirm_margin, pq_confirm_delta);
          scratch.reset();
        } catch (const diskann::ANNException &e) {
          warmup_errors.capture(e);
        } catch (const std::exception &e) {
          warmup_errors.capture(e);
        } catch (...) {
          warmup_errors.capture_unknown();
        }
      }
    }
    warmup_errors.rethrow_if_failed("query warmup failed");
  }

  auto begin = std::chrono::high_resolution_clock::now();
  ParallelErrorState query_errors;
#pragma omp parallel num_threads(query_threads)
  {
    InPlaceSearchScratch scratch;
    scratch.init(aligned_dim, pq_chunks, sizeof(T));
#pragma omp for schedule(runtime)
    for (int64_t i = 0; i < static_cast<int64_t>(query_num); ++i) {
      if (query_errors.failed.load(std::memory_order_acquire)) continue;
      try {
        auto q0 = std::chrono::high_resolution_clock::now();
        std::vector<diskann::Neighbor> results;
        scratch.reset();
        graph_iterate_to_fixed_point<T>(query + i * query_aligned_dim, candidate_pool_L,
                                        init_ids, seed_coord_cache.empty() ? nullptr : seed_coord_cache.data(),
                                        beamwidth, static_query_fast_path,
                                        &store, aligned_dim, dist_cmp, &scratch, results,
                                        nullptr, DistanceScope::QUERY, confirm_topk,
                                        pq_confirm_mode, pq_confirm_margin, pq_confirm_delta);
        scratch.flush_distance_stats(store.stats());
        auto q1 = std::chrono::high_resolution_clock::now();
        latency_us[static_cast<size_t>(i)] = std::chrono::duration<double, std::micro>(q1 - q0).count();
        uint32_t written = 0;
        for (uint32_t k = 0; k < results.size() && written < recall_at; ++k) {
          if (!static_query_fast_path && !store.is_active(results[k].id)) continue;
          result_tags[static_cast<size_t>(i) * recall_at + written] = results[k].id;
          ++written;
        }
      } catch (const diskann::ANNException &e) {
        query_errors.capture(e);
      } catch (const std::exception &e) {
        query_errors.capture(e);
      } catch (...) {
        query_errors.capture_unknown();
      }
    }
  }
  query_errors.rethrow_if_failed("checkpoint query failed");
  auto end = std::chrono::high_resolution_clock::now();
  std::sort(latency_us.begin(), latency_us.end());

  SearchMetrics m;
  double elapsed_s = std::chrono::duration<double>(end - begin).count();
  m.query_wall_time_s = elapsed_s;
  double total_lat = std::accumulate(latency_us.begin(), latency_us.end(), 0.0);
  m.qps = elapsed_s > 0 ? static_cast<double>(query_num) / elapsed_s : 0.0;
  m.lat_avg_us = query_num > 0 ? total_lat / static_cast<double>(query_num) : 0.0;
  m.lat_p50_us = percentile_sorted(latency_us, 0.50);
  m.lat_p95_us = percentile_sorted(latency_us, 0.95);
  m.lat_p99_us = percentile_sorted(latency_us, 0.99);
  uint64_t q_ns1 = store.stats().query_distance_ns.load();
  uint64_t q_ops1 = store.stats().query_distance_ops.load();
  m.query_distance_us = static_cast<double>(q_ns1 - q_ns0) / 1000.0;
  m.query_distance_ops = static_cast<double>(q_ops1 - q_ops0);
  CacheSnapshot cache1 = read_query_cache(store);
  m.query_cache_hits = cache1.hits - cache0.hits;
  m.query_cache_misses = cache1.misses - cache0.misses;
  uint64_t cache_accesses = m.query_cache_hits + m.query_cache_misses;
  m.query_cache_hit_rate = cache_accesses > 0 ? (100.0 * static_cast<double>(m.query_cache_hits) /
                                                 static_cast<double>(cache_accesses))
                                              : 0.0;
  m.candidate_pool_L_effective = candidate_pool_L;
  m.pq_frontier_confirm_topk_effective = confirm_topk;
  uint64_t p_read1 = store.stats().physical_bytes_read.load();
  uint64_t pages_read = store.page_size() > 0 ? (p_read1 - p_read0) / store.page_size() : 0;
  m.disk_ios = query_num > 0 ? static_cast<double>(pages_read) / static_cast<double>(query_num) : 0.0;

  if (!truthset_file.empty() && path_exists(truthset_file)) {
    unsigned *gt_ids = nullptr;
    float *gt_dists = nullptr;
    unsigned *gt_tags = nullptr;
    size_t gt_num = 0, gt_dim = 0;
    diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim, &gt_tags);
    m.recall = diskann::calculate_recall(static_cast<unsigned>(query_num), gt_ids, gt_dists,
                                         static_cast<unsigned>(gt_dim), result_tags.data(),
                                         recall_at, recall_at);
    delete[] gt_ids;
    delete[] gt_dists;
    delete[] gt_tags;
  }

  diskann::aligned_free(query);
  return m;
}

static std::string json_escape(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

static void append_row(std::ofstream &out, const std::string &workload,
                       uint32_t checkpoint_id, uint32_t round_index,
                       uint32_t base_points, uint32_t active_points,
                       uint32_t cumulative_inserts, uint32_t cumulative_deletes,
                       uint32_t recall_at, uint32_t search_L, uint32_t beamwidth,
                       uint32_t query_threads, uint32_t insert_threads, uint32_t delete_threads,
                       uint32_t buffer_pool_frames, uint32_t page_size,
                       uint32_t insert_search_L, uint32_t prune_R, uint32_t prune_C,
                       const SearchMetrics &search, const BatchMetrics &batch,
                       uint64_t disk_size_bytes, uint64_t aux_size_bytes,
                       const ProcIo &io_start, const ProcIo &io_prev, const ProcIo &io_cur,
                       const NativeIOSnapshot &native_start,
                       const NativeIOSnapshot &native_prev,
                       const NativeIOSnapshot &native_cur,
                       const ProcMem &mem, const std::string &maintenance_policy,
                       bool maintenance_triggered, double foreground_update_wall_time_s,
                       double maintenance_wall_time_s, uint32_t pending_updates,
                       const InPlaceGraphStore &store,
                       const std::string &entry_init_mode,
                       const std::string &query_pool_mode,
                       uint32_t query_pool_slack,
                       const std::string &notes) {
  constexpr double kProcPageSize = 4096.0;
  double checkpoint_pages_read =
      static_cast<double>(io_cur.read_bytes - io_prev.read_bytes) / kProcPageSize;
  out << std::fixed << std::setprecision(4);
  out << "{";
  out << "\"system\":\"pageann\",";
  out << "\"workload\":\"" << workload << "\",";
  out << "\"status\":\"ok\",";
  out << "\"checkpoint_id\":" << checkpoint_id << ",";
  out << "\"round_index\":" << round_index << ",";
  out << "\"base_points\":" << base_points << ",";
  out << "\"active_points\":" << active_points << ",";
  out << "\"cumulative_inserts\":" << cumulative_inserts << ",";
  out << "\"cumulative_deletes\":" << cumulative_deletes << ",";
  out << "\"recall_at\":" << recall_at << ",";
  out << "\"search_L\":" << search_L << ",";
  out << "\"beamwidth\":" << beamwidth << ",";
  out << "\"query_threads\":" << query_threads << ",";
  out << "\"insert_threads\":" << insert_threads << ",";
  out << "\"delete_threads\":" << delete_threads << ",";
  out << "\"buffer_pool_frames\":" << buffer_pool_frames << ",";
  out << "\"page_size\":" << page_size << ",";
  out << "\"insert_search_L\":" << insert_search_L << ",";
  out << "\"prune_R\":" << prune_R << ",";
  out << "\"prune_C\":" << prune_C << ",";
  out << "\"entry_init_mode\":\"" << json_escape(entry_init_mode) << "\",";
  out << "\"query_pool_mode\":\"" << json_escape(query_pool_mode) << "\",";
  out << "\"query_pool_slack\":" << query_pool_slack << ",";
  out << "\"candidate_pool_L_effective\":" << search.candidate_pool_L_effective << ",";
  out << "\"pq_frontier_confirm_topk_effective\":" << search.pq_frontier_confirm_topk_effective << ",";
  out << "\"qps\":" << search.qps << ",";
  out << "\"query_throughput\":" << search.qps << ",";
  out << "\"insert_throughput\":" << batch.insert_throughput << ",";
  out << "\"delete_throughput\":" << batch.delete_throughput << ",";
  out << "\"update_throughput\":" << batch.update_throughput << ",";
  out << "\"query_wall_time_s\":" << search.query_wall_time_s << ",";
  out << "\"foreground_update_wall_time_s\":" << foreground_update_wall_time_s << ",";
  out << "\"maintenance_wall_time_s\":" << maintenance_wall_time_s << ",";
  out << "\"round_update_wall_time_s\":" << batch.round_update_wall_time_s << ",";
  out << "\"lat_avg_us\":" << search.lat_avg_us << ",";
  out << "\"lat_p50_us\":" << search.lat_p50_us << ",";
  out << "\"lat_p95_us\":" << search.lat_p95_us << ",";
  out << "\"lat_p99_us\":" << search.lat_p99_us << ",";
  out << "\"recall\":" << search.recall << ",";
  out << "\"disk_ios\":" << search.disk_ios << ",";
  out << "\"disk_ios_native\":" << search.disk_ios << ",";
  out << "\"query_io_us_native\":\"\",";
  out << "\"proc_read_pages\":" << checkpoint_pages_read << ",";
  out << "\"rss_kb\":" << mem.rss_kb << ",";
  out << "\"peak_rss_kb\":" << mem.peak_rss_kb << ",";
  out << "\"disk_index_size_bytes\":" << disk_size_bytes << ",";
  out << "\"aux_index_artifact_bytes\":" << aux_size_bytes << ",";
  out << "\"index_num_nodes\":" << active_points << ",";
  out << "\"native_index_bytes_read\":"
      << (native_cur.physical_bytes_read - native_prev.physical_bytes_read) << ",";
  out << "\"native_index_bytes_written\":"
      << (native_cur.physical_bytes_written - native_prev.physical_bytes_written) << ",";
  out << "\"cumulative_native_index_bytes_read\":"
      << (native_cur.physical_bytes_read - native_start.physical_bytes_read) << ",";
  out << "\"cumulative_native_index_bytes_written\":"
      << (native_cur.physical_bytes_written - native_start.physical_bytes_written) << ",";
  out << "\"native_pages_flushed\":"
      << (native_cur.pages_flushed - native_prev.pages_flushed) << ",";
  out << "\"cumulative_native_pages_flushed\":"
      << (native_cur.pages_flushed - native_start.pages_flushed) << ",";
  uint64_t logical_write_delta =
      native_cur.logical_bytes_written - native_prev.logical_bytes_written;
  uint64_t dirty_write_delta =
      native_cur.dirty_page_bytes_flushed - native_prev.dirty_page_bytes_flushed;
  double interval_write_amp =
      logical_write_delta > 0 ? static_cast<double>(dirty_write_delta) /
                                    static_cast<double>(logical_write_delta)
                              : 0.0;
  out << "\"native_write_amplification\":" << interval_write_amp << ",";
  out << "\"checkpoint_bytes_read\":" << (io_cur.read_bytes - io_prev.read_bytes) << ",";
  out << "\"checkpoint_bytes_written\":" << (io_cur.write_bytes - io_prev.write_bytes) << ",";
  out << "\"cumulative_bytes_read\":" << (io_cur.read_bytes - io_start.read_bytes) << ",";
  out << "\"cumulative_bytes_written\":" << (io_cur.write_bytes - io_start.write_bytes) << ",";
  out << "\"query_cache_hits\":" << search.query_cache_hits << ",";
  out << "\"query_cache_misses\":" << search.query_cache_misses << ",";
  out << "\"query_cache_hit_rate\":" << search.query_cache_hit_rate << ",";
  out << "\"query_distance_us\":" << search.query_distance_us << ",";
  out << "\"query_distance_ops\":" << search.query_distance_ops << ",";
  out << "\"maintenance_policy\":\"" << json_escape(maintenance_policy) << "\",";
  out << "\"maintenance_triggered\":" << (maintenance_triggered ? 1 : 0) << ",";
  out << "\"pending_updates\":" << pending_updates << ",";
  out << "\"overlay_pending_edges\":" << (store.deferred_edge_bytes() / sizeof(DeferredEdge)) << ",";
  out << "\"overlay_pending_targets\":" << store.deferred_edge_target_count() << ",";
  out << "\"repair_queue_backlog\":" << store.repair_queue_backlog() << ",";
  out << "\"oversized_node_backlog\":" << store.pending_oversized_nodes() << ",";
  out << "\"dirty_page_count\":" << store.dirty_page_count() << ",";
  out << "\"tombstone_count\":" << store.stats().tombstone_count.load(std::memory_order_relaxed) << ",";
  out << "\"unsupported_reason\":\"\",";
  out << "\"notes\":\"" << json_escape(notes) << "\"";
  out << "}\n";
  out.flush();
}

template <typename T>
static int run_fair(const std::string &workload, const std::string &base_bin,
                    const std::string &full_bin, const std::string &query_bin,
                    const std::string &trace_prefix, const std::string &gt_prefix,
                    const std::string &pq_prefix, uint32_t pq_chunks, double pq_sample_rate,
                    const std::string &heap_path, const std::string &result_jsonl,
                    uint32_t base_points, uint32_t update_rounds, uint32_t recall_at,
                    uint32_t search_L, uint32_t beamwidth,
                    uint32_t query_threads, uint32_t insert_threads, uint32_t delete_threads,
                    int query_schedule, int update_schedule,
                    bool warmup_enabled, uint32_t warmup_query_count,
                    bool warmup_before_each_checkpoint,
                    uint32_t buffer_pool_frames, uint32_t page_size,
                    uint32_t build_R, uint32_t build_L, uint32_t build_C, float build_alpha,
                    uint32_t build_threads, bool saturate_graph,
                    uint32_t insert_search_L, uint32_t prune_R, uint32_t prune_C, float prune_alpha,
                    float bp_query_frac, float bp_update_frac,
                    bool flush_after_build_before_measurement,
                    bool drain_every_round, bool sweep_every_round,
                    const std::string &sweep_budget_mode, uint32_t sweep_budget_value,
                    bool flush_before_checkpoint, uint32_t deferred_edge_high_water_mark,
                    uint32_t drain_high_water_mark, uint32_t oversize_slack,
                    uint32_t oversize_prune_high_water_mark,
                    uint32_t sweep_round_period,
                    uint32_t sweep_deleted_threshold,
                    uint32_t flush_round_period,
                    double flush_dirty_ratio_threshold,
                    uint32_t pq_frontier_confirm_topk,
                    const std::string &pq_confirm_mode,
                    double pq_confirm_margin,
                    uint32_t pq_confirm_delta,
                    const std::string &query_pool_mode,
                    uint32_t query_pool_slack,
                    uint32_t flush_budget_pages_per_cycle,
                    uint32_t flush_wakeup_ms,
                    const std::string &build_mode,
                    float build_memory_budget_gb,
                    const std::string &build_temp_root,
                    const std::string &entry_init_mode,
                    uint32_t query_only_checkpoints,
                    bool reuse_existing_index) {
  size_t base_npts = 0, dim = 0;
  diskann::get_bin_metadata(base_bin, base_npts, dim);
  uint32_t aligned_dim = static_cast<uint32_t>(ROUND_UP(dim, 8));
  uint32_t reverse_edge_overlay_high_water_mark = deferred_edge_high_water_mark;
  uint32_t reverse_edge_overlay_target_high_water_mark = drain_high_water_mark;
  uint32_t reverse_edge_pending_threshold = std::max<uint32_t>(1, oversize_slack);
  uint32_t tombstone_sweep_high_water_mark = sweep_deleted_threshold;
  uint32_t flush_dirty_page_high_water_mark = flush_round_period;
  uint32_t store_degree_cap = std::max(build_R, prune_R);
  std::string heap_meta_path = heap_path + ".meta";
  std::string medoids_path = heap_path + ".medoids";

  InPlaceGraphStore store;
  store.init(static_cast<uint32_t>(dim), store_degree_cap, sizeof(T), page_size, buffer_pool_frames,
             heap_path,
             bp_query_frac, bp_update_frac,
             flush_budget_pages_per_cycle, flush_wakeup_ms,
             !reuse_existing_index);
  ensure_diskann_family_pq<T>(full_bin, pq_prefix, pq_chunks, pq_sample_rate);
  if (pq_chunks > 0) {
    store.load_pq_from_disk_index(pq_prefix, pq_chunks);
  }
  std::vector<unsigned> medoids;
  if (reuse_existing_index) {
    store.load_snapshot(heap_meta_path);
    medoids = load_medoids_file(medoids_path);
  } else {
    if (build_mode == "disk_stream") {
      medoids = build_disk_stream_index<T>(base_bin, build_temp_root, base_points,
                                           static_cast<uint32_t>(dim), aligned_dim,
                                           build_R, build_L, build_threads,
                                           build_memory_budget_gb, store);
    } else {
      T *base_data = nullptr;
      size_t base_aligned_dim = 0;
      diskann::load_aligned_bin<T>(base_bin, base_data, base_npts, dim, base_aligned_dim);
      diskann::Index<T, uint32_t> mem_index(diskann::Metric::L2, dim, base_points + 1024,
                                            false, false, false);
      build_mem_index(base_bin, base_points, static_cast<uint32_t>(dim), aligned_dim,
                      build_R, build_L, build_C, build_alpha, build_threads,
                      saturate_graph, mem_index);
      store.bulk_load_from_index(mem_index, base_points);
      uint32_t sampled_entry = choose_sampled_entry_point(base_data, base_points, aligned_dim);
      store.set_entry_point(sampled_entry);
      medoids.push_back(sampled_entry);
      diskann::aligned_free(base_data);
    }
    store.flush();
    store.save_snapshot(heap_meta_path);
    save_medoids_file(medoids_path, medoids);
  }
  if (flush_after_build_before_measurement) {
    store.flush();
  }
  bool bg_flush_running = false;
  if (workload != "query_only") {
    store.start_bg_flush();
    bg_flush_running = true;
  }

  std::ofstream out(result_jsonl, std::ios::out | std::ios::trunc);
  ProcIo io_start = read_proc_io();
  ProcIo io_prev = io_start;
  NativeIOSnapshot native_start = read_native_io(store);
  NativeIOSnapshot native_prev = native_start;
  uint64_t aux_size_bytes = inplace_aux_index_artifact_bytes(pq_prefix);
  uint32_t active_points = base_points;
  uint32_t cumulative_inserts = 0;
  uint32_t cumulative_deletes = 0;
  uint32_t pending_updates = 0;

  if (workload_has_query(workload)) {
    if (workload == "query_only") {
      bool query_only_static_fast_path = true;
      for (uint32_t checkpoint = 0; checkpoint < std::max<uint32_t>(1, query_only_checkpoints); ++checkpoint) {
        SearchMetrics search = run_checkpoint_search<T>(store, query_bin, gt_prefix + "0.fbin",
                                                        recall_at, search_L, aligned_dim,
                                                        beamwidth, medoids, entry_init_mode,
                                                        query_only_static_fast_path,
                                                        query_threads, query_schedule,
                                                        warmup_enabled && checkpoint == 0, warmup_query_count,
                                                        pq_frontier_confirm_topk,
                                                        pq_confirm_mode, pq_confirm_margin, pq_confirm_delta,
                                                        query_pool_mode, query_pool_slack);
        ProcIo io_cur = read_proc_io();
        NativeIOSnapshot native_cur = read_native_io(store);
        uint64_t disk_size = file_size_if_exists(heap_path);
        append_row(out, workload, checkpoint, 0, base_points, active_points, 0, 0, recall_at,
                   search_L, beamwidth, query_threads, insert_threads, delete_threads,
                   buffer_pool_frames, page_size, insert_search_L, prune_R, prune_C,
                   search, BatchMetrics(), disk_size, aux_size_bytes, io_start, io_prev, io_cur,
                   native_start, native_prev, native_cur,
                   current_mem_kb(), "native_online", false, 0.0, 0.0, 0, store,
                   entry_init_mode, query_pool_mode, query_pool_slack,
                   "maintenance_mode=native_online;maintenance=threshold_only");
        io_prev = io_cur;
        native_prev = native_cur;
      }
      if (bg_flush_running) store.stop_bg_flush();
      return 0;
    }
    SearchMetrics search = run_checkpoint_search<T>(store, query_bin, gt_prefix + "0.fbin",
                                                    recall_at, search_L, aligned_dim,
                                                    beamwidth, medoids, entry_init_mode,
                                                    false,
                                                    query_threads, query_schedule,
                                                    warmup_enabled, warmup_query_count,
                                                    pq_frontier_confirm_topk,
                                                    pq_confirm_mode, pq_confirm_margin, pq_confirm_delta,
                                                    query_pool_mode, query_pool_slack);
    ProcIo io_cur = read_proc_io();
    NativeIOSnapshot native_cur = read_native_io(store);
    uint64_t disk_size = file_size_if_exists(heap_path);
    append_row(out, workload, 0, 0, base_points, active_points, 0, 0, recall_at,
               search_L, beamwidth, query_threads, insert_threads, delete_threads,
               buffer_pool_frames, page_size, insert_search_L, prune_R, prune_C,
               search, BatchMetrics(), disk_size, aux_size_bytes, io_start, io_prev, io_cur,
               native_start, native_prev, native_cur,
               current_mem_kb(), "native_online", false, 0.0, 0.0, 0, store,
               entry_init_mode, query_pool_mode, query_pool_slack,
               "maintenance_mode=native_online;maintenance=threshold_only");
    io_prev = io_cur;
    native_prev = native_cur;
  }

  diskann::DistanceL2 dist_cmp_l2;
  diskann::Distance<T> *dist_cmp = reinterpret_cast<diskann::Distance<T> *>(&dist_cmp_l2);
  omp_set_schedule(static_cast<omp_sched_t>(update_schedule), 1);

  for (uint32_t round = 0; round < update_rounds; ++round) {
    store.set_maintenance_epoch(round + 1);
    std::vector<uint32_t> delete_ids;
    std::vector<uint32_t> insert_ids;
    read_trace_file(trace_prefix + std::to_string(round), delete_ids, insert_ids);
    if (!workload_has_deletes(workload)) delete_ids.clear();
    if (!workload_has_inserts(workload)) insert_ids.clear();

    auto round_begin = std::chrono::high_resolution_clock::now();
    auto delete_begin = std::chrono::high_resolution_clock::now();
    ParallelErrorState delete_errors;
#pragma omp parallel for num_threads(std::max<uint32_t>(1, delete_threads)) schedule(runtime)
    for (int64_t i = 0; i < static_cast<int64_t>(delete_ids.size()); ++i) {
      if (delete_errors.failed.load(std::memory_order_acquire)) continue;
      try {
        store.mark_deleted(delete_ids[static_cast<size_t>(i)]);
      } catch (const diskann::ANNException &e) {
        delete_errors.capture(e);
      } catch (const std::exception &e) {
        delete_errors.capture(e);
      } catch (...) {
        delete_errors.capture_unknown();
      }
    }
    delete_errors.rethrow_if_failed("delete phase failed");
    auto delete_end = std::chrono::high_resolution_clock::now();

    std::vector<T> insert_vectors;
    size_t insert_dim = 0, insert_aligned_dim = 0;
    load_selected_vectors<T>(full_bin, insert_ids, insert_vectors, insert_dim, insert_aligned_dim);
    auto insert_begin = std::chrono::high_resolution_clock::now();
    if (!insert_ids.empty()) {
      std::vector<unsigned> init_ids = gather_start_ids(store, beamwidth, medoids, entry_init_mode);
      if (init_ids.empty() && store.is_active(store.entry_point())) {
        init_ids.push_back(static_cast<unsigned>(store.entry_point()));
      }
      store.protect_seed_pages(init_ids, 1);
      std::vector<std::vector<unsigned>> thread_pruned(std::max<uint32_t>(1, insert_threads));
      std::vector<std::vector<diskann::Neighbor>> thread_candidates(std::max<uint32_t>(1, insert_threads));
      const size_t chunk_size = std::max<size_t>(8, std::min<size_t>(
          64, insert_ids.size() / std::max<uint32_t>(1, insert_threads) + 1));
      ParallelErrorState insert_errors;
#pragma omp parallel num_threads(std::max<uint32_t>(1, insert_threads))
      {
        InPlaceSearchScratch local_scratch;
        local_scratch.init(aligned_dim, store.n_chunks(), sizeof(T));
        int tid = omp_get_thread_num();
        auto &candidates = thread_candidates[static_cast<size_t>(tid)];
        auto &pruned = thread_pruned[static_cast<size_t>(tid)];
#pragma omp for schedule(runtime)
        for (int64_t chunk_begin = 0; chunk_begin < static_cast<int64_t>(insert_ids.size());
             chunk_begin += static_cast<int64_t>(chunk_size)) {
          if (insert_errors.failed.load(std::memory_order_acquire)) continue;
          size_t chunk_start = static_cast<size_t>(chunk_begin);
          size_t chunk_end = std::min(chunk_start + chunk_size, insert_ids.size());
          try {
            std::vector<uint32_t> chunk_node_ids(insert_ids.begin() + static_cast<std::ptrdiff_t>(chunk_start),
                                                 insert_ids.begin() + static_cast<std::ptrdiff_t>(chunk_end));
            store.allocate_nodes_batch(chunk_node_ids.data(), chunk_node_ids.size());

            std::vector<std::vector<unsigned>> chunk_pruned(chunk_node_ids.size());
            for (size_t local_idx = 0; local_idx < chunk_node_ids.size(); ++local_idx) {
              size_t global_idx = chunk_start + local_idx;
              uint32_t node_id = chunk_node_ids[local_idx];
              const T *coords = insert_vectors.data() + global_idx * insert_aligned_dim;
              local_scratch.reset();
              candidates.clear();
              graph_iterate_to_fixed_point<T>(coords, insert_search_L, init_ids, nullptr,
                                              beamwidth, false,
                                              &store, aligned_dim, dist_cmp, &local_scratch,
                                              candidates, nullptr, DistanceScope::UPDATE,
                                              pq_frontier_confirm_topk,
                                              pq_confirm_mode, pq_confirm_margin, pq_confirm_delta);
              pruned.clear();
              graph_prune_neighbors_pq<T>(node_id, candidates, prune_R, prune_C, prune_alpha, pruned,
                                          &store, &local_scratch, 0, nullptr, DistanceScope::UPDATE);
              auto view = store.pin_node(node_id, WRITE);
              std::memcpy(view.coords, coords, aligned_dim * sizeof(T));
              view.degree = static_cast<uint16_t>(std::min<size_t>(prune_R, pruned.size()));
              for (uint16_t j = 0; j < view.degree; ++j) view.neighbors[j] = pruned[j];
              store.commit_node(view);
              store.unpin_node(view);
              chunk_pruned[local_idx] = pruned;
            }
            for (size_t local_idx = 0; local_idx < chunk_node_ids.size(); ++local_idx) {
              size_t global_idx = chunk_start + local_idx;
              const T *coords = insert_vectors.data() + global_idx * insert_aligned_dim;
              maybe_encode_pq<T>(store, chunk_node_ids[local_idx], coords);
            }
            store.publish_nodes_batch(chunk_node_ids.data(), chunk_node_ids.size());
            for (size_t local_idx = 0; local_idx < chunk_node_ids.size(); ++local_idx) {
              graph_inter_insert_deferred(chunk_node_ids[local_idx], chunk_pruned[local_idx],
                                          prune_R, &store);
            }
          } catch (const diskann::ANNException &e) {
            insert_errors.capture(e);
          } catch (const std::exception &e) {
            insert_errors.capture(e);
          } catch (...) {
            insert_errors.capture_unknown();
          }
        }
        local_scratch.flush_distance_stats(store.stats());
      }
      insert_errors.rethrow_if_failed("insert phase failed");
    }
    auto insert_end = std::chrono::high_resolution_clock::now();
    double foreground_s = std::chrono::duration<double>(insert_end - round_begin).count();
    pending_updates += static_cast<uint32_t>(delete_ids.size() + insert_ids.size());
    bool maintenance_triggered = false;
    auto maintenance_begin = std::chrono::high_resolution_clock::now();
    (void)drain_every_round;
    (void)sweep_every_round;
    (void)flush_before_checkpoint;
    uint32_t maintenance_budget = compute_sweep_budget(
        sweep_budget_mode, sweep_budget_value,
        std::max<uint32_t>(1, static_cast<uint32_t>(delete_ids.size() + insert_ids.size())),
        std::max<uint32_t>(1, active_points));
    uint32_t queued_edges = static_cast<uint32_t>(store.deferred_edge_bytes() / sizeof(DeferredEdge));
    uint32_t queued_targets = static_cast<uint32_t>(store.deferred_edge_target_count());
    bool overlay_pressure =
        (reverse_edge_overlay_high_water_mark > 0 &&
         queued_edges >= reverse_edge_overlay_high_water_mark) ||
        (reverse_edge_overlay_target_high_water_mark > 0 &&
         queued_targets >= reverse_edge_overlay_target_high_water_mark);
    if (overlay_pressure) {
      store.drain_deferred_edges<T>(prune_R, prune_C, prune_alpha, reverse_edge_pending_threshold,
                                    aligned_dim, std::max<uint32_t>(1, maintenance_budget), dist_cmp);
      maintenance_triggered = true;
    }
    if ((oversize_prune_high_water_mark > 0 &&
         store.pending_oversized_nodes() >= oversize_prune_high_water_mark) ||
        store.has_aged_oversized_nodes(sweep_round_period)) {
      store.prune_oversized_nodes<T>(std::max<uint32_t>(1, maintenance_budget),
                                     prune_R, prune_C, prune_alpha, aligned_dim,
                                     sweep_round_period, dist_cmp);
      maintenance_triggered = true;
    }
    if (tombstone_sweep_high_water_mark > 0 &&
        store.stats().tombstone_count.load(std::memory_order_relaxed) >= tombstone_sweep_high_water_mark) {
      store.sweep_repair_round<T>(std::max<uint32_t>(1, maintenance_budget),
                                  prune_R, prune_C, prune_alpha, aligned_dim, dist_cmp);
      maintenance_triggered = true;
    }
    if ((flush_dirty_page_high_water_mark > 0 &&
         store.dirty_page_count() >= flush_dirty_page_high_water_mark) ||
        (flush_dirty_ratio_threshold > 0.0 && store.dirty_ratio() >= flush_dirty_ratio_threshold)) {
      store.flush();
      maintenance_triggered = true;
    }
    auto maintenance_end = std::chrono::high_resolution_clock::now();
    if (maintenance_triggered) {
      pending_updates = 0;
    }
    auto round_end = maintenance_end;

    cumulative_inserts += static_cast<uint32_t>(insert_ids.size());
    cumulative_deletes += static_cast<uint32_t>(delete_ids.size());
    active_points = store.num_active();

    BatchMetrics batch;
    double delete_s = std::chrono::duration<double>(delete_end - delete_begin).count();
    double insert_s = std::chrono::duration<double>(insert_end - insert_begin).count();
    double maintenance_s = std::chrono::duration<double>(maintenance_end - maintenance_begin).count();
    double round_s = std::chrono::duration<double>(round_end - round_begin).count();
    batch.delete_throughput = delete_s > 0 ? static_cast<double>(delete_ids.size()) / delete_s : 0.0;
    batch.insert_throughput = insert_s > 0 ? static_cast<double>(insert_ids.size()) / insert_s : 0.0;
    batch.update_throughput = round_s > 0 ? static_cast<double>(delete_ids.size() + insert_ids.size()) / round_s : 0.0;
    batch.round_update_wall_time_s = round_s;
    SearchMetrics search;
    if (workload_has_query(workload) && workload != "query_only") {
      search = run_checkpoint_search<T>(store, query_bin, gt_prefix + std::to_string(round + 1) + ".fbin",
                                        recall_at, search_L, aligned_dim,
                                        beamwidth, medoids, entry_init_mode,
                                        false,
                                        query_threads, query_schedule,
                                        warmup_enabled && warmup_before_each_checkpoint,
                                        warmup_query_count,
                                        pq_frontier_confirm_topk,
                                        pq_confirm_mode, pq_confirm_margin, pq_confirm_delta,
                                        query_pool_mode, query_pool_slack);
    } else {
      search.candidate_pool_L_effective = insert_search_L;
      search.pq_frontier_confirm_topk_effective = 0;
    }
    ProcIo io_cur = read_proc_io();
    NativeIOSnapshot native_cur = read_native_io(store);
    uint64_t disk_size = file_size_if_exists(heap_path);
    append_row(out, workload, round + 1, round + 1, base_points, active_points,
               cumulative_inserts, cumulative_deletes, recall_at, search_L, beamwidth,
               query_threads, insert_threads, delete_threads,
               buffer_pool_frames, page_size, insert_search_L, prune_R, prune_C,
               search, batch, disk_size, aux_size_bytes, io_start, io_prev, io_cur,
               native_start, native_prev, native_cur, current_mem_kb(),
               "native_online", maintenance_triggered, foreground_s, maintenance_s, pending_updates, store,
               entry_init_mode, query_pool_mode, query_pool_slack,
               workload == "update_only"
                   ? "maintenance_mode=native_online;recall_disabled;maintenance=overlay_truth_thresholded"
                   : "maintenance_mode=native_online;maintenance=overlay_truth_thresholded");
    io_prev = io_cur;
    native_prev = native_cur;
  }

  store.drain_deferred_edges<T>(prune_R, prune_C, prune_alpha, 0,
                                aligned_dim, std::numeric_limits<uint32_t>::max(), dist_cmp);
  store.sweep_repair_round<T>(std::max<uint32_t>(1, pending_updates == 0 ? 1 : pending_updates),
                              prune_R, prune_C, prune_alpha, aligned_dim, dist_cmp);
  store.flush();
  if (bg_flush_running) store.stop_bg_flush();
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 69) {
    std::cout << "Usage: " << argv[0]
              << " <type> <workload> <base_bin> <full_bin> <query_bin> <trace_prefix> <gt_prefix>"
              << " <pq_prefix> <pq_chunks> <pq_sample_rate> <heap_path> <result_jsonl> <base_points> <update_rounds> <recall_at>"
              << " <search_L> <beamwidth> <query_threads> <insert_threads> <delete_threads>"
              << " <query_schedule> <update_schedule> <warmup_enabled> <warmup_query_count>"
              << " <warmup_before_each_checkpoint> <buffer_pool_frames> <page_size>"
              << " <build_R> <build_L> <build_C> <build_alpha> <build_threads> <saturate_graph>"
              << " <insert_search_L> <prune_R> <prune_C> <prune_alpha> <bp_query_frac> <bp_update_frac>"
              << " <flush_after_build_before_measurement> <drain_every_round_compat> <sweep_every_round_compat>"
              << " <maintenance_budget_mode> <maintenance_budget_value> <flush_before_checkpoint_compat>"
              << " <reverse_edge_overlay_high_water_mark> <reverse_edge_overlay_target_high_water_mark> <reverse_edge_pending_threshold> <oversize_prune_high_water_mark> <overlay_scan_age_compat> <tombstone_sweep_high_water_mark> <flush_dirty_page_high_water_mark> <flush_dirty_ratio_threshold>"
              << " <pq_frontier_confirm_enabled> <pq_frontier_confirm_topk>"
              << " <pq_confirm_mode> <pq_confirm_margin> <pq_confirm_delta> <query_pool_mode> <query_pool_slack>"
              << " <flush_budget_pages_per_cycle> <flush_wakeup_ms>"
              << " <build_mode> <build_memory_budget_gb> <build_temp_root> <entry_init_mode> <query_only_checkpoints> <reuse_existing_index>" << std::endl;
    return -1;
  }
  std::string type = argv[1];
  std::string workload = argv[2];
  std::string base_bin = argv[3];
  std::string full_bin = argv[4];
  std::string query_bin = argv[5];
  std::string trace_prefix = argv[6];
  std::string gt_prefix = argv[7];
  std::string pq_prefix = argv[8];
  uint32_t pq_chunks = static_cast<uint32_t>(std::stoul(argv[9]));
  double pq_sample_rate = std::atof(argv[10]);
  std::string heap_path = argv[11];
  std::string result_jsonl = argv[12];
  uint32_t base_points = static_cast<uint32_t>(std::stoul(argv[13]));
  uint32_t update_rounds = static_cast<uint32_t>(std::stoul(argv[14]));
  uint32_t recall_at = static_cast<uint32_t>(std::stoul(argv[15]));
  uint32_t search_L = static_cast<uint32_t>(std::stoul(argv[16]));
  uint32_t beamwidth = static_cast<uint32_t>(std::stoul(argv[17]));
  uint32_t query_threads = static_cast<uint32_t>(std::stoul(argv[18]));
  uint32_t insert_threads = static_cast<uint32_t>(std::stoul(argv[19]));
  uint32_t delete_threads = static_cast<uint32_t>(std::stoul(argv[20]));
  int query_schedule = parse_schedule(argv[21]);
  int update_schedule = parse_schedule(argv[22]);
  bool warmup_enabled = std::stoi(argv[23]) != 0;
  uint32_t warmup_query_count = static_cast<uint32_t>(std::stoul(argv[24]));
  bool warmup_before_each_checkpoint = std::stoi(argv[25]) != 0;
  uint32_t buffer_pool_frames = static_cast<uint32_t>(std::stoul(argv[26]));
  uint32_t page_size = static_cast<uint32_t>(std::stoul(argv[27]));
  uint32_t build_R = static_cast<uint32_t>(std::stoul(argv[28]));
  uint32_t build_L = static_cast<uint32_t>(std::stoul(argv[29]));
  uint32_t build_C = static_cast<uint32_t>(std::stoul(argv[30]));
  float build_alpha = std::atof(argv[31]);
  uint32_t build_threads = static_cast<uint32_t>(std::stoul(argv[32]));
  bool saturate_graph = std::stoi(argv[33]) != 0;
  uint32_t insert_search_L = static_cast<uint32_t>(std::stoul(argv[34]));
  uint32_t prune_R = static_cast<uint32_t>(std::stoul(argv[35]));
  uint32_t prune_C = static_cast<uint32_t>(std::stoul(argv[36]));
  float prune_alpha = std::atof(argv[37]);
  float bp_query_frac = std::atof(argv[38]);
  float bp_update_frac = std::atof(argv[39]);
  bool flush_after_build_before_measurement = std::stoi(argv[40]) != 0;
  bool drain_every_round = std::stoi(argv[41]) != 0;
  bool sweep_every_round = std::stoi(argv[42]) != 0;
  std::string sweep_budget_mode = argv[43];
  uint32_t sweep_budget_value = static_cast<uint32_t>(std::stoul(argv[44]));
  bool flush_before_checkpoint = std::stoi(argv[45]) != 0;
  uint32_t deferred_edge_high_water_mark = static_cast<uint32_t>(std::stoul(argv[46]));
  uint32_t drain_high_water_mark = static_cast<uint32_t>(std::stoul(argv[47]));
  uint32_t oversize_slack = static_cast<uint32_t>(std::stoul(argv[48]));
  uint32_t oversize_prune_high_water_mark = static_cast<uint32_t>(std::stoul(argv[49]));
  uint32_t sweep_round_period = static_cast<uint32_t>(std::stoul(argv[50]));
  uint32_t sweep_deleted_threshold = static_cast<uint32_t>(std::stoul(argv[51]));
  uint32_t flush_round_period = static_cast<uint32_t>(std::stoul(argv[52]));
  double flush_dirty_ratio_threshold = std::atof(argv[53]);
  bool pq_frontier_confirm_enabled = std::stoi(argv[54]) != 0;
  uint32_t pq_frontier_confirm_topk = static_cast<uint32_t>(std::stoul(argv[55]));
  std::string pq_confirm_mode = argv[56];
  double pq_confirm_margin = std::atof(argv[57]);
  uint32_t pq_confirm_delta = static_cast<uint32_t>(std::stoul(argv[58]));
  std::string query_pool_mode = argv[59];
  uint32_t query_pool_slack = static_cast<uint32_t>(std::stoul(argv[60]));
  uint32_t flush_budget_pages_per_cycle = static_cast<uint32_t>(std::stoul(argv[61]));
  uint32_t flush_wakeup_ms = static_cast<uint32_t>(std::stoul(argv[62]));
  std::string build_mode = argv[63];
  float build_memory_budget_gb = std::atof(argv[64]);
  std::string build_temp_root = argv[65];
  std::string entry_init_mode = argv[66];
  uint32_t query_only_checkpoints = static_cast<uint32_t>(std::stoul(argv[67]));
  bool reuse_existing_index = std::stoi(argv[68]) != 0;
  if (!pq_frontier_confirm_enabled) pq_frontier_confirm_topk = 0;

  try {
    if (type == "float") {
      return run_fair<float>(workload, base_bin, full_bin, query_bin, trace_prefix, gt_prefix,
                             pq_prefix, pq_chunks, pq_sample_rate,
                             heap_path, result_jsonl, base_points, update_rounds, recall_at,
                             search_L, beamwidth, query_threads, insert_threads, delete_threads,
                             query_schedule, update_schedule, warmup_enabled, warmup_query_count,
                             warmup_before_each_checkpoint, buffer_pool_frames, page_size,
                             build_R, build_L, build_C, build_alpha, build_threads, saturate_graph,
                             insert_search_L, prune_R, prune_C, prune_alpha, bp_query_frac, bp_update_frac,
                             flush_after_build_before_measurement, drain_every_round, sweep_every_round,
                             sweep_budget_mode, sweep_budget_value, flush_before_checkpoint,
                             deferred_edge_high_water_mark, drain_high_water_mark, oversize_slack, oversize_prune_high_water_mark, sweep_round_period,
                             sweep_deleted_threshold, flush_round_period, flush_dirty_ratio_threshold,
                             pq_frontier_confirm_topk,
                             pq_confirm_mode, pq_confirm_margin, pq_confirm_delta,
                             query_pool_mode, query_pool_slack,
                             flush_budget_pages_per_cycle, flush_wakeup_ms,
                             build_mode, build_memory_budget_gb, build_temp_root, entry_init_mode,
                             query_only_checkpoints, reuse_existing_index);
    }
    if (type == "uint8") {
      return run_fair<uint8_t>(workload, base_bin, full_bin, query_bin, trace_prefix, gt_prefix,
                               pq_prefix, pq_chunks, pq_sample_rate,
                               heap_path, result_jsonl, base_points, update_rounds, recall_at,
                               search_L, beamwidth, query_threads, insert_threads, delete_threads,
                               query_schedule, update_schedule, warmup_enabled, warmup_query_count,
                               warmup_before_each_checkpoint, buffer_pool_frames, page_size,
                               build_R, build_L, build_C, build_alpha, build_threads, saturate_graph,
                               insert_search_L, prune_R, prune_C, prune_alpha, bp_query_frac, bp_update_frac,
                               flush_after_build_before_measurement, drain_every_round, sweep_every_round,
                               sweep_budget_mode, sweep_budget_value, flush_before_checkpoint,
                               deferred_edge_high_water_mark, drain_high_water_mark, oversize_slack, oversize_prune_high_water_mark, sweep_round_period,
                               sweep_deleted_threshold, flush_round_period, flush_dirty_ratio_threshold,
                               pq_frontier_confirm_topk,
                               pq_confirm_mode, pq_confirm_margin, pq_confirm_delta,
                               query_pool_mode, query_pool_slack,
                               flush_budget_pages_per_cycle, flush_wakeup_ms,
                               build_mode, build_memory_budget_gb, build_temp_root, entry_init_mode,
                               query_only_checkpoints, reuse_existing_index);
    }
    if (type == "int8") {
      return run_fair<int8_t>(workload, base_bin, full_bin, query_bin, trace_prefix, gt_prefix,
                              pq_prefix, pq_chunks, pq_sample_rate,
                              heap_path, result_jsonl, base_points, update_rounds, recall_at,
                              search_L, beamwidth, query_threads, insert_threads, delete_threads,
                              query_schedule, update_schedule, warmup_enabled, warmup_query_count,
                              warmup_before_each_checkpoint, buffer_pool_frames, page_size,
                              build_R, build_L, build_C, build_alpha, build_threads, saturate_graph,
                              insert_search_L, prune_R, prune_C, prune_alpha, bp_query_frac, bp_update_frac,
                              flush_after_build_before_measurement, drain_every_round, sweep_every_round,
                              sweep_budget_mode, sweep_budget_value, flush_before_checkpoint,
                              deferred_edge_high_water_mark, drain_high_water_mark, oversize_slack, oversize_prune_high_water_mark, sweep_round_period,
                              sweep_deleted_threshold, flush_round_period, flush_dirty_ratio_threshold,
                              pq_frontier_confirm_topk,
                              pq_confirm_mode, pq_confirm_margin, pq_confirm_delta,
                              query_pool_mode, query_pool_slack,
                              flush_budget_pages_per_cycle, flush_wakeup_ms,
                              build_mode, build_memory_budget_gb, build_temp_root, entry_init_mode,
                              query_only_checkpoints, reuse_existing_index);
    }
    std::cerr << "Unsupported type: " << type << std::endl;
    return -1;
  } catch (const diskann::ANNException &e) {
    std::cerr << "pageann_benchmark failed with ANNException: " << e.message()
              << " (error_code=" << e.errorCode() << ")" << std::endl;
    return -1;
  } catch (const std::exception &e) {
    std::cerr << "pageann_benchmark failed: " << e.what() << std::endl;
    return -1;
  }
}
