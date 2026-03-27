// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <omp.h>

#include "Neighbor_Tag.h"
#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "timer.h"
#include "utils.h"
#include "v2/merge_insert.h"

namespace {

struct ProcIo {
  uint64_t read_bytes = 0;
  uint64_t write_bytes = 0;
};

struct ProcMem {
  double rss_kb = 0;
  double peak_rss_kb = 0;
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
  double query_io_us = 0;
  double query_distance_us = 0;
  double query_distance_ops = 0;
  uint64_t query_cache_hits = 0;
  uint64_t query_cache_misses = 0;
  double query_cache_hit_rate = 0;
};

struct BatchMetrics {
  double insert_throughput = 0;
  double delete_throughput = 0;
  double update_throughput = 0;
  double round_update_wall_time_s = 0;
};

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

static bool path_exists(const std::string &path) {
  return access(path.c_str(), F_OK) == 0;
}

static int parse_schedule(const std::string &schedule) {
  if (schedule == "dynamic") return omp_sched_dynamic;
  return omp_sched_static;
}

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

static uint64_t disk_index_size_bytes(const std::string &prefix) {
  return file_size_if_exists(prefix + "_disk.index");
}

static uint64_t aux_index_artifact_bytes(const std::string &prefix) {
  uint64_t total = 0;
  const std::vector<std::string> suffixes = {
      "_disk.index.tags",
      "_disk.index_with_only_nbrs",
      "_pq_compressed.bin",
      "_pq_pivots.bin",
      "_pq_pivots.bin_centroid.bin",
      "_pq_pivots.bin_rearrangement_perm.bin",
      "_pq_pivots.bin_chunk_offsets.bin",
  };
  for (const auto &suffix : suffixes) total += file_size_if_exists(prefix + suffix);
  return total;
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
static void load_selected_vectors(const std::string &data_path,
                                  const std::vector<uint32_t> &ids,
                                  std::vector<T> &out, size_t &dim,
                                  size_t &aligned_dim) {
  std::ifstream in(data_path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open data file: " + data_path);
  int32_t npts = 0, raw_dim = 0;
  in.read(reinterpret_cast<char *>(&npts), sizeof(int32_t));
  in.read(reinterpret_cast<char *>(&raw_dim), sizeof(int32_t));
  dim = static_cast<size_t>(raw_dim);
  aligned_dim = ROUND_UP(dim, 8);
  out.assign(ids.size() * aligned_dim, static_cast<T>(0));
  for (size_t i = 0; i < ids.size(); ++i) {
    const uint64_t offset = 2ULL * sizeof(int32_t) +
                            static_cast<uint64_t>(ids[i]) * dim * sizeof(T);
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    in.read(reinterpret_cast<char *>(out.data() + i * aligned_dim), dim * sizeof(T));
  }
}

template <typename T>
static void generate_topology_sidecar(const std::string &disk_index_path,
                                      const std::string &topology_path) {
  constexpr uint64_t kSectorLen = 4096;
  constexpr size_t kRequiredMetaItems = 5;

  std::unique_ptr<uint64_t[]> meta;
  size_t meta_npts = 0;
  size_t meta_dim = 0;
  diskann::load_bin<uint64_t>(disk_index_path, meta, meta_npts, meta_dim, 0);
  if (meta_npts < kRequiredMetaItems || meta_dim != 1) {
    throw std::runtime_error("unexpected Greator disk index metadata in " + disk_index_path);
  }

  const uint64_t npts = meta[0];
  const uint64_t ndims = meta[1];
  const uint64_t max_node_len = meta[3];
  const uint64_t nnodes_per_sector = meta[4];
  if (max_node_len <= (ndims * sizeof(T)) + sizeof(uint32_t) || nnodes_per_sector == 0) {
    throw std::runtime_error("invalid Greator disk index layout in " + disk_index_path);
  }

  const uint64_t range =
      (max_node_len - (ndims * sizeof(T)) - sizeof(uint32_t)) / sizeof(uint32_t);
  const uint64_t topo_node_len = (1 + range) * sizeof(uint32_t);
  const uint64_t topo_nodes_per_sector = kSectorLen / topo_node_len;
  if (range == 0 || topo_nodes_per_sector == 0) {
    throw std::runtime_error("invalid topology sidecar layout for " + disk_index_path);
  }

  std::ifstream in(disk_index_path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open Greator disk index: " + disk_index_path);
  std::ofstream out(topology_path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("failed to open Greator topology file: " + topology_path);

  std::vector<char> read_sector(kSectorLen);
  std::vector<char> write_sector(kSectorLen, 0);
  std::memcpy(write_sector.data(), &npts, sizeof(uint64_t));
  std::memcpy(write_sector.data() + sizeof(uint64_t), &range, sizeof(uint64_t));
  out.write(write_sector.data(), static_cast<std::streamsize>(write_sector.size()));

  uint64_t cur_node = 0;
  uint64_t out_nodes_in_sector = 0;
  const uint64_t total_input_sectors =
      (npts + nnodes_per_sector - 1) / nnodes_per_sector;
  for (uint64_t sector = 0; sector < total_input_sectors; ++sector) {
    in.clear();
    in.seekg(static_cast<std::streamoff>(kSectorLen * (sector + 1)), std::ios::beg);
    in.read(read_sector.data(), static_cast<std::streamsize>(read_sector.size()));
    if (!in) {
      throw std::runtime_error("failed to read Greator disk index sector from " + disk_index_path);
    }
    for (uint64_t sector_node = 0;
         sector_node < nnodes_per_sector && cur_node < npts;
         ++sector_node, ++cur_node) {
      if (out_nodes_in_sector == topo_nodes_per_sector) {
        out.write(write_sector.data(), static_cast<std::streamsize>(write_sector.size()));
        std::fill(write_sector.begin(), write_sector.end(), 0);
        out_nodes_in_sector = 0;
      }
      const uint64_t in_offset =
          (sector_node * max_node_len) + (ndims * sizeof(T));
      uint32_t nnbrs = 0;
      std::memcpy(&nnbrs, read_sector.data() + in_offset, sizeof(uint32_t));

      const uint64_t out_offset = out_nodes_in_sector * topo_node_len;
      std::memcpy(write_sector.data() + out_offset, &nnbrs, sizeof(uint32_t));
      std::memcpy(write_sector.data() + out_offset + sizeof(uint32_t),
                  read_sector.data() + in_offset + sizeof(uint32_t),
                  range * sizeof(uint32_t));
      ++out_nodes_in_sector;
    }
  }

  if (out_nodes_in_sector != 0) {
    out.write(write_sector.data(), static_cast<std::streamsize>(write_sector.size()));
  }
}

template <typename TagT>
static double compute_recall_from_gt(const std::string &truthset_file,
                                     const std::vector<TagT> &result_tags,
                                     uint32_t query_num, uint32_t recall_at) {
  unsigned *gt_ids = nullptr;
  float *gt_dists = nullptr;
  unsigned *gt_tags = nullptr;
  size_t gt_num = 0, gt_dim = 0;
  diskann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim, &gt_tags);
  double recall = diskann::calculate_recall(
      query_num, gt_ids, gt_dists, static_cast<unsigned>(gt_dim),
      reinterpret_cast<unsigned *>(const_cast<TagT *>(result_tags.data())),
      recall_at, recall_at);
  delete[] gt_ids;
  delete[] gt_dists;
  delete[] gt_tags;
  return recall;
}

template <typename T, typename TagT = uint32_t>
static SearchMetrics run_checkpoint_search(
    diskann::MergeInsert<T, TagT> &index, const std::string &query_file,
    const std::string &truthset_file, uint32_t recall_at, uint32_t search_L,
    uint32_t search_threads, int query_schedule,
    bool warmup_enabled, uint32_t warmup_query_count) {
  T *query = nullptr;
  size_t query_num = 0, query_dim = 0, query_aligned_dim = 0;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

  std::vector<TagT> result_tags(query_num * recall_at);
  std::vector<float> result_dists(query_num * recall_at);
  std::vector<double> latency_us(query_num, 0.0);
  std::vector<diskann::QueryStats> stats(query_num);

  omp_set_schedule(static_cast<omp_sched_t>(query_schedule), 1);
  if (warmup_enabled) {
#pragma omp parallel for num_threads(search_threads) schedule(runtime)
    for (int64_t i = 0; i < static_cast<int64_t>(std::min<size_t>(query_num, warmup_query_count)); ++i) {
      std::vector<TagT> warm_tags(recall_at);
      std::vector<float> warm_dists(recall_at);
      diskann::QueryStats warm_stats;
      index.search_sync(query + (i * query_aligned_dim), recall_at, search_L,
                        warm_tags.data(), warm_dists.data(), &warm_stats);
    }
  }

  auto begin = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(search_threads) schedule(runtime)
  for (int64_t i = 0; i < static_cast<int64_t>(query_num); ++i) {
    auto q0 = std::chrono::high_resolution_clock::now();
    index.search_sync(query + (i * query_aligned_dim), recall_at, search_L,
                      result_tags.data() + (i * recall_at),
                      result_dists.data() + (i * recall_at), stats.data() + i);
    auto q1 = std::chrono::high_resolution_clock::now();
    latency_us[static_cast<size_t>(i)] =
        std::chrono::duration<double, std::micro>(q1 - q0).count();
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::sort(latency_us.begin(), latency_us.end());

  double elapsed_s = std::chrono::duration<double>(end - begin).count();
  double total_lat = std::accumulate(latency_us.begin(), latency_us.end(), 0.0);
  double total_ios = 0.0;
  double total_io_us = 0.0;
  double total_cpu_us = 0.0;
  double total_cmps = 0.0;
  double total_cache_hits = 0.0;
  for (size_t i = 0; i < query_num; ++i) {
    total_ios += stats[i].n_ios;
    total_io_us += stats[i].io_us;
    total_cpu_us += stats[i].cpu_us;
    total_cmps += stats[i].n_cmps;
    total_cache_hits += stats[i].n_cache_hits;
  }

  SearchMetrics metrics;
  metrics.query_wall_time_s = elapsed_s;
  metrics.qps = elapsed_s > 0 ? static_cast<double>(query_num) / elapsed_s : 0.0;
  metrics.lat_avg_us = query_num > 0 ? total_lat / static_cast<double>(query_num) : 0.0;
  metrics.lat_p50_us = percentile_sorted(latency_us, 0.50);
  metrics.lat_p95_us = percentile_sorted(latency_us, 0.95);
  metrics.lat_p99_us = percentile_sorted(latency_us, 0.99);
  metrics.disk_ios = query_num > 0 ? total_ios / static_cast<double>(query_num) : 0.0;
  metrics.query_io_us = query_num > 0 ? total_io_us / static_cast<double>(query_num) : 0.0;
  metrics.query_distance_us = total_cpu_us;
  metrics.query_distance_ops = total_cmps;
  metrics.query_cache_hits = static_cast<uint64_t>(total_cache_hits);
  metrics.query_cache_misses = static_cast<uint64_t>(total_ios);
  double total_cache_accesses = total_cache_hits + total_ios;
  metrics.query_cache_hit_rate =
      total_cache_accesses > 0 ? (100.0 * total_cache_hits / total_cache_accesses) : 0.0;
  if (!truthset_file.empty() && path_exists(truthset_file)) {
    metrics.recall = compute_recall_from_gt(truthset_file, result_tags,
                                            static_cast<uint32_t>(query_num), recall_at);
  }

  diskann::aligned_free(query);
  return metrics;
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
                       const SearchMetrics &search, const BatchMetrics &batch,
                       uint64_t disk_size_bytes, uint64_t aux_size_bytes,
                       const ProcIo &io_start, const ProcIo &io_prev, const ProcIo &io_cur,
                       const ProcMem &mem, const std::string &maintenance_policy,
                       bool maintenance_triggered, double foreground_update_wall_time_s,
                       double maintenance_wall_time_s, uint32_t pending_updates,
                       const std::string &notes) {
  constexpr double kProcPageSize = 4096.0;
  double checkpoint_pages_read =
      static_cast<double>(io_cur.read_bytes - io_prev.read_bytes) / kProcPageSize;
  out << std::fixed << std::setprecision(4);
  out << "{";
  out << "\"system\":\"greator\",";
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
  out << "\"query_io_us_native\":" << search.query_io_us << ",";
  out << "\"proc_read_pages\":" << checkpoint_pages_read << ",";
  out << "\"rss_kb\":" << mem.rss_kb << ",";
  out << "\"peak_rss_kb\":" << mem.peak_rss_kb << ",";
  out << "\"disk_index_size_bytes\":" << disk_size_bytes << ",";
  out << "\"aux_index_artifact_bytes\":" << aux_size_bytes << ",";
  out << "\"index_num_nodes\":" << active_points << ",";
  out << "\"native_index_bytes_read\":\"\",";
  out << "\"native_index_bytes_written\":\"\",";
  out << "\"cumulative_native_index_bytes_read\":\"\",";
  out << "\"cumulative_native_index_bytes_written\":\"\",";
  out << "\"native_pages_flushed\":\"\",";
  out << "\"cumulative_native_pages_flushed\":\"\",";
  out << "\"native_write_amplification\":\"\",";
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
  out << "\"unsupported_reason\":\"\",";
  out << "\"notes\":\"" << json_escape(notes) << "\"";
  out << "}\n";
  out.flush();
}

template <typename T>
static int run_fair(const std::string &workload, const std::string &base_bin,
                    const std::string &full_bin, const std::string &query_bin,
                    const std::string &trace_prefix, const std::string &gt_prefix,
                    const std::string &index_prefix, const std::string &result_jsonl,
                    const std::string &tag_file, uint32_t update_rounds,
                    uint32_t base_points, uint32_t recall_at, uint32_t search_L,
                    uint32_t beamwidth, uint32_t search_threads,
                    uint32_t insert_threads, uint32_t delete_threads,
                    int query_schedule, int update_schedule,
                    bool warmup_enabled, uint32_t warmup_query_count,
                    bool warmup_before_each_checkpoint,
                    unsigned build_R, unsigned build_L, double build_B,
                    unsigned build_M, unsigned build_T, unsigned mem_L,
                    unsigned mem_R, float mem_alpha, unsigned disk_L,
                    unsigned disk_R, float disk_alpha, unsigned nodes_to_cache,
                    uint32_t id_map, double merge_trigger_ratio,
                    uint32_t query_only_checkpoints, bool reuse_existing_index) {
  std::string build_params = std::to_string(build_R) + " " + std::to_string(build_L) + " " +
                             std::to_string(build_B) + " " + std::to_string(build_M) + " " +
                             std::to_string(build_T);
  diskann::Metric metric = diskann::Metric::L2;
  if (!reuse_existing_index &&
      !diskann::build_disk_index<T>(base_bin.c_str(), index_prefix.c_str(),
                                    build_params.c_str(), metric, false,
                                    tag_file.empty() ? nullptr : tag_file.c_str())) {
    throw std::runtime_error("build_disk_index failed");
  }
  if (!reuse_existing_index && id_map == 2) {
    generate_topology_sidecar<T>(index_prefix + "_disk.index",
                                 index_prefix + "_disk.index_with_only_nbrs");
  }

  T *base_data = nullptr;
  size_t base_npts = 0, dim = 0, aligned_dim = 0;
  diskann::load_aligned_bin<T>(base_bin, base_data, base_npts, dim, aligned_dim);

  diskann::DistanceL2 dist_float;
  diskann::DistanceL2Int8 dist_i8;
  diskann::DistanceL2UInt8 dist_u8;
  diskann::Distance<T> *dist = nullptr;
  if (std::is_same<T, float>::value) dist = reinterpret_cast<diskann::Distance<T> *>(&dist_float);
  if (std::is_same<T, int8_t>::value) dist = reinterpret_cast<diskann::Distance<T> *>(&dist_i8);
  if (std::is_same<T, uint8_t>::value) dist = reinterpret_cast<diskann::Distance<T> *>(&dist_u8);

  diskann::Parameters params;
  params.Set<unsigned>("L_mem", mem_L);
  params.Set<unsigned>("R_mem", mem_R);
  params.Set<float>("alpha_mem", mem_alpha);
  params.Set<unsigned>("L_disk", disk_L);
  params.Set<unsigned>("R_disk", disk_R);
  params.Set<float>("alpha_disk", disk_alpha);
  params.Set<unsigned>("C", 160);
  params.Set<unsigned>("beamwidth", beamwidth);
  params.Set<unsigned>("nodes_to_cache", nodes_to_cache);
  params.Set<unsigned>("num_search_threads", search_threads);

  auto make_index = [&](const std::string &disk_prefix_in,
                        const std::string &disk_prefix_out) {
    return std::make_unique<diskann::MergeInsert<T, uint32_t>>(
        params, dim, index_prefix + "_mem", disk_prefix_in, disk_prefix_out,
        dist, metric, false, index_prefix);
  };

  auto index = make_index(index_prefix, index_prefix + "_merge");
  if (id_map == 2) {
    // Match Greator's native driver: id_map=2 merges back into the base prefix.
    index->_disk_index_prefix_out = index_prefix;
  }

  std::ofstream out(result_jsonl, std::ios::out | std::ios::trunc);
  ProcIo io_start = read_proc_io();
  ProcIo io_prev = io_start;

  uint32_t active_points = base_points;
  uint32_t cumulative_inserts = 0;
  uint32_t cumulative_deletes = 0;
  uint32_t pending_updates = 0;
  const uint32_t merge_trigger_updates =
      std::max<uint32_t>(1, static_cast<uint32_t>(std::ceil(base_points * merge_trigger_ratio)));

  if (workload_has_query(workload)) {
    if (workload == "query_only") {
      for (uint32_t checkpoint = 0; checkpoint < std::max<uint32_t>(1, query_only_checkpoints); ++checkpoint) {
        SearchMetrics search =
            run_checkpoint_search(*index, query_bin, gt_prefix + "0.fbin",
                                  recall_at, search_L, search_threads, query_schedule,
                                  warmup_enabled && checkpoint == 0, warmup_query_count);
        ProcIo io_cur = read_proc_io();
        uint64_t disk_size = disk_index_size_bytes(index->ret_merge_prefix());
        uint64_t aux_size = aux_index_artifact_bytes(index->ret_merge_prefix());
        append_row(out, workload, checkpoint, 0, base_points, active_points,
                   cumulative_inserts, cumulative_deletes, recall_at, search_L,
                   beamwidth, search, BatchMetrics(), disk_size, aux_size, io_start, io_prev, io_cur,
                   current_mem_kb(), "native_online", false, 0.0, 0.0, 0,
                   "maintenance_mode=native_online;id_map=" + std::to_string(id_map));
        io_prev = io_cur;
      }
      diskann::aligned_free(base_data);
      return 0;
    }
    SearchMetrics search =
        run_checkpoint_search(*index, query_bin, gt_prefix + "0.fbin",
                              recall_at, search_L, search_threads, query_schedule,
                              warmup_enabled, warmup_query_count);
    ProcIo io_cur = read_proc_io();
    uint64_t disk_size = disk_index_size_bytes(index->ret_merge_prefix());
    uint64_t aux_size = aux_index_artifact_bytes(index->ret_merge_prefix());
    append_row(out, workload, 0, 0, base_points, active_points,
               cumulative_inserts, cumulative_deletes, recall_at, search_L,
               beamwidth, search, BatchMetrics(), disk_size, aux_size, io_start, io_prev, io_cur,
               current_mem_kb(), "native_online", false, 0.0, 0.0, 0,
               "maintenance_mode=native_online;id_map=" + std::to_string(id_map));
    io_prev = io_cur;
  }

  omp_set_num_threads(std::max(std::max(search_threads, insert_threads), delete_threads));
  omp_set_schedule(static_cast<omp_sched_t>(update_schedule), 1);
  for (uint32_t round = 0; round < update_rounds; ++round) {
    std::vector<uint32_t> delete_ids;
    std::vector<uint32_t> insert_ids;
    read_trace_file(trace_prefix + std::to_string(round), delete_ids, insert_ids);
    if (!workload_has_deletes(workload)) delete_ids.clear();
    if (!workload_has_inserts(workload)) insert_ids.clear();

    auto round_begin = std::chrono::high_resolution_clock::now();
    auto delete_begin = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(std::max<uint32_t>(1, delete_threads)) schedule(runtime)
    for (int64_t i = 0; i < static_cast<int64_t>(delete_ids.size()); ++i) {
      index->lazy_delete(delete_ids[static_cast<size_t>(i)]);
    }
    auto delete_end = std::chrono::high_resolution_clock::now();

    std::vector<T> insert_vectors;
    size_t insert_dim = 0, insert_aligned_dim = 0;
    load_selected_vectors<T>(full_bin, insert_ids, insert_vectors, insert_dim, insert_aligned_dim);
    auto insert_begin = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(std::max<uint32_t>(1, insert_threads)) schedule(runtime)
    for (int64_t i = 0; i < static_cast<int64_t>(insert_ids.size()); ++i) {
      index->insert(insert_vectors.data() + static_cast<size_t>(i) * insert_aligned_dim,
                    insert_ids[static_cast<size_t>(i)]);
    }
    auto insert_end = std::chrono::high_resolution_clock::now();
    double foreground_s =
        std::chrono::duration<double>(insert_end - round_begin).count();
    pending_updates += static_cast<uint32_t>(delete_ids.size() + insert_ids.size());
    bool maintenance_triggered = pending_updates >= merge_trigger_updates;
    auto maintenance_begin = std::chrono::high_resolution_clock::now();
    if (maintenance_triggered) {
      index->final_merge(id_map);
      pending_updates = 0;
    }
    auto maintenance_end = std::chrono::high_resolution_clock::now();
    auto round_end = maintenance_end;

    cumulative_inserts += static_cast<uint32_t>(insert_ids.size());
    cumulative_deletes += static_cast<uint32_t>(delete_ids.size());
    active_points = base_points + cumulative_inserts - cumulative_deletes;

    double delete_s = std::chrono::duration<double>(delete_end - delete_begin).count();
    double insert_s = std::chrono::duration<double>(insert_end - insert_begin).count();
    double maintenance_s = std::chrono::duration<double>(maintenance_end - maintenance_begin).count();
    double round_s = std::chrono::duration<double>(round_end - round_begin).count();
    BatchMetrics batch;
    batch.delete_throughput = delete_s > 0 ? static_cast<double>(delete_ids.size()) / delete_s : 0.0;
    batch.insert_throughput = insert_s > 0 ? static_cast<double>(insert_ids.size()) / insert_s : 0.0;
    batch.update_throughput = round_s > 0 ? static_cast<double>(delete_ids.size() + insert_ids.size()) / round_s : 0.0;
    batch.round_update_wall_time_s = round_s;
    SearchMetrics search;
    if (workload_has_query(workload) && workload != "query_only") {
      search = run_checkpoint_search(*index, query_bin,
                                     gt_prefix + std::to_string(round + 1) + ".fbin",
                                     recall_at, search_L, search_threads, query_schedule,
                                     warmup_enabled && warmup_before_each_checkpoint,
                                     warmup_query_count);
    }

    ProcIo io_cur = read_proc_io();
    uint64_t disk_size = disk_index_size_bytes(index->ret_merge_prefix());
    uint64_t aux_size = aux_index_artifact_bytes(index->ret_merge_prefix());
    append_row(out, workload, round + 1, round + 1, base_points, active_points,
               cumulative_inserts, cumulative_deletes, recall_at, search_L,
               beamwidth, search, batch, disk_size, aux_size, io_start, io_prev, io_cur, current_mem_kb(),
               "native_online", maintenance_triggered, foreground_s, maintenance_s, pending_updates,
               workload == "update_only"
                   ? "maintenance_mode=native_online;recall_disabled;merge_policy=mem_ratio_3pct;id_map=" + std::to_string(id_map)
                   : "maintenance_mode=native_online;merge_policy=mem_ratio_3pct;id_map=" + std::to_string(id_map));
    io_prev = io_cur;
  }

  if (pending_updates > 0) {
    index->final_merge(id_map);
  }

  diskann::aligned_free(base_data);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 40) {
    std::cout << "Usage: " << argv[0]
              << " <type> <workload> <base_bin> <full_bin> <query_bin> <trace_prefix> <gt_prefix>"
              << " <index_prefix> <result_jsonl> <tag_file> <base_points> <update_rounds>"
              << " <recall_at> <search_L> <beamwidth> <search_threads> <insert_threads>"
              << " <delete_threads> <query_schedule> <update_schedule> <warmup_enabled>"
              << " <warmup_query_count> <warmup_before_each_checkpoint> <build_R> <build_L>"
              << " <build_B> <build_M> <build_T> <nodes_to_cache> <L_mem> <R_mem> <alpha_mem>"
              << " <L_disk> <R_disk> <alpha_disk> <id_map> <merge_trigger_ratio> <query_only_checkpoints> <reuse_existing_index>"
              << std::endl;
    return -1;
  }

  std::string type = argv[1];
  std::string workload = argv[2];
  std::string base_bin = argv[3];
  std::string full_bin = argv[4];
  std::string query_bin = argv[5];
  std::string trace_prefix = argv[6];
  std::string gt_prefix = argv[7];
  std::string index_prefix = argv[8];
  std::string result_jsonl = argv[9];
  std::string tag_file = argv[10];
  uint32_t base_points = static_cast<uint32_t>(std::stoul(argv[11]));
  uint32_t update_rounds = static_cast<uint32_t>(std::stoul(argv[12]));
  uint32_t recall_at = static_cast<uint32_t>(std::stoul(argv[13]));
  uint32_t search_L = static_cast<uint32_t>(std::stoul(argv[14]));
  uint32_t beamwidth = static_cast<uint32_t>(std::stoul(argv[15]));
  uint32_t search_threads = static_cast<uint32_t>(std::stoul(argv[16]));
  uint32_t insert_threads = static_cast<uint32_t>(std::stoul(argv[17]));
  uint32_t delete_threads = static_cast<uint32_t>(std::stoul(argv[18]));
  int query_schedule = parse_schedule(argv[19]);
  int update_schedule = parse_schedule(argv[20]);
  bool warmup_enabled = std::stoi(argv[21]) != 0;
  uint32_t warmup_query_count = static_cast<uint32_t>(std::stoul(argv[22]));
  bool warmup_before_each_checkpoint = std::stoi(argv[23]) != 0;
  unsigned build_R = static_cast<unsigned>(std::stoul(argv[24]));
  unsigned build_L = static_cast<unsigned>(std::stoul(argv[25]));
  double build_B = std::atof(argv[26]);
  unsigned build_M = static_cast<unsigned>(std::stoul(argv[27]));
  unsigned build_T = static_cast<unsigned>(std::stoul(argv[28]));
  unsigned nodes_to_cache = static_cast<unsigned>(std::stoul(argv[29]));
  unsigned L_mem = static_cast<unsigned>(std::stoul(argv[30]));
  unsigned R_mem = static_cast<unsigned>(std::stoul(argv[31]));
  float alpha_mem = static_cast<float>(std::atof(argv[32]));
  unsigned L_disk = static_cast<unsigned>(std::stoul(argv[33]));
  unsigned R_disk = static_cast<unsigned>(std::stoul(argv[34]));
  float alpha_disk = static_cast<float>(std::atof(argv[35]));
  uint32_t id_map = static_cast<uint32_t>(std::stoul(argv[36]));
  double merge_trigger_ratio = std::atof(argv[37]);
  uint32_t query_only_checkpoints = static_cast<uint32_t>(std::stoul(argv[38]));
  bool reuse_existing_index = std::stoi(argv[39]) != 0;

  try {
    if (type == "float") {
      return run_fair<float>(workload, base_bin, full_bin, query_bin, trace_prefix, gt_prefix,
                             index_prefix, result_jsonl, tag_file, update_rounds, base_points,
                             recall_at, search_L, beamwidth, search_threads, insert_threads,
                             delete_threads, query_schedule, update_schedule, warmup_enabled,
                             warmup_query_count, warmup_before_each_checkpoint,
                             build_R, build_L, build_B, build_M, build_T,
                             L_mem, R_mem, alpha_mem, L_disk, R_disk, alpha_disk,
                             nodes_to_cache, id_map, merge_trigger_ratio, query_only_checkpoints, reuse_existing_index);
    }
    if (type == "int8") {
      return run_fair<int8_t>(workload, base_bin, full_bin, query_bin, trace_prefix, gt_prefix,
                              index_prefix, result_jsonl, tag_file, update_rounds, base_points,
                              recall_at, search_L, beamwidth, search_threads, insert_threads,
                              delete_threads, query_schedule, update_schedule, warmup_enabled,
                              warmup_query_count, warmup_before_each_checkpoint,
                              build_R, build_L, build_B, build_M, build_T,
                              L_mem, R_mem, alpha_mem, L_disk, R_disk, alpha_disk,
                              nodes_to_cache, id_map, merge_trigger_ratio, query_only_checkpoints, reuse_existing_index);
    }
    if (type == "uint8") {
      return run_fair<uint8_t>(workload, base_bin, full_bin, query_bin, trace_prefix, gt_prefix,
                               index_prefix, result_jsonl, tag_file, update_rounds, base_points,
                               recall_at, search_L, beamwidth, search_threads, insert_threads,
                               delete_threads, query_schedule, update_schedule, warmup_enabled,
                               warmup_query_count, warmup_before_each_checkpoint,
                               build_R, build_L, build_B, build_M, build_T,
                               L_mem, R_mem, alpha_mem, L_disk, R_disk, alpha_disk,
                               nodes_to_cache, id_map, merge_trigger_ratio, query_only_checkpoints, reuse_existing_index);
    }
    std::cerr << "Unsupported type: " << type << std::endl;
    return -1;
  } catch (const std::exception &e) {
    std::cerr << "benchmark_adapter failed: " << e.what() << std::endl;
    return -1;
  }
}
