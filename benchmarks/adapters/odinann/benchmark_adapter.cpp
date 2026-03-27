#include <algorithm>
#include <chrono>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <omp.h>

#include "distance.h"
#include "dynamic_index.h"
#include "index.h"
#include "nbr/nbr.h"
#include "utils/partition.h"
#include "utils/index_build_utils.h"
#include "utils/log.h"
#include "utils.h"

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

static uint64_t pipeann_index_size_bytes(const std::string &active_prefix,
                                         const std::string &base_prefix) {
  (void)base_prefix;
  return file_size_if_exists(active_prefix + "_disk.index");
}

static uint64_t pipeann_aux_index_artifact_bytes(const std::string &active_prefix,
                                                 const std::string &base_prefix) {
  uint64_t total = 0;
  const std::vector<std::string> common_suffixes = {
      "_disk.index.tags",
      "_pq_compressed.bin",
      "_pq_pivots.bin",
      "_pq_pivots.bin_centroid.bin",
      "_pq_pivots.bin_rearrangement_perm.bin",
      "_pq_pivots.bin_chunk_offsets.bin",
  };
  for (const auto &suffix : common_suffixes) total += file_size_if_exists(active_prefix + suffix);
  for (const auto &suffix : {std::string("_mem.index"), std::string("_mem.index.data"), std::string("_mem.index.tags")}) {
    total += file_size_if_exists(base_prefix + suffix);
  }
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
static void load_selected_vectors(const std::string &data_path, const std::vector<uint32_t> &ids,
                                  std::vector<T> &out, size_t &dim) {
  std::ifstream in(data_path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to open data file: " + data_path);
  int32_t npts = 0, raw_dim = 0;
  in.read(reinterpret_cast<char *>(&npts), sizeof(int32_t));
  in.read(reinterpret_cast<char *>(&raw_dim), sizeof(int32_t));
  dim = static_cast<size_t>(raw_dim);
  out.assign(ids.size() * dim, static_cast<T>(0));
  for (size_t i = 0; i < ids.size(); ++i) {
    uint64_t offset = 2ULL * sizeof(int32_t) + static_cast<uint64_t>(ids[i]) * dim * sizeof(T);
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    in.read(reinterpret_cast<char *>(out.data() + i * dim), dim * sizeof(T));
  }
}

template <typename T, typename TagT = uint32_t>
static void build_sampled_mem_index(const std::string &base_bin, const std::string &index_prefix,
                                    uint32_t build_R, uint32_t build_L, uint32_t build_threads,
                                    pipeann::Metric metric, double sample_rate) {
  size_t base_num = 0, base_dim = 0;
  pipeann::get_bin_metadata(base_bin, base_num, base_dim);
  pipeann::IndexBuildParameters mem_params;
  mem_params.set(build_R, build_L, 384, 1.2f, build_threads, true);
  if (base_num <= 500000) {
    std::vector<TagT> tags(base_num);
    for (size_t i = 0; i < base_num; ++i) tags[i] = static_cast<TagT>(i);
    pipeann::Index<T, TagT> mem_index(metric, base_dim);
    mem_index.build(base_bin.c_str(), base_num, mem_params, tags);
    mem_index.save((index_prefix + "_mem.index").c_str());
    return;
  }

  std::string sample_prefix = index_prefix + "_mem_sample";
  gen_random_slice<T>(base_bin, sample_prefix, sample_rate);

  std::string sample_data_bin = sample_prefix + "_data.bin";
  std::string sample_id_bin = sample_prefix + "_ids.bin";

  size_t data_num = 0, data_dim = 0;
  pipeann::get_bin_metadata(sample_data_bin, data_num, data_dim);
  std::vector<TagT> tags;
  size_t tag_num = 0, tag_dim = 0;
  pipeann::load_bin<TagT>(sample_id_bin, tags, tag_num, tag_dim, 0);
  if (tag_num != data_num || tag_dim != 1) {
    throw std::runtime_error("sampled mem-index tags mismatch");
  }

  pipeann::Index<T, TagT> mem_index(metric, data_dim);
  mem_index.build(sample_data_bin.c_str(), data_num, mem_params, tags);
  mem_index.save((index_prefix + "_mem.index").c_str());

  std::remove(sample_data_bin.c_str());
  std::remove(sample_id_bin.c_str());
}

static std::string json_escape(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

template <typename T, typename TagT = uint32_t>
static SearchMetrics run_checkpoint_search(pipeann::DynamicSSDIndex<T, TagT> &index,
                                           const std::string &query_file,
                                           const std::string &truthset_file,
                                           uint32_t recall_at, uint32_t search_L,
                                           uint32_t mem_L, uint32_t beamwidth,
                                           uint32_t search_threads, int query_schedule,
                                           bool warmup_enabled, uint32_t warmup_query_count) {
  T *query = nullptr;
  size_t query_num = 0, query_dim = 0;
  pipeann::load_bin<T>(query_file, query, query_num, query_dim);

  std::vector<TagT> result_tags(query_num * recall_at);
  std::vector<float> result_dists(query_num * recall_at);
  std::vector<double> latency_us(query_num, 0.0);
  std::vector<pipeann::QueryStats> stats(query_num);

  omp_set_schedule(static_cast<omp_sched_t>(query_schedule), 1);
  if (warmup_enabled) {
#pragma omp parallel for num_threads(search_threads) schedule(runtime)
    for (int64_t i = 0; i < static_cast<int64_t>(std::min<size_t>(query_num, warmup_query_count)); ++i) {
      std::vector<TagT> warm_tags(recall_at);
      std::vector<float> warm_dists(recall_at);
      pipeann::QueryStats warm_stats;
      index.search(query + (i * query_dim), recall_at, mem_L, search_L, beamwidth,
                   warm_tags.data(), warm_dists.data(), &warm_stats, true);
    }
  }

  auto begin = std::chrono::high_resolution_clock::now();
#pragma omp parallel for num_threads(search_threads) schedule(runtime)
  for (int64_t i = 0; i < static_cast<int64_t>(query_num); ++i) {
    auto q0 = std::chrono::high_resolution_clock::now();
    index.search(query + (i * query_dim), recall_at, mem_L, search_L, beamwidth,
                 result_tags.data() + (i * recall_at),
                 result_dists.data() + (i * recall_at), stats.data() + i, true);
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

  SearchMetrics m;
  m.query_wall_time_s = elapsed_s;
  m.qps = elapsed_s > 0 ? static_cast<double>(query_num) / elapsed_s : 0.0;
  m.lat_avg_us = query_num > 0 ? total_lat / static_cast<double>(query_num) : 0.0;
  m.lat_p50_us = percentile_sorted(latency_us, 0.50);
  m.lat_p95_us = percentile_sorted(latency_us, 0.95);
  m.lat_p99_us = percentile_sorted(latency_us, 0.99);
  m.disk_ios = query_num > 0 ? total_ios / static_cast<double>(query_num) : 0.0;
  m.query_io_us = query_num > 0 ? total_io_us / static_cast<double>(query_num) : 0.0;
  m.query_distance_us = total_cpu_us;
  m.query_distance_ops = total_cmps;
  m.query_cache_hits = static_cast<uint64_t>(total_cache_hits);
  m.query_cache_misses = static_cast<uint64_t>(total_ios);
  double total_cache_accesses = total_cache_hits + total_ios;
  m.query_cache_hit_rate =
      total_cache_accesses > 0 ? (100.0 * total_cache_hits / total_cache_accesses) : 0.0;

  if (path_exists(truthset_file)) {
    unsigned *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t gt_num = 0, gt_dim = 0;
    pipeann::load_truthset(truthset_file, gt_ids, gt_dists, gt_num, gt_dim);
    m.recall = pipeann::calculate_recall(static_cast<unsigned>(query_num), gt_ids, gt_dists,
                                         static_cast<unsigned>(gt_dim),
                                         reinterpret_cast<unsigned *>(result_tags.data()),
                                         recall_at, recall_at);
    delete[] gt_ids;
    delete[] gt_dists;
  }

  delete[] query;
  return m;
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
  out << "\"system\":\"odinann\",";
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
                    const std::string &trace_prefix, const std::string &gt_dir,
                    const std::string &index_prefix, const std::string &result_jsonl,
                    const std::string &tag_file, uint32_t base_points,
                    uint32_t update_rounds, uint32_t recall_at, uint32_t search_L,
                    uint32_t mem_L, uint32_t beamwidth, uint32_t search_threads,
                    uint32_t insert_threads, uint32_t delete_threads,
                    int query_schedule, int update_schedule,
                    bool warmup_enabled, uint32_t warmup_query_count,
                    bool warmup_before_each_checkpoint,
                    uint32_t build_R, uint32_t build_L, uint32_t pq_bytes,
                    uint32_t build_M, uint32_t build_T, uint32_t merge_threads,
                    uint32_t merge_sampled_nbrs, const std::string &nbr_type,
                    double mem_sample_rate, uint32_t query_only_checkpoints,
                    bool reuse_existing_index, uint32_t merge_round,
                    double merge_io_threshold) {
  pipeann::Metric metric = pipeann::Metric::L2;
  pipeann::AbstractNeighbor<T> *nbr_handler = pipeann::get_nbr_handler<T>(metric, nbr_type);
  const bool use_mem_index =
      ((workload == "query_only" || workload == "build_only") && mem_L > 0);
  const uint32_t effective_mem_L = use_mem_index ? mem_L : 0;
  if (!reuse_existing_index &&
      !pipeann::build_disk_index<T>(base_bin.c_str(), index_prefix.c_str(),
                                    build_R, build_L, build_M, build_T, pq_bytes,
                                    metric, tag_file.empty() ? nullptr : tag_file.c_str(),
                                    nbr_handler, nullptr)) {
    throw std::runtime_error("build_disk_index failed");
  }
  if (!reuse_existing_index && use_mem_index) {
    build_sampled_mem_index<T, uint32_t>(base_bin, index_prefix, build_R, build_L,
                                         std::max({search_threads, insert_threads, delete_threads}),
                                         metric, mem_sample_rate);
  }
  if (workload == "build_only") {
    return 0;
  }

  pipeann::IndexBuildParameters params;
  params.set(build_R, build_L, 384, 1.2f, std::max({search_threads, insert_threads, delete_threads}),
             true, beamwidth);

  pipeann::DistanceL2Float dist_float;
  pipeann::DistanceL2Int8 dist_i8;
  pipeann::DistanceL2UInt8 dist_u8;
  pipeann::Distance<T> *dist = nullptr;
  if (std::is_same<T, float>::value) dist = reinterpret_cast<pipeann::Distance<T> *>(&dist_float);
  if (std::is_same<T, int8_t>::value) dist = reinterpret_cast<pipeann::Distance<T> *>(&dist_i8);
  if (std::is_same<T, uint8_t>::value) dist = reinterpret_cast<pipeann::Distance<T> *>(&dist_u8);

  pipeann::DynamicSSDIndex<T, uint32_t> index(params, index_prefix, index_prefix + "_merge", dist, metric,
                                              BEAM_SEARCH, use_mem_index);

  std::ofstream out(result_jsonl, std::ios::out | std::ios::trunc);
  ProcIo io_start = read_proc_io();
  ProcIo io_prev = io_start;

  uint32_t active_points = base_points;
  uint32_t cumulative_inserts = 0;
  uint32_t cumulative_deletes = 0;
  uint32_t pending_updates = 0;
  double ref_disk_ios = 0.0;
  double last_query_disk_ios = 0.0;

  if (workload_has_query(workload)) {
    if (workload == "query_only") {
      for (uint32_t checkpoint = 0; checkpoint < std::max<uint32_t>(1, query_only_checkpoints); ++checkpoint) {
        SearchMetrics search = run_checkpoint_search(index, query_bin, gt_dir + "/gt_0.bin",
                                                     recall_at, search_L, effective_mem_L, beamwidth, search_threads,
                                                     query_schedule, warmup_enabled && checkpoint == 0, warmup_query_count);
        ProcIo io_cur = read_proc_io();
        uint64_t disk_size = pipeann_index_size_bytes(index._disk_index_prefix_in, index_prefix);
        uint64_t aux_size = pipeann_aux_index_artifact_bytes(index._disk_index_prefix_in, index_prefix);
        append_row(out, workload, checkpoint, 0, base_points, active_points, 0, 0, recall_at,
                   search_L, beamwidth, search, BatchMetrics(), disk_size, aux_size, io_start, io_prev, io_cur,
                   current_mem_kb(), "native_online", false, 0.0, 0.0, 0,
                   "maintenance_mode=native_online");
        io_prev = io_cur;
      }
      return 0;
    }
    SearchMetrics search = run_checkpoint_search(index, query_bin, gt_dir + "/gt_0.bin",
                                                 recall_at, search_L, effective_mem_L, beamwidth, search_threads,
                                                 query_schedule, warmup_enabled, warmup_query_count);
    ref_disk_ios = search.disk_ios;
    last_query_disk_ios = search.disk_ios;
    ProcIo io_cur = read_proc_io();
    uint64_t disk_size = pipeann_index_size_bytes(index._disk_index_prefix_in, index_prefix);
    uint64_t aux_size = pipeann_aux_index_artifact_bytes(index._disk_index_prefix_in, index_prefix);
    append_row(out, workload, 0, 0, base_points, active_points, 0, 0, recall_at,
               search_L, beamwidth, search, BatchMetrics(), disk_size, aux_size, io_start, io_prev, io_cur,
               current_mem_kb(), "native_online", false, 0.0, 0.0, 0,
               "maintenance_mode=native_online");
    io_prev = io_cur;
  }

  omp_set_num_threads(std::max({search_threads, insert_threads, delete_threads}));
  omp_set_schedule(static_cast<omp_sched_t>(update_schedule), 1);
  for (uint32_t round = 0; round < update_rounds; ++round) {
    std::vector<uint32_t> delete_ids, insert_ids;
    read_trace_file(trace_prefix + std::to_string(round), delete_ids, insert_ids);
    if (!workload_has_deletes(workload)) delete_ids.clear();
    if (!workload_has_inserts(workload)) insert_ids.clear();

    auto round_begin = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Round " << round << " begin deletes=" << delete_ids.size()
              << " inserts=" << insert_ids.size();
    std::vector<T> insert_vectors;
    size_t dim = 0;
    load_selected_vectors<T>(full_bin, insert_ids, insert_vectors, dim);
    LOG(INFO) << "Round " << round << " insert vectors loaded dim=" << dim;
    auto delete_begin = std::chrono::high_resolution_clock::now();
    auto delete_future = std::async(std::launch::async, [&]() {
#pragma omp parallel for num_threads(std::max<uint32_t>(1, delete_threads)) schedule(runtime)
      for (int64_t i = 0; i < static_cast<int64_t>(delete_ids.size()); ++i) {
        index.lazy_delete(delete_ids[static_cast<size_t>(i)]);
      }
    });
    auto insert_begin = std::chrono::high_resolution_clock::now();
    auto insert_future = std::async(std::launch::async, [&]() {
#pragma omp parallel for num_threads(std::max<uint32_t>(1, insert_threads)) schedule(runtime)
      for (int64_t i = 0; i < static_cast<int64_t>(insert_ids.size()); ++i) {
        index.insert(insert_vectors.data() + static_cast<size_t>(i) * dim, insert_ids[static_cast<size_t>(i)]);
      }
    });
    delete_future.get();
    auto delete_end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Round " << round << " deletes complete";
    insert_future.get();
    auto insert_end = std::chrono::high_resolution_clock::now();
    LOG(INFO) << "Round " << round << " inserts complete";
    auto foreground_end = insert_end;
    if (delete_end > foreground_end) foreground_end = delete_end;
    pending_updates += static_cast<uint32_t>(delete_ids.size() + insert_ids.size());
    bool maintenance_triggered = (((round + 1) % std::max<uint32_t>(1, merge_round)) == 0);
    if (!maintenance_triggered && workload_has_query(workload) && ref_disk_ios > 0.0 &&
        last_query_disk_ios > 0.0) {
      maintenance_triggered =
          (last_query_disk_ios / ref_disk_ios) > merge_io_threshold;
    }
    auto maintenance_begin = std::chrono::high_resolution_clock::now();
    if (maintenance_triggered) {
      LOG(INFO) << "Round " << round << " final_merge begin";
      index.final_merge(merge_threads, merge_sampled_nbrs);
      LOG(INFO) << "Round " << round << " final_merge complete";
      pending_updates = 0;
    }
    auto maintenance_end = std::chrono::high_resolution_clock::now();
    auto round_end = maintenance_end;

    cumulative_inserts += static_cast<uint32_t>(insert_ids.size());
    cumulative_deletes += static_cast<uint32_t>(delete_ids.size());
    active_points = base_points + cumulative_inserts - cumulative_deletes;

    double delete_s = std::chrono::duration<double>(delete_end - delete_begin).count();
    double insert_s = std::chrono::duration<double>(insert_end - insert_begin).count();
    double foreground_s = std::chrono::duration<double>(foreground_end - round_begin).count();
    double maintenance_s = std::chrono::duration<double>(maintenance_end - maintenance_begin).count();
    double round_s = std::chrono::duration<double>(round_end - round_begin).count();
    BatchMetrics batch;
    batch.delete_throughput = delete_s > 0 ? static_cast<double>(delete_ids.size()) / delete_s : 0.0;
    batch.insert_throughput = insert_s > 0 ? static_cast<double>(insert_ids.size()) / insert_s : 0.0;
    batch.update_throughput = round_s > 0 ? static_cast<double>(delete_ids.size() + insert_ids.size()) / round_s : 0.0;
    batch.round_update_wall_time_s = round_s;
    SearchMetrics search;
    if (workload_has_query(workload) && workload != "query_only") {
      uint32_t gt_key = workload_has_inserts(workload) && !workload_has_deletes(workload)
                            ? cumulative_inserts
                            : cumulative_deletes;
      search = run_checkpoint_search(index, query_bin,
                                     gt_dir + "/gt_" + std::to_string(gt_key) + ".bin",
                                     recall_at, search_L, effective_mem_L, beamwidth, search_threads,
                                     query_schedule,
                                     warmup_enabled && warmup_before_each_checkpoint,
                                     warmup_query_count);
      last_query_disk_ios = search.disk_ios;
    }

    ProcIo io_cur = read_proc_io();
    uint64_t disk_size = pipeann_index_size_bytes(index._disk_index_prefix_in, index_prefix);
    uint64_t aux_size = pipeann_aux_index_artifact_bytes(index._disk_index_prefix_in, index_prefix);
    append_row(out, workload, round + 1, round + 1, base_points, active_points,
               cumulative_inserts, cumulative_deletes, recall_at, search_L,
               beamwidth, search, batch, disk_size, aux_size, io_start, io_prev, io_cur, current_mem_kb(),
               "native_online", maintenance_triggered, foreground_s, maintenance_s, pending_updates,
               workload == "update_only"
                   ? "maintenance_mode=native_online;recall_disabled;merge_policy=native_periodic_or_io"
                   : "maintenance_mode=native_online;merge_policy=native_periodic_or_io");
    io_prev = io_cur;
  }

  if (pending_updates > 0) {
    index.final_merge(merge_threads, merge_sampled_nbrs);
  }

  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 36) {
    std::cout << "Usage: " << argv[0]
              << " <type> <workload> <base_bin> <full_bin> <query_bin> <trace_prefix> <gt_dir>"
              << " <index_prefix> <result_jsonl> <tag_file> <base_points> <update_rounds>"
              << " <recall_at> <search_L> <mem_L> <beamwidth> <search_threads> <insert_threads>"
              << " <delete_threads> <query_schedule> <update_schedule> <warmup_enabled>"
              << " <warmup_query_count> <warmup_before_each_checkpoint> <R> <L> <pq_bytes>"
              << " <merge_threads> <merge_sampled_nbrs> <nbr_type> <mem_sample_rate> <merge_round> <merge_io_threshold> <query_only_checkpoints> <reuse_existing_index>"
              << std::endl;
    return -1;
  }

  std::string type = argv[1];
  std::string workload = argv[2];
  std::string base_bin = argv[3];
  std::string full_bin = argv[4];
  std::string query_bin = argv[5];
  std::string trace_prefix = argv[6];
  std::string gt_dir = argv[7];
  std::string index_prefix = argv[8];
  std::string result_jsonl = argv[9];
  std::string tag_file = argv[10];
  uint32_t base_points = static_cast<uint32_t>(std::stoul(argv[11]));
  uint32_t update_rounds = static_cast<uint32_t>(std::stoul(argv[12]));
  uint32_t recall_at = static_cast<uint32_t>(std::stoul(argv[13]));
  uint32_t search_L = static_cast<uint32_t>(std::stoul(argv[14]));
  uint32_t mem_L = static_cast<uint32_t>(std::stoul(argv[15]));
  uint32_t beamwidth = static_cast<uint32_t>(std::stoul(argv[16]));
  uint32_t search_threads = static_cast<uint32_t>(std::stoul(argv[17]));
  uint32_t insert_threads = static_cast<uint32_t>(std::stoul(argv[18]));
  uint32_t delete_threads = static_cast<uint32_t>(std::stoul(argv[19]));
  int query_schedule = parse_schedule(argv[20]);
  int update_schedule = parse_schedule(argv[21]);
  bool warmup_enabled = std::stoi(argv[22]) != 0;
  uint32_t warmup_query_count = static_cast<uint32_t>(std::stoul(argv[23]));
  bool warmup_before_each_checkpoint = std::stoi(argv[24]) != 0;
  uint32_t build_R = static_cast<uint32_t>(std::stoul(argv[25]));
  uint32_t build_L = static_cast<uint32_t>(std::stoul(argv[26]));
  uint32_t pq_bytes = static_cast<uint32_t>(std::stoul(argv[27]));
  uint32_t merge_threads = static_cast<uint32_t>(std::stoul(argv[28]));
  uint32_t merge_sampled_nbrs = static_cast<uint32_t>(std::stoul(argv[29]));
  std::string nbr_type = argv[30];
  double mem_sample_rate = std::stod(argv[31]);
  uint32_t merge_round = static_cast<uint32_t>(std::stoul(argv[32]));
  double merge_io_threshold = std::stod(argv[33]);
  uint32_t query_only_checkpoints = static_cast<uint32_t>(std::stoul(argv[34]));
  bool reuse_existing_index = std::stoi(argv[35]) != 0;

  try {
    if (type == "float") {
      return run_fair<float>(workload, base_bin, full_bin, query_bin, trace_prefix, gt_dir,
                             index_prefix, result_jsonl, tag_file, base_points, update_rounds,
                             recall_at, search_L, mem_L, beamwidth, search_threads, insert_threads,
                             delete_threads, query_schedule, update_schedule, warmup_enabled,
                             warmup_query_count, warmup_before_each_checkpoint,
                             build_R, build_L, pq_bytes, 1,
                             std::max({search_threads, insert_threads, delete_threads}),
                             merge_threads, merge_sampled_nbrs, nbr_type,
                             mem_sample_rate, query_only_checkpoints, reuse_existing_index,
                             merge_round, merge_io_threshold);
    }
    if (type == "int8") {
      return run_fair<int8_t>(workload, base_bin, full_bin, query_bin, trace_prefix, gt_dir,
                              index_prefix, result_jsonl, tag_file, base_points, update_rounds,
                              recall_at, search_L, mem_L, beamwidth, search_threads, insert_threads,
                              delete_threads, query_schedule, update_schedule, warmup_enabled,
                              warmup_query_count, warmup_before_each_checkpoint,
                              build_R, build_L, pq_bytes, 1,
                              std::max({search_threads, insert_threads, delete_threads}),
                              merge_threads, merge_sampled_nbrs, nbr_type,
                              mem_sample_rate, query_only_checkpoints, reuse_existing_index,
                              merge_round, merge_io_threshold);
    }
    if (type == "uint8") {
      return run_fair<uint8_t>(workload, base_bin, full_bin, query_bin, trace_prefix, gt_dir,
                               index_prefix, result_jsonl, tag_file, base_points, update_rounds,
                               recall_at, search_L, mem_L, beamwidth, search_threads, insert_threads,
                               delete_threads, query_schedule, update_schedule, warmup_enabled,
                               warmup_query_count, warmup_before_each_checkpoint,
                               build_R, build_L, pq_bytes, 1,
                               std::max({search_threads, insert_threads, delete_threads}),
                               merge_threads, merge_sampled_nbrs, nbr_type,
                               mem_sample_rate, query_only_checkpoints, reuse_existing_index,
                               merge_round, merge_io_threshold);
    }
    return -1;
  } catch (const std::exception &e) {
    std::cerr << "benchmark_adapter failed: " << e.what() << std::endl;
    return -1;
  }
}
