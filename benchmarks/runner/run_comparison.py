#!/usr/bin/env python3
import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import os
import random
import shutil
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


RUNNER_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = RUNNER_DIR.parent
PAGEANN_ROOT = BENCHMARK_ROOT.parent
WORKSPACE_ROOT = Path(os.environ.get("ANN_WORKSPACE_ROOT", str(PAGEANN_ROOT.parent))).resolve()
THIS_DIR = BENCHMARK_ROOT
CONFIG_DIR = BENCHMARK_ROOT / "config"
INDICES_ROOT = Path(os.environ.get("ANN_INDICES_ROOT", "/anns/indices/pageann-benchmarks"))
SHARED_TAG_ROOT = Path(os.environ.get("ANN_SHARED_TAG_ROOT", "/anns/indices/shared-tags"))
SHARED_PREPARED_ROOT = Path(os.environ.get("ANN_SHARED_PREPARED_ROOT", str(INDICES_ROOT / "shared-prepared")))
SHARED_INDEX_ROOT = Path(os.environ.get("ANN_SHARED_INDEX_ROOT", str(INDICES_ROOT / "shared-indices")))
LATEST_DIR = THIS_DIR / "latest"

DTYPE_MAP = {
    "float": np.float32,
    "uint8": np.uint8,
    "int8": np.int8,
}

COMMON_FIELDS = [
    "run_id",
    "system",
    "system_label",
    "adapter_label",
    "dataset",
    "profile",
    "workload",
    "status",
    "checkpoint_id",
    "round_index",
    "base_points",
    "active_points",
    "query_count",
    "source_points",
    "update_rounds",
    "updates_per_round",
    "trace_mode",
    "base_fraction",
    "update_fraction_per_round",
    "trace_seed",
    "trace_continue_mode",
    "cumulative_inserts",
    "cumulative_deletes",
    "query_threads",
    "insert_threads",
    "delete_threads",
    "update_threads",
    "background_threads",
    "warmup_enabled",
    "warmup_query_count",
    "search_L",
    "beamwidth",
    "buffer_pool_frames",
    "page_size",
    "insert_search_L",
    "prune_R",
    "prune_C",
    "entry_init_mode",
    "query_pool_mode",
    "query_pool_slack",
    "candidate_pool_L_effective",
    "pq_frontier_confirm_topk_effective",
    "qps",
    "query_throughput",
    "insert_throughput",
    "delete_throughput",
    "update_throughput",
    "query_wall_time_s",
    "foreground_update_wall_time_s",
    "maintenance_wall_time_s",
    "round_update_wall_time_s",
    "checkpoint_wall_time_s",
    "local_elapsed_time_s",
    "elapsed_time_s",
    "lat_avg_us",
    "lat_p50_us",
    "lat_p95_us",
    "lat_p99_us",
    "recall",
    "recall_at",
    "disk_ios",
    "disk_ios_native",
    "query_io_us_native",
    "rss_kb",
    "peak_rss_kb",
    "disk_index_size_bytes",
    "aux_index_artifact_bytes",
    "index_num_nodes",
    "native_index_bytes_read",
    "native_index_bytes_written",
    "cumulative_native_index_bytes_read",
    "cumulative_native_index_bytes_written",
    "native_pages_flushed",
    "cumulative_native_pages_flushed",
    "native_write_amplification",
    "query_cache_hits",
    "query_cache_misses",
    "query_cache_hit_rate",
    "query_distance_us",
    "query_distance_ops",
    "command",
    "log_path",
    "result_path",
    "maintenance_policy",
    "maintenance_triggered",
    "pending_updates",
    "overlay_pending_edges",
    "overlay_pending_targets",
    "repair_queue_backlog",
    "oversized_node_backlog",
    "dirty_page_count",
    "tombstone_count",
    "unsupported_reason",
    "notes",
]

WORKLOAD_SPLITS = {
    "query_only": "query_only",
    "update_only": "update_only",
    "query_update_round": "mix",
    "query_insert_round": "query_insert",
    "query_delete_round": "query_delete",
}

WORKLOAD_TITLES = {
    "query_only": "Query Only",
    "update_only": "Update Only",
    "mix": "Mix (query_update_round)",
    "query_insert": "Query + Insert Only",
    "query_delete": "Query + Delete Only",
}


@dataclass
class PreparedAssets:
    dataset_name: str
    dataset_dir: Path
    full_bin: Path
    base_bin: Path
    query_bin: Path
    full_tags: Path
    base_tags: Path
    trace_mix_prefix: Path
    trace_insert_prefix: Path
    trace_delete_prefix: Path
    greator_gt_mix_prefix: Path
    greator_gt_insert_prefix: Path
    greator_gt_delete_prefix: Path
    odin_gt_mix_dir: Path
    odin_gt_insert_dir: Path
    odin_gt_delete_dir: Path
    source_points: int
    total_points: int
    base_points: int
    query_count: int
    update_rounds: int
    updates_per_round: int
    recall_at: int
    dtype_name: str
    metric: str
    trace_mode: str
    base_fraction: float
    update_fraction_per_round: float
    trace_seed: int
    trace_continue_mode: str
    cache_reused: bool


def merged_cfg(profile_cfg: Dict, system: str) -> Dict:
    merged = dict(profile_cfg)
    merged.update(profile_cfg.get(system, {}))
    return merged


def deep_merge_dict(dst: Dict, src: Dict) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_merge_dict(dst[key], value)
        else:
            dst[key] = value


def deep_fill_dict(dst: Dict, src: Dict) -> None:
    for key, value in src.items():
        if isinstance(value, dict):
            child = dst.get(key)
            if not isinstance(child, dict):
                dst[key] = json.loads(json.dumps(value))
            else:
                deep_fill_dict(child, value)
        elif key not in dst or dst[key] is None:
            dst[key] = value


def effective_search_cfg(cfg: Dict, system: str) -> Dict[str, int]:
    params = merged_cfg(cfg, system)
    return {
        "search_L": int(params.get("search_L_override", cfg["search_L"])),
        "beamwidth": int(params.get("beamwidth_override", cfg["beamwidth"])),
        "pq_bytes": int(params.get("pq_bytes_override", params.get("pq_bytes", cfg.get("pq_bytes", 0)))),
    }


def dataset_profile_cfg(profile_cfg: Dict, dataset_name: str, dataset_cfg: Dict) -> Dict:
    merged = json.loads(json.dumps(profile_cfg))
    if not profile_cfg.get("use_dataset_defaults", False):
        merged["_dataset_name"] = dataset_name
        return merged
    scalar_map = {
        "base_points_default": "base_points",
        "query_count_default": "query_count",
        "update_rounds_default": "update_rounds",
        "updates_per_round_default": "updates_per_round",
        "search_L_default": "search_L",
        "beamwidth_default": "beamwidth",
        "nodes_to_cache_default": "nodes_to_cache",
        "query_threads_default": "query_threads",
        "insert_threads_default": "insert_threads",
        "delete_threads_default": "delete_threads",
        "build_threads_default": "build_threads",
        "pq_bytes_default": "pq_bytes",
    }
    trace_mode = ((merged.get("workload_policy") or {}).get("trace_mode") or "").strip()
    for src, dst in scalar_map.items():
        if merged.get(dst) is not None:
            continue
        if src in dataset_cfg and dataset_cfg[src] is not None:
            merged[dst] = dataset_cfg[src]
    if trace_mode == "active_inactive_pool":
        merged["base_points"] = profile_cfg.get("base_points")
        merged["updates_per_round"] = profile_cfg.get("updates_per_round")
    merged.setdefault("inplace", {})
    if "inplace_buffer_pool_frames_default" in dataset_cfg and merged["inplace"].get("buffer_pool_frames") is None:
        merged.setdefault("inplace", {})
        merged["inplace"]["buffer_pool_frames"] = dataset_cfg["inplace_buffer_pool_frames_default"]
    if "inplace_buffer_pool_frames_query_default" in dataset_cfg and merged["inplace"].get("buffer_pool_frames_query") is None:
        merged["inplace"]["buffer_pool_frames_query"] = dataset_cfg["inplace_buffer_pool_frames_query_default"]
    if "inplace_buffer_pool_frames_update_default" in dataset_cfg and merged["inplace"].get("buffer_pool_frames_update") is None:
        merged["inplace"]["buffer_pool_frames_update"] = dataset_cfg["inplace_buffer_pool_frames_update_default"]
    if "inplace_buffer_pool_frames_mix_default" in dataset_cfg and merged["inplace"].get("buffer_pool_frames_mix") is None:
        merged["inplace"]["buffer_pool_frames_mix"] = dataset_cfg["inplace_buffer_pool_frames_mix_default"]
    if "odin_mem_L_default" in dataset_cfg and merged.setdefault("odinann", {}).get("mem_L") is None:
        merged.setdefault("odinann", {})
        merged["odinann"]["mem_L"] = dataset_cfg["odin_mem_L_default"]
    if "build" in dataset_cfg:
        build = dataset_cfg["build"]
        if "R" in build and merged.get("build_R") is None:
            merged["build_R"] = build["R"]
        if "L" in build and merged.get("build_L") is None:
            merged["build_L"] = build["L"]
        if "B" in build and merged.setdefault("disk_build", {}).get("B") is None:
            merged.setdefault("disk_build", {})
            merged["disk_build"]["B"] = build["B"]
        if "M" in build and merged.setdefault("disk_build", {}).get("M") is None:
            merged.setdefault("disk_build", {})
            merged["disk_build"]["M"] = build["M"]
        if "T" in build and merged.setdefault("disk_build", {}).get("T") is None:
            merged.setdefault("disk_build", {})
            merged["disk_build"]["T"] = build["T"]
    if dataset_cfg.get("k") is not None and merged.get("recall_at") is None:
        merged["recall_at"] = dataset_cfg["k"]
    if "system_defaults" in dataset_cfg:
        for system, defaults in dataset_cfg["system_defaults"].items():
            merged.setdefault(system, {})
            deep_fill_dict(merged[system], defaults)
    profile_dataset_overrides = (profile_cfg.get("dataset_overrides") or {}).get(dataset_name)
    if profile_dataset_overrides:
        deep_merge_dict(merged, profile_dataset_overrides)
    merged["_dataset_name"] = dataset_name
    return merged


def workload_has_query(workload: str) -> bool:
    return workload in {"query_only", "query_update_round", "query_insert_round", "query_delete_round"}


def workload_has_updates(workload: str) -> bool:
    return workload in {"update_only", "query_update_round", "query_insert_round", "query_delete_round"}


def workload_has_inserts(workload: str) -> bool:
    return workload in {"update_only", "query_update_round", "query_insert_round"}


def workload_has_deletes(workload: str) -> bool:
    return workload in {"update_only", "query_update_round", "query_delete_round"}


def inplace_buffer_pool_frames_for_workload(params: Dict, workload: str) -> int:
    if workload == "query_only" and "buffer_pool_frames_query" in params:
        return int(params["buffer_pool_frames_query"])
    if workload == "update_only" and "buffer_pool_frames_update" in params:
        return int(params["buffer_pool_frames_update"])
    if workload in {"query_update_round", "query_insert_round", "query_delete_round"} and "buffer_pool_frames_mix" in params:
        return int(params["buffer_pool_frames_mix"])
    return int(params.get("buffer_pool_frames", 64))


def read_bin_metadata(path: Path) -> Tuple[int, int]:
    with open(path, "rb") as fh:
        header = np.fromfile(fh, dtype=np.int32, count=2)
    if header.size != 2:
        raise ValueError(f"Invalid bin header: {path}")
    return int(header[0]), int(header[1])


def stream_copy_bin_prefix(source: Path, dest: Path, dtype_name: str, count: int) -> None:
    total_points, dim = read_bin_metadata(source)
    if count > total_points:
        raise ValueError(f"Requested {count} points from {source}, only {total_points} available")
    ensure_dir(dest.parent)
    dtype = np.dtype(DTYPE_MAP[dtype_name])
    chunk_points = max(1, (64 * 1024 * 1024) // max(1, dim * dtype.itemsize))
    with open(source, "rb") as src, open(dest, "wb") as out:
        src.seek(2 * np.dtype(np.int32).itemsize, os.SEEK_SET)
        np.array([count, dim], dtype=np.int32).tofile(out)
        remaining = count
        while remaining > 0:
            take = min(remaining, chunk_points)
            chunk = np.fromfile(src, dtype=dtype, count=take * dim)
            if chunk.size != take * dim:
                raise ValueError(f"Unexpected EOF while copying {source}")
            chunk.tofile(out)
            remaining -= take


def schedule_token(schedule: str) -> str:
    schedule = (schedule or "static").lower()
    if schedule not in {"static", "dynamic"}:
        raise ValueError(f"Unsupported schedule: {schedule}")
    return schedule


def project_root(project: str) -> Path:
    if project == "PageANN":
        return PAGEANN_ROOT
    return WORKSPACE_ROOT / project


def build_dir_for(project: str, build_type: str) -> Path:
    return project_root(project) / "build"


def build_cmake_args(project: str, build_type: str) -> List[str]:
    args = [f"-DCMAKE_BUILD_TYPE={build_type}"]
    if project == "PipeANN":
        args.append("-DUSE_AIO=ON")
    return args


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_bin(path: Path, dtype_name: str, limit: Optional[int] = None) -> np.ndarray:
    dtype = np.dtype(DTYPE_MAP[dtype_name])
    with open(path, "rb") as fh:
        header = np.fromfile(fh, dtype=np.int32, count=2)
        if header.size != 2:
            raise ValueError(f"Invalid bin header: {path}")
        npts, dim = int(header[0]), int(header[1])
        take = npts if limit is None else min(npts, limit)
        data = np.fromfile(fh, dtype=dtype, count=take * dim)
    if data.size != take * dim:
        raise ValueError(f"Unexpected payload size in {path}")
    return data.reshape(take, dim)


def write_bin(path: Path, data: np.ndarray) -> None:
    ensure_dir(path.parent)
    arr = np.ascontiguousarray(data)
    with open(path, "wb") as fh:
        np.array([arr.shape[0], arr.shape[1]], dtype=np.int32).tofile(fh)
        arr.tofile(fh)


def write_trace(path: Path, delete_ids: np.ndarray, insert_ids: np.ndarray) -> None:
    ensure_dir(path.parent)
    with open(path, "wb") as fh:
        np.array([delete_ids.shape[0]], dtype=np.int32).tofile(fh)
        delete_ids.astype(np.int32, copy=False).tofile(fh)
        insert_ids.astype(np.int32, copy=False).tofile(fh)


def write_truthset(path: Path, ids: np.ndarray, dists: np.ndarray) -> None:
    ensure_dir(path.parent)
    with open(path, "wb") as fh:
        np.array([ids.shape[0], ids.shape[1]], dtype=np.int32).tofile(fh)
        ids.astype(np.uint32, copy=False).tofile(fh)
        dists.astype(np.float32, copy=False).tofile(fh)


def ensure_tags_file(path: Path, count: int) -> Path:
    if path.exists():
        return path
    ensure_dir(path.parent)
    with open(path, "wb") as fh:
        np.array([count, 1], dtype=np.int32).tofile(fh)
        np.arange(count, dtype=np.uint32).tofile(fh)
    return path


def compute_l2_truth(full_vectors: np.ndarray, active_ids: np.ndarray,
                     queries: np.ndarray, topk: int, chunk_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    active = full_vectors[active_ids].astype(np.float32, copy=False)
    queries_f = queries.astype(np.float32, copy=False)
    q_sq = np.sum(queries_f * queries_f, axis=1, dtype=np.float64)
    top_ids = np.full((queries.shape[0], topk), -1, dtype=np.int64)
    top_d = np.full((queries.shape[0], topk), np.inf, dtype=np.float64)
    for start in range(0, active.shape[0], chunk_size):
        end = min(start + chunk_size, active.shape[0])
        chunk = active[start:end]
        b_sq = np.sum(chunk * chunk, axis=1, dtype=np.float64)
        cross = queries_f @ chunk.T
        dists = q_sq[:, None] + b_sq[None, :] - 2.0 * cross.astype(np.float64)
        dists = np.maximum(dists, 0.0)
        ids = active_ids[start:end]
        combined_d = np.concatenate([top_d, dists], axis=1)
        tiled_ids = np.broadcast_to(ids[None, :], dists.shape)
        combined_ids = np.concatenate([top_ids, tiled_ids], axis=1)
        idx = np.argpartition(combined_d, kth=topk - 1, axis=1)[:, :topk]
        row_ids = np.arange(queries.shape[0])[:, None]
        sel_d = combined_d[row_ids, idx]
        order = np.argsort(sel_d, axis=1)
        top_d = sel_d[row_ids, order]
        top_ids = combined_ids[row_ids, idx[row_ids, order]]
    return top_ids.astype(np.uint32), top_d.astype(np.float32)


def system_supports(system: str, workload: str) -> bool:
    supported = {
        "inplace": {"query_only", "update_only", "query_update_round", "query_insert_round", "query_delete_round"},
        "diskann": {"query_only", "update_only", "query_update_round", "query_insert_round", "query_delete_round"},
        "greator": {"query_only", "update_only", "query_update_round", "query_insert_round", "query_delete_round"},
        "odinann": {"query_only", "update_only", "query_update_round", "query_insert_round", "query_delete_round"},
    }
    return workload in supported.get(system, set())


def stable_digest(payload: Dict) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def ensure_parent(path: Path) -> None:
    ensure_dir(path.parent)


def remove_prefix_family(prefix: Path) -> None:
    parent = prefix.parent
    if not parent.exists():
        return
    for path in parent.glob(prefix.name + "*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def copy_prefix_family(src_prefix: Path, dst_prefix: Path) -> None:
    ensure_dir(dst_prefix.parent)
    remove_prefix_family(dst_prefix)
    for path in src_prefix.parent.glob(src_prefix.name + "*"):
        dst = dst_prefix.parent / path.name.replace(src_prefix.name, dst_prefix.name, 1)
        if path.is_dir():
            shutil.copytree(path, dst)
        else:
            shutil.copy2(path, dst)


def copy_inplace_cache(src_heap: Path, dst_heap: Path) -> None:
    ensure_dir(dst_heap.parent)
    for src, dst in (
        (src_heap, dst_heap),
        (Path(str(src_heap) + ".meta"), Path(str(dst_heap) + ".meta")),
        (Path(str(src_heap) + ".medoids"), Path(str(dst_heap) + ".medoids")),
    ):
        if not src.exists():
            continue
        ensure_parent(dst)
        shutil.copy2(src, dst)


def remove_inplace_cache(heap_path: Path) -> None:
    for path in (
        heap_path,
        Path(str(heap_path) + ".meta"),
        Path(str(heap_path) + ".medoids"),
    ):
        if path.exists():
            path.unlink()


def clean_odinann_runtime_artifacts(prefix: Path) -> None:
    for path in prefix.parent.glob(prefix.name + "_shadow*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def copy_odinann_cache(src_prefix: Path, dst_prefix: Path, include_mem_index: bool) -> None:
    ensure_dir(dst_prefix.parent)
    remove_prefix_family(dst_prefix)
    canonical_suffixes = [
        "_disk.index",
        "_disk.index.tags",
        "_pq_compressed.bin",
        "_pq_pivots.bin",
    ]
    if include_mem_index:
        canonical_suffixes.extend([
            "_mem.index",
            "_mem.index.data",
            "_mem.index.tags",
        ])
    for suffix in canonical_suffixes:
        src = src_prefix.parent / f"{src_prefix.name}{suffix}"
        dst = dst_prefix.parent / f"{dst_prefix.name}{suffix}"
        if src.exists():
            shutil.copy2(src, dst)


def diskann_cache_ready(prefix: Path) -> bool:
    required = [
        prefix.parent / f"{prefix.name}_disk.index",
        prefix.parent / f"{prefix.name}_disk.index.tags",
        prefix.parent / f"{prefix.name}_pq_compressed.bin",
        prefix.parent / f"{prefix.name}_pq_pivots.bin",
        prefix.parent / f"{prefix.name}_sample_ids.bin",
        prefix.parent / f"{prefix.name}_sample_data.bin",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def greator_cache_ready(prefix: Path, id_map: int) -> bool:
    required = [
        prefix.parent / f"{prefix.name}_disk.index",
        prefix.parent / f"{prefix.name}_disk.index.tags",
        prefix.parent / f"{prefix.name}_pq_compressed.bin",
        prefix.parent / f"{prefix.name}_pq_pivots.bin",
        prefix.parent / f"{prefix.name}_sample_ids.bin",
        prefix.parent / f"{prefix.name}_sample_data.bin",
    ]
    if id_map == 2:
        required.append(prefix.parent / f"{prefix.name}_disk.index_with_only_nbrs")
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def odinann_cache_ready(prefix: Path, use_mem_index: bool) -> bool:
    required = [
        prefix.parent / f"{prefix.name}_disk.index",
        prefix.parent / f"{prefix.name}_disk.index.tags",
        prefix.parent / f"{prefix.name}_pq_compressed.bin",
        prefix.parent / f"{prefix.name}_pq_pivots.bin",
    ]
    if use_mem_index:
        required.extend([
            prefix.parent / f"{prefix.name}_mem.index",
            prefix.parent / f"{prefix.name}_mem.index.data",
            prefix.parent / f"{prefix.name}_mem.index.tags",
        ])
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def inplace_cache_ready(heap_path: Path) -> bool:
    required = [
        heap_path,
        Path(str(heap_path) + ".meta"),
        Path(str(heap_path) + ".medoids"),
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def shared_index_prefix(system: str, dataset_name: str, payload: Dict, name: str = "index") -> Path:
    key = stable_digest(payload)
    return SHARED_INDEX_ROOT / system / dataset_name / key / name


@contextlib.contextmanager
def cache_build_lock(lock_path: Path):
    ensure_parent(lock_path)
    with open(lock_path, "w", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def greator_gt_for_workload(assets: PreparedAssets, workload: str) -> Path:
    if workload == "query_insert_round":
        return assets.greator_gt_insert_prefix
    if workload == "query_delete_round":
        return assets.greator_gt_delete_prefix
    return assets.greator_gt_mix_prefix


def odin_gt_for_workload(assets: PreparedAssets, workload: str) -> Path:
    if workload == "query_insert_round":
        return assets.odin_gt_insert_dir
    if workload == "query_delete_round":
        return assets.odin_gt_delete_dir
    return assets.odin_gt_mix_dir


def trace_prefix_for_workload(assets: PreparedAssets, workload: str) -> Path:
    if workload == "query_insert_round":
        return assets.trace_insert_prefix
    if workload == "query_delete_round":
        return assets.trace_delete_prefix
    return assets.trace_mix_prefix


def effective_workload_policy(profile_cfg: Dict) -> Dict:
    policy = {
        "trace_mode": "contiguous_prefix",
        "base_fraction": 0.9,
        "updates_per_round_fraction": 0.01,
        "trace_seed": 12345,
        "continue_mode": "recycle_deletes",
    }
    deep_merge_dict(policy, profile_cfg.get("workload_policy", {}))
    return policy


def prepared_profile_key(payload: Dict) -> str:
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return (f"bp{payload['base_points']}_q{payload['query_count']}_r{payload['update_rounds']}"
            f"_u{payload['updates_per_round']}_k{payload['recall_at']}_{digest}")


def prepared_required_paths(
    source_base: Path,
    source_query: Path,
    full_bin_cache: Path,
    base_bin_cache: Path,
    query_bin_cache: Path,
    expected_manifest: Dict,
    trace_mix_prefix: Path,
    trace_insert_prefix: Path,
    trace_delete_prefix: Path,
    greator_gt_mix_prefix: Path,
    greator_gt_insert_prefix: Path,
    greator_gt_delete_prefix: Path,
    odin_gt_mix_dir: Path,
    odin_gt_insert_dir: Path,
    odin_gt_delete_dir: Path,
    update_rounds: int,
    updates_per_round: int,
    full_tags: Path,
    base_tags: Path,
) -> List[Path]:
    required = [
        source_base if expected_manifest["full_bin_mode"] == "source" else full_bin_cache,
        source_base if expected_manifest["base_bin_mode"] == "source" else base_bin_cache,
        source_query if expected_manifest["query_bin_mode"] == "source" else query_bin_cache,
        full_tags,
        base_tags,
    ]
    for step in range(update_rounds):
        required.extend([
            Path(f"{trace_mix_prefix}{step}"),
            Path(f"{trace_insert_prefix}{step}"),
            Path(f"{trace_delete_prefix}{step}"),
        ])
    for step in range(update_rounds + 1):
        offset = step * updates_per_round
        required.extend([
            Path(f"{greator_gt_mix_prefix}{step}.fbin"),
            Path(f"{greator_gt_insert_prefix}{step}.fbin"),
            Path(f"{greator_gt_delete_prefix}{step}.fbin"),
            odin_gt_mix_dir / f"gt_{offset}.bin",
            odin_gt_insert_dir / f"gt_{offset}.bin",
            odin_gt_delete_dir / f"gt_{offset}.bin",
        ])
    return required


def sample_without_replacement(pool: np.ndarray, count: int,
                               rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if count < 0:
        raise ValueError(f"Invalid sample size: {count}")
    if count == 0:
        return np.empty(0, dtype=np.uint32), pool
    if count > pool.shape[0]:
        raise ValueError(f"Cannot sample {count} ids from pool of size {pool.shape[0]}")
    idx = rng.choice(pool.shape[0], size=count, replace=False)
    mask = np.ones(pool.shape[0], dtype=bool)
    mask[idx] = False
    return pool[idx].astype(np.uint32, copy=False), pool[mask].astype(np.uint32, copy=False)


def write_workload_traces_and_states(
    trace_prefix: Path,
    initial_active: np.ndarray,
    initial_inactive: np.ndarray,
    update_rounds: int,
    updates_per_round: int,
    rng: np.random.Generator,
    mode: str,
) -> List[np.ndarray]:
    active = initial_active.astype(np.uint32, copy=True)
    inactive = initial_inactive.astype(np.uint32, copy=True)
    states = [active.copy()]

    for step in range(update_rounds):
        if mode == "mix":
            delete_ids, active_after_delete = sample_without_replacement(active, updates_per_round, rng)
            insert_ids, inactive_after_insert = sample_without_replacement(inactive, updates_per_round, rng)
            active = np.concatenate([active_after_delete, insert_ids]).astype(np.uint32, copy=False)
            inactive = np.concatenate([inactive_after_insert, delete_ids]).astype(np.uint32, copy=False)
        elif mode == "insert_only":
            insert_ids, inactive = sample_without_replacement(inactive, updates_per_round, rng)
            delete_ids = insert_ids.copy()
            active = np.concatenate([active, insert_ids]).astype(np.uint32, copy=False)
        elif mode == "delete_only":
            delete_ids, active = sample_without_replacement(active, updates_per_round, rng)
            insert_ids = delete_ids.copy()
        else:
            raise ValueError(f"Unsupported trace generation mode: {mode}")
        write_trace(Path(f"{trace_prefix}{step}"), delete_ids, insert_ids)
        states.append(active.copy())

    return states


def prepare_assets(dataset_name: str, dataset_cfg: Dict, profile_cfg: Dict) -> PreparedAssets:
    dtype_name = dataset_cfg["type"]
    metric = dataset_cfg.get("similarity", "l2")
    workload_policy = effective_workload_policy(profile_cfg)
    trace_mode = str(workload_policy.get("trace_mode", "active_inactive_pool"))
    trace_seed = int(workload_policy.get("trace_seed", 12345))
    trace_continue_mode = str(workload_policy.get("continue_mode", "recycle_deletes"))
    if trace_mode != "active_inactive_pool":
        trace_continue_mode = "none"
    query_count = int(profile_cfg["query_count"])
    update_rounds = int(profile_cfg["update_rounds"])
    recall_at = int(profile_cfg.get("recall_at", 10))

    source_base = Path(dataset_cfg["data_file"])
    source_query = Path(dataset_cfg["query_file"])
    total_source_points, _ = read_bin_metadata(source_base)
    explicit_base_points = profile_cfg.get("base_points")
    explicit_updates_per_round = profile_cfg.get("updates_per_round")
    if explicit_base_points is None:
        base_fraction = float(workload_policy.get("base_fraction", 0.9))
        base_points = max(1, min(total_source_points - 1, int(total_source_points * base_fraction)))
    else:
        base_points = int(explicit_base_points)
        base_fraction = float(base_points) / float(total_source_points)
    if explicit_updates_per_round is None:
        update_fraction_per_round = float(workload_policy.get("updates_per_round_fraction", 0.01))
        updates_per_round = max(1, int(total_source_points * update_fraction_per_round))
    else:
        updates_per_round = int(explicit_updates_per_round)
        update_fraction_per_round = float(updates_per_round) / float(total_source_points)
    total_updates = update_rounds * updates_per_round
    if base_points <= 0 or base_points >= total_source_points:
        raise ValueError(
            f"Invalid base_points({base_points}) for dataset of size {total_source_points}: {dataset_name}"
        )
    if updates_per_round <= 0:
        raise ValueError(f"updates_per_round must be positive for {dataset_name}")
    if trace_mode == "active_inactive_pool":
        if updates_per_round > base_points:
            raise ValueError(
                f"updates_per_round({updates_per_round}) exceeds active base_points({base_points}) for {dataset_name}"
            )
        total_needed = total_source_points
        full_dataset = True
    else:
        if total_source_points <= total_updates:
            raise ValueError("Not enough source vectors for requested update workload")
        total_needed = base_points + total_updates
        full_dataset = total_needed == total_source_points
        if total_needed > total_source_points:
            raise ValueError(
                f"Requested base_points({base_points}) + total_updates({total_updates}) "
                f"exceeds dataset size {total_source_points} for {dataset_name}"
            )
    source_query_points, _ = read_bin_metadata(source_query)
    if query_count > source_query_points:
        raise ValueError(f"Requested {query_count} queries but only {source_query_points} are available")

    prepared_root = Path(profile_cfg.get("prepared_cache_root", SHARED_PREPARED_ROOT))
    dataset_alias = dataset_cfg.get("prepared_alias", dataset_name)
    profile_key_payload = {
        "dataset": dataset_alias,
        "dtype": dtype_name,
        "metric": metric,
        "base_points": base_points,
        "query_count": query_count,
        "update_rounds": update_rounds,
        "updates_per_round": updates_per_round,
        "recall_at": recall_at,
        "trace_mode": trace_mode,
        "trace_seed": trace_seed,
        "trace_continue_mode": trace_continue_mode,
        "base_fraction": round(base_fraction, 8),
        "update_fraction_per_round": round(update_fraction_per_round, 8),
    }
    profile_key = prepared_profile_key(profile_key_payload)
    dataset_dir = prepared_root / dataset_alias / profile_key
    full_bin_cache = dataset_dir / f"{dataset_name}_full.bin"
    base_bin_cache = dataset_dir / f"{dataset_name}_base.bin"
    query_bin_cache = dataset_dir / f"{dataset_name}_query.bin"
    trace_mix_prefix = dataset_dir / "traces_mix" / "trace_"
    trace_insert_prefix = dataset_dir / "traces_insert" / "trace_"
    trace_delete_prefix = dataset_dir / "traces_delete" / "trace_"
    greator_gt_mix_prefix = dataset_dir / "greator_gt_mix" / "gt_"
    greator_gt_insert_prefix = dataset_dir / "greator_gt_insert" / "gt_"
    greator_gt_delete_prefix = dataset_dir / "greator_gt_delete" / "gt_"
    odin_gt_mix_dir = dataset_dir / "odin_gt_mix"
    odin_gt_insert_dir = dataset_dir / "odin_gt_insert"
    odin_gt_delete_dir = dataset_dir / "odin_gt_delete"
    full_tags = ensure_tags_file(SHARED_TAG_ROOT / f"{dataset_name}_{total_needed}.tags",
                                 total_needed)
    base_tags = ensure_tags_file(SHARED_TAG_ROOT / f"{dataset_name}_{base_points}.tags", base_points)

    manifest_path = dataset_dir / "manifest.json"
    expected_manifest = {
        "dataset": dataset_name,
        "dataset_alias": dataset_alias,
        "base_points": base_points,
        "full_dataset": full_dataset,
        "source_points": total_source_points,
        "query_count": query_count,
        "update_rounds": update_rounds,
        "updates_per_round": updates_per_round,
        "recall_at": recall_at,
        "dtype_name": dtype_name,
        "metric": metric,
        "trace_mode": trace_mode,
        "trace_seed": trace_seed,
        "trace_continue_mode": trace_continue_mode,
        "base_fraction": base_fraction,
        "update_fraction_per_round": update_fraction_per_round,
        "trace_mix_prefix": str(trace_mix_prefix),
        "trace_insert_prefix": str(trace_insert_prefix),
        "trace_delete_prefix": str(trace_delete_prefix),
        "greator_gt_mix_prefix": str(greator_gt_mix_prefix),
        "greator_gt_insert_prefix": str(greator_gt_insert_prefix),
        "greator_gt_delete_prefix": str(greator_gt_delete_prefix),
        "odin_gt_mix_dir": str(odin_gt_mix_dir),
        "odin_gt_insert_dir": str(odin_gt_insert_dir),
        "odin_gt_delete_dir": str(odin_gt_delete_dir),
        "full_bin_mode": "source" if total_needed == total_source_points else "cached",
        "base_bin_mode": "source" if base_points == total_source_points else "cached",
        "query_bin_mode": "source" if query_count == source_query_points else "cached",
        "base_tags": str(base_tags),
        "full_tags": str(full_tags),
    }
    if not profile_cfg.get("force_rebuild_prepared", False) and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as fh:
            existing = json.load(fh)
        required_paths = prepared_required_paths(
            source_base,
            source_query,
            full_bin_cache,
            base_bin_cache,
            query_bin_cache,
            expected_manifest,
            trace_mix_prefix,
            trace_insert_prefix,
            trace_delete_prefix,
            greator_gt_mix_prefix,
            greator_gt_insert_prefix,
            greator_gt_delete_prefix,
            odin_gt_mix_dir,
            odin_gt_insert_dir,
            odin_gt_delete_dir,
            update_rounds,
            updates_per_round,
            full_tags,
            base_tags,
        )
        if existing == expected_manifest and all(path.exists() and path.stat().st_size > 0 for path in required_paths):
            return PreparedAssets(
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                full_bin=source_base if expected_manifest["full_bin_mode"] == "source" else full_bin_cache,
                base_bin=source_base if expected_manifest["base_bin_mode"] == "source" else base_bin_cache,
                query_bin=source_query if expected_manifest["query_bin_mode"] == "source" else query_bin_cache,
                full_tags=full_tags,
                base_tags=base_tags,
                trace_mix_prefix=trace_mix_prefix,
                trace_insert_prefix=trace_insert_prefix,
                trace_delete_prefix=trace_delete_prefix,
                greator_gt_mix_prefix=greator_gt_mix_prefix,
                greator_gt_insert_prefix=greator_gt_insert_prefix,
                greator_gt_delete_prefix=greator_gt_delete_prefix,
                odin_gt_mix_dir=odin_gt_mix_dir,
                odin_gt_insert_dir=odin_gt_insert_dir,
                odin_gt_delete_dir=odin_gt_delete_dir,
                source_points=total_source_points,
                total_points=total_needed,
                base_points=base_points,
                query_count=query_count,
                update_rounds=update_rounds,
                updates_per_round=updates_per_round,
                recall_at=recall_at,
                dtype_name=dtype_name,
                metric=metric,
                trace_mode=trace_mode,
                base_fraction=base_fraction,
                update_fraction_per_round=update_fraction_per_round,
                trace_seed=trace_seed,
                trace_continue_mode=trace_continue_mode,
                cache_reused=True,
            )

    full_bin = source_base if total_needed == total_source_points else full_bin_cache
    base_bin = source_base if base_points == total_source_points else base_bin_cache
    query_bin = source_query if query_count == source_query_points else query_bin_cache

    if full_bin == full_bin_cache:
        stream_copy_bin_prefix(source_base, full_bin_cache, dtype_name, total_needed)
    if base_bin == base_bin_cache:
        stream_copy_bin_prefix(source_base, base_bin_cache, dtype_name, base_points)
    if query_bin == query_bin_cache:
        stream_copy_bin_prefix(source_query, query_bin_cache, dtype_name, query_count)

    full_vectors = read_bin(full_bin, dtype_name, total_needed)
    queries = read_bin(query_bin, dtype_name, query_count)

    if trace_mode == "active_inactive_pool":
        initial_active = np.arange(base_points, dtype=np.uint32)
        initial_inactive = np.arange(base_points, total_source_points, dtype=np.uint32)
        mix_states = write_workload_traces_and_states(
            trace_mix_prefix,
            initial_active,
            initial_inactive,
            update_rounds,
            updates_per_round,
            np.random.default_rng(trace_seed),
            "mix",
        )
        insert_states = write_workload_traces_and_states(
            trace_insert_prefix,
            initial_active,
            initial_inactive,
            update_rounds,
            updates_per_round,
            np.random.default_rng(trace_seed + 1),
            "insert_only",
        )
        delete_states = write_workload_traces_and_states(
            trace_delete_prefix,
            initial_active,
            initial_inactive,
            update_rounds,
            updates_per_round,
            np.random.default_rng(trace_seed + 2),
            "delete_only",
        )
    else:
        mix_states = []
        insert_states = []
        delete_states = []
        for step in range(update_rounds):
            delete_ids = np.arange(step * updates_per_round, (step + 1) * updates_per_round, dtype=np.uint32)
            insert_ids = np.arange(base_points + step * updates_per_round,
                                   base_points + (step + 1) * updates_per_round, dtype=np.uint32)
            write_trace(Path(f"{trace_mix_prefix}{step}"), delete_ids, insert_ids)
            write_trace(Path(f"{trace_insert_prefix}{step}"), delete_ids, insert_ids)
            write_trace(Path(f"{trace_delete_prefix}{step}"), delete_ids, insert_ids)
        for step in range(update_rounds + 1):
            deleted = step * updates_per_round
            mix_states.append(np.concatenate([
                np.arange(deleted, base_points, dtype=np.uint32),
                np.arange(base_points, base_points + deleted, dtype=np.uint32),
            ], axis=0))
            insert_states.append(np.arange(0, base_points + deleted, dtype=np.uint32))
            delete_states.append(np.arange(deleted, base_points, dtype=np.uint32))

    for step in range(update_rounds + 1):
        mix_ids, mix_dists = compute_l2_truth(full_vectors, mix_states[step].astype(np.int64, copy=False), queries, recall_at)
        write_truthset(Path(f"{greator_gt_mix_prefix}{step}.fbin"), mix_ids, mix_dists)
        write_truthset(odin_gt_mix_dir / f"gt_{step * updates_per_round}.bin", mix_ids, mix_dists)

        insert_ids, insert_dists = compute_l2_truth(
            full_vectors,
            insert_states[step].astype(np.int64, copy=False),
            queries,
            recall_at,
        )
        write_truthset(Path(f"{greator_gt_insert_prefix}{step}.fbin"), insert_ids, insert_dists)
        write_truthset(odin_gt_insert_dir / f"gt_{step * updates_per_round}.bin", insert_ids, insert_dists)

        delete_ids, delete_dists = compute_l2_truth(
            full_vectors,
            delete_states[step].astype(np.int64, copy=False),
            queries,
            recall_at,
        )
        write_truthset(Path(f"{greator_gt_delete_prefix}{step}.fbin"), delete_ids, delete_dists)
        write_truthset(odin_gt_delete_dir / f"gt_{step * updates_per_round}.bin", delete_ids, delete_dists)

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(expected_manifest, fh, indent=2)

    return PreparedAssets(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        full_bin=full_bin,
        base_bin=base_bin,
        query_bin=query_bin,
        full_tags=full_tags,
        base_tags=base_tags,
        trace_mix_prefix=trace_mix_prefix,
        trace_insert_prefix=trace_insert_prefix,
        trace_delete_prefix=trace_delete_prefix,
        greator_gt_mix_prefix=greator_gt_mix_prefix,
        greator_gt_insert_prefix=greator_gt_insert_prefix,
        greator_gt_delete_prefix=greator_gt_delete_prefix,
        odin_gt_mix_dir=odin_gt_mix_dir,
        odin_gt_insert_dir=odin_gt_insert_dir,
        odin_gt_delete_dir=odin_gt_delete_dir,
        source_points=total_source_points,
        total_points=total_needed,
        base_points=base_points,
        query_count=query_count,
        update_rounds=update_rounds,
        updates_per_round=updates_per_round,
        recall_at=recall_at,
        dtype_name=dtype_name,
        metric=metric,
        trace_mode=trace_mode,
        base_fraction=base_fraction,
        update_fraction_per_round=update_fraction_per_round,
        trace_seed=trace_seed,
        trace_continue_mode=trace_continue_mode,
        cache_reused=False,
    )


def run_and_capture(cmd: Sequence[str], log_path: Path, env: Optional[Dict[str, str]] = None,
                    cwd: Optional[Path] = None, timeout: Optional[int] = None) -> None:
    ensure_dir(log_path.parent)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_fh:
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                sys.stdout.write(line)
                log_fh.write(line)
            elif proc.poll() is not None:
                break
            if timeout and time.time() - start > timeout:
                proc.kill()
                raise TimeoutError(f"Command timed out: {' '.join(cmd)}")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with code {rc}: {' '.join(cmd)}")


def ensure_target(project_dir: Path, build_dir: Path, targets: Sequence[str], cmake_args: Sequence[str]) -> None:
    subprocess.run(
        ["cmake", "-S", str(project_dir), "-B", str(build_dir), *cmake_args],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", *targets, "-j", str(os.cpu_count() or 4)],
        check=True,
    )


def parse_jsonl_rows(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def normalize_rows(run_id: str, dataset: str, profile: str, workload: str,
                   assets: PreparedAssets, cmd: Sequence[str], log_path: Path,
                   result_path: Path, parsed_rows: List[Dict], extra: Dict) -> List[Dict]:
    rows: List[Dict] = []
    for parsed in parsed_rows:
        row = {field: "" for field in COMMON_FIELDS}
        row.update({
            "run_id": run_id,
            "system": parsed.get("system", extra.get("system", "")),
            "system_label": parsed.get("system_label", extra.get("system_label", parsed.get("system", extra.get("system", "")))),
            "adapter_label": parsed.get("adapter_label", extra.get("adapter_label", "")),
            "dataset": dataset,
            "profile": profile,
            "workload": workload,
            "status": parsed.get("status", "ok"),
            "query_count": assets.query_count if workload_has_query(workload) else 0,
            "source_points": assets.source_points,
            "update_rounds": assets.update_rounds if workload_has_updates(workload) else 0,
            "updates_per_round": assets.updates_per_round if workload_has_updates(workload) else 0,
            "trace_mode": assets.trace_mode,
            "base_fraction": assets.base_fraction,
            "update_fraction_per_round": assets.update_fraction_per_round,
            "trace_seed": assets.trace_seed,
            "trace_continue_mode": assets.trace_continue_mode,
            "command": " ".join(cmd),
            "log_path": str(log_path),
            "result_path": str(result_path),
        })
        for key, value in extra.items():
            if key in row:
                row[key] = value
        for key, value in parsed.items():
            if key in row:
                row[key] = value
        rows.append(row)
    return rows


def _safe_float(value, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def enrich_elapsed_columns(rows: List[Dict]) -> None:
    grouped: Dict[Tuple[str, str, str], List[Dict]] = {}
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("system", "")),
            str(row.get("workload", "")),
        )
        grouped.setdefault(key, []).append(row)

    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: (
            int(_safe_float(row.get("checkpoint_id"), 0)),
            int(_safe_float(row.get("round_index"), 0)),
        ))
        elapsed_s = 0.0
        for row in group_rows:
            workload = str(row.get("workload", ""))
            query_wall_s = _safe_float(row.get("query_wall_time_s"), 0.0)
            if query_wall_s <= 0.0 and workload_has_query(workload):
                qps = _safe_float(row.get("qps"), 0.0)
                query_count = _safe_float(row.get("query_count"), 0.0)
                if qps > 0.0 and query_count > 0.0:
                    query_wall_s = query_count / qps
            row["query_wall_time_s"] = query_wall_s

            foreground_update_wall_s = _safe_float(row.get("foreground_update_wall_time_s"), 0.0)
            maintenance_wall_s = _safe_float(row.get("maintenance_wall_time_s"), 0.0)
            update_wall_s = _safe_float(row.get("round_update_wall_time_s"), 0.0)
            if update_wall_s <= 0.0 and (foreground_update_wall_s > 0.0 or maintenance_wall_s > 0.0):
                update_wall_s = foreground_update_wall_s + maintenance_wall_s
                row["round_update_wall_time_s"] = update_wall_s
            checkpoint_wall_s = query_wall_s + update_wall_s
            row["checkpoint_wall_time_s"] = checkpoint_wall_s
            elapsed_s += checkpoint_wall_s
            row["local_elapsed_time_s"] = elapsed_s
            row["elapsed_time_s"] = elapsed_s


def thread_fields(workload: str, query_threads: int, insert_threads: int,
                  delete_threads: int, update_threads: int,
                  background_threads: int = 0) -> Dict:
    fields = {
        "query_threads": query_threads if workload_has_query(workload) else 0,
        "insert_threads": insert_threads if workload_has_inserts(workload) else 0,
        "delete_threads": delete_threads if workload_has_deletes(workload) else 0,
        "update_threads": update_threads if workload_has_updates(workload) else 0,
        "background_threads": background_threads,
    }
    return fields


def run_inplace(run_id: str, profile: str, assets: PreparedAssets, workload: str,
                cfg: Dict, run_dir: Path) -> List[Dict]:
    build_type = cfg.get("build_type", "Release")
    build_dir = build_dir_for("PageANN", build_type)
    ensure_target(project_root("PageANN"), build_dir, ["pageann_benchmark"], build_cmake_args("PageANN", build_type))
    binary = build_dir / "tests" / "pageann_benchmark"
    params = merged_cfg(cfg, "inplace")
    eff = effective_search_cfg(cfg, "inplace")
    result_jsonl = run_dir / "results" / f"{assets.dataset_name}_{workload}_pageann.jsonl"
    pq_prefix = assets.dataset_dir / f"diskann_family_pq_{eff['pq_bytes']}b"
    index_dir = INDICES_ROOT / run_id / "inplace" / assets.dataset_name / profile
    ensure_dir(index_dir)
    run_heap = index_dir / f"{workload}.heap"
    shared_heap = shared_index_prefix("inplace", assets.dataset_name, {
        "dtype": assets.dtype_name,
        "base_bin": str(assets.base_bin),
        "base_points": assets.base_points,
        "page_size": int(params.get("page_size", 4096)),
        "build_R": int(params.get("build_R", cfg["build_R"])),
        "prune_R": int(params.get("prune_R", cfg["build_R"])),
        "reverse_edge_layout_version": 2,
        "build_L": int(params.get("build_L", cfg["build_L"])),
        "build_C": int(params.get("build_C", cfg.get("build_C", cfg["build_R"] * 4))),
        "build_alpha": float(params.get("build_alpha", cfg["alpha"])),
        "build_threads": int(params.get("build_threads", cfg.get("build_threads", max(1, params.get("query_threads", cfg["query_threads"]))))),
        "saturate_graph": bool(params.get("saturate_graph", cfg.get("saturate_graph", True))),
        "build_mode": params.get("build_mode", cfg.get("build_mode", "mem_index")),
        "build_memory_budget_gb": float(params.get("build_memory_budget_gb", cfg.get("build_memory_budget_gb", 0))),
    }, name="base.heap")
    build_temp_root = index_dir / "build_tmp"
    reuse_built_index = bool(cfg.get("reuse_built_index", True))
    force_rebuild_index = bool(cfg.get("force_rebuild_index", False))

    def make_cmd(heap_path: Path, result_path: Path, reuse_existing_index: bool,
                 cmd_workload: str, query_only_checkpoints: Optional[int] = None) -> List[str]:
        local_params = dict(params)
        if cmd_workload == "query_only":
            for key in [
                "buffer_pool_frames",
                "query_threads",
                "entry_init_mode",
                "query_pool_mode",
                "query_pool_slack",
                "pq_frontier_confirm_enabled",
                "pq_frontier_confirm_topk",
                "pq_confirm_mode",
                "pq_confirm_margin",
                "pq_confirm_delta",
            ]:
                override_key = f"query_only_{key}"
                if override_key in params:
                    local_params[key] = params[override_key]
        checkpoints = int(query_only_checkpoints if query_only_checkpoints is not None
                          else cfg.get("query_only_checkpoints", 1))
        return [
            str(binary),
            assets.dtype_name,
            cmd_workload,
            str(assets.base_bin),
            str(assets.full_bin),
            str(assets.query_bin),
            str(trace_prefix_for_workload(assets, cmd_workload)),
            str(greator_gt_for_workload(assets, cmd_workload)),
            str(pq_prefix),
            str(eff["pq_bytes"]),
            str(local_params.get("pq_sample_rate", cfg.get("pq_sample_rate", 0.1))),
            str(heap_path),
            str(result_path),
            str(assets.base_points),
            str(assets.update_rounds),
            str(assets.recall_at),
            str(eff["search_L"]),
            str(eff["beamwidth"]),
            str(local_params.get("query_threads", cfg["query_threads"])),
            str(local_params.get("insert_threads", cfg["insert_threads"])),
            str(local_params.get("delete_threads", cfg["delete_threads"])),
            schedule_token(cfg.get("query_schedule", "dynamic")),
            schedule_token(cfg.get("update_schedule", "static")),
            "1" if cfg.get("warmup_enabled", True) else "0",
            str(int(cfg.get("warmup_query_count", min(100, assets.query_count)))),
            "1" if cfg.get("warmup_before_each_checkpoint", True) else "0",
            str(inplace_buffer_pool_frames_for_workload(local_params, cmd_workload)),
            str(local_params.get("page_size", 4096)),
            str(local_params.get("build_R", cfg["build_R"])),
            str(local_params.get("build_L", cfg["build_L"])),
            str(local_params.get("build_C", cfg.get("build_C", cfg["build_R"] * 4))),
            str(local_params.get("build_alpha", cfg["alpha"])),
            str(local_params.get("build_threads", cfg.get("build_threads", max(1, local_params.get("query_threads", cfg["query_threads"]))))),
            "1" if local_params.get("saturate_graph", cfg.get("saturate_graph", True)) else "0",
            str(local_params.get("insert_search_L", local_params.get("build_L", cfg["build_L"]))),
            str(local_params.get("prune_R", cfg["build_R"])),
            str(local_params.get("prune_C", cfg["build_R"] * 2)),
            str(local_params.get("prune_alpha", cfg["alpha"])),
            str(local_params.get("bp_query_frac", 1.0 if cmd_workload == "query_only" else 0.0 if cmd_workload == "update_only" else 0.65)),
            str(local_params.get("bp_update_frac", 0.0 if cmd_workload == "query_only" else 1.0 if cmd_workload == "update_only" else 0.25)),
            "1" if local_params.get("flush_after_build_before_measurement", cfg.get("flush_after_build_before_measurement", True)) else "0",
            "1" if local_params.get("drain_every_round", False) else "0",
            "1" if local_params.get("sweep_every_round", False) else "0",
            str(local_params.get("maintenance_budget_mode", local_params.get("sweep_budget_mode", "nodes"))),
            str(local_params.get("maintenance_budget_value", local_params.get("sweep_budget_value", 0))),
            "1" if local_params.get("flush_before_checkpoint", False) else "0",
            str(local_params.get("reverse_edge_overlay_high_water_mark", local_params.get("deferred_edge_high_water_mark", 0))),
            str(local_params.get("reverse_edge_overlay_target_high_water_mark", local_params.get("oversize_prune_high_water_mark", local_params.get("drain_high_water_mark", 0)))),
            str(local_params.get("reverse_edge_pending_threshold", local_params.get("oversize_slack", 2))),
            str(local_params.get("oversize_prune_high_water_mark", local_params.get("reverse_edge_overlay_target_high_water_mark", local_params.get("drain_high_water_mark", 0)))),
            str(local_params.get("oversize_age_threshold_rounds", local_params.get("sweep_round_period", 0))),
            str(local_params.get("tombstone_sweep_high_water_mark", local_params.get("sweep_deleted_threshold", 0))),
            str(local_params.get("flush_dirty_page_high_water_mark", local_params.get("flush_round_period", 0))),
            str(local_params.get("flush_dirty_ratio_threshold", 0.7)),
            "1" if local_params.get("pq_frontier_confirm_enabled", False) else "0",
            str(local_params.get("pq_frontier_confirm_topk", 0)),
            str(local_params.get("pq_confirm_mode", "fixed")),
            str(local_params.get("pq_confirm_margin", 0.05)),
            str(local_params.get("pq_confirm_delta", 4)),
            str(local_params.get("query_pool_mode", "legacy")),
            str(local_params.get("query_pool_slack", 0)),
            str(local_params.get("flush_budget_pages_per_cycle", 32)),
            str(local_params.get("flush_wakeup_ms", 100)),
            str(local_params.get("build_mode", cfg.get("build_mode", "mem_index"))),
            str(local_params.get("build_memory_budget_gb", cfg.get("build_memory_budget_gb", 0))),
            str(build_temp_root),
            str(local_params.get("entry_init_mode", cfg.get("entry_init_mode", "medoid_plus_neighbors"))),
            str(checkpoints),
            "1" if reuse_existing_index else "0",
        ]

    active_heap = run_heap
    reuse_existing_index = False
    if reuse_built_index:
        if workload == "query_only":
            active_heap = shared_heap
            ensure_dir(active_heap.parent)
            with cache_build_lock(shared_heap.parent / ".build.lock"):
                if force_rebuild_index and inplace_cache_ready(shared_heap):
                    remove_inplace_cache(shared_heap)
                if not inplace_cache_ready(shared_heap):
                    cache_result = run_dir / "results" / f"{assets.dataset_name}_cache_build_pageann.jsonl"
                    cache_log = run_dir / "logs" / f"{assets.dataset_name}_cache_build_pageann.log"
                    ensure_dir(shared_heap.parent)
                    run_and_capture(make_cmd(shared_heap, cache_result, False, "query_only", 1), cache_log,
                                    timeout=int(cfg.get("timeout_sec", 1800)))
            reuse_existing_index = True
        else:
            with cache_build_lock(shared_heap.parent / ".build.lock"):
                if force_rebuild_index and inplace_cache_ready(shared_heap):
                    remove_inplace_cache(shared_heap)
                if not inplace_cache_ready(shared_heap):
                    cache_result = run_dir / "results" / f"{assets.dataset_name}_cache_build_pageann.jsonl"
                    cache_log = run_dir / "logs" / f"{assets.dataset_name}_cache_build_pageann.log"
                    ensure_dir(shared_heap.parent)
                    run_and_capture(make_cmd(shared_heap, cache_result, False, "query_only", 1), cache_log,
                                    timeout=int(cfg.get("timeout_sec", 1800)))
            copy_inplace_cache(shared_heap, run_heap)
            active_heap = run_heap
            reuse_existing_index = True

    cmd = make_cmd(active_heap, result_jsonl, reuse_existing_index, workload)
    log_path = run_dir / "logs" / f"{assets.dataset_name}_{workload}_pageann.log"
    run_and_capture(cmd, log_path, timeout=int(cfg.get("timeout_sec", 1800)))
    parsed_rows = parse_jsonl_rows(result_jsonl)
    extra = thread_fields(workload,
                          params.get("query_threads", cfg["query_threads"]),
                          params.get("insert_threads", cfg["insert_threads"]),
                          params.get("delete_threads", cfg["delete_threads"]),
                          max(params.get("insert_threads", cfg["insert_threads"]),
                              params.get("delete_threads", cfg["delete_threads"])),
                          1)
    extra.update({
        "system_label": "pageann",
        "adapter_label": "PageANN",
        "warmup_enabled": 1 if cfg.get("warmup_enabled", True) else 0,
        "warmup_query_count": int(cfg.get("warmup_query_count", min(100, assets.query_count))),
        "search_L": eff["search_L"],
        "beamwidth": eff["beamwidth"],
        "maintenance_policy": cfg.get("maintenance_mode", "native_online"),
    })
    return normalize_rows(run_id, assets.dataset_name, profile, workload, assets, cmd, log_path,
                          result_jsonl, parsed_rows, extra)


def run_diskann(run_id: str, profile: str, assets: PreparedAssets, workload: str,
                cfg: Dict, run_dir: Path) -> List[Dict]:
    build_type = cfg.get("build_type", "Release")
    build_dir = build_dir_for("DiskANN", build_type)
    ensure_target(project_root("DiskANN"), build_dir, ["benchmark_adapter"], build_cmake_args("DiskANN", build_type))
    binary = build_dir / "tests" / "benchmark_adapter"
    params = merged_cfg(cfg, "diskann")
    eff = effective_search_cfg(cfg, "diskann")
    build_cfg = cfg["disk_build"]
    reuse_built_index = bool(cfg.get("reuse_built_index", True))
    force_rebuild_index = bool(cfg.get("force_rebuild_index", False))
    shared_prefix = shared_index_prefix("diskann", assets.dataset_name, {
        "dtype": assets.dtype_name,
        "base_bin": str(assets.base_bin),
        "tag_file": str(assets.base_tags),
        "base_points": assets.base_points,
        "build_R": int(params.get("build_R", cfg["build_R"])),
        "build_L": int(params.get("build_L", cfg["build_L"])),
        "build_B": build_cfg["B"],
        "build_M": build_cfg["M"],
        "build_T": build_cfg["T"],
        "alpha": float(params.get("alpha", cfg["alpha"])),
    })
    run_prefix = INDICES_ROOT / run_id / "diskann" / assets.dataset_name / profile / "index"
    ensure_dir(run_prefix.parent)
    result_jsonl = run_dir / "results" / f"{assets.dataset_name}_{workload}_diskann.jsonl"
    def make_cmd(index_prefix: Path, result_path: Path, reuse_existing_index: bool,
                 cmd_workload: str, query_only_checkpoints: Optional[int] = None) -> List[str]:
        checkpoints = int(query_only_checkpoints if query_only_checkpoints is not None
                          else cfg.get("query_only_checkpoints", 1))
        return [
            str(binary),
            assets.dtype_name,
            cmd_workload,
            str(assets.base_bin),
            str(assets.full_bin),
            str(assets.query_bin),
            str(trace_prefix_for_workload(assets, cmd_workload)),
            str(greator_gt_for_workload(assets, cmd_workload)),
            str(index_prefix),
            str(result_path),
            str(assets.base_tags),
            str(assets.base_points),
            str(assets.update_rounds),
            str(assets.recall_at),
            str(eff["search_L"]),
            str(eff["beamwidth"]),
            str(params.get("query_threads", cfg["query_threads"])),
            str(params.get("insert_threads", cfg["insert_threads"])),
            str(params.get("delete_threads", cfg["delete_threads"])),
            schedule_token(cfg.get("query_schedule", "dynamic")),
            schedule_token(cfg.get("update_schedule", "static")),
            "1" if cfg.get("warmup_enabled", True) else "0",
            str(int(cfg.get("warmup_query_count", min(100, assets.query_count)))),
            "1" if cfg.get("warmup_before_each_checkpoint", True) else "0",
            str(params.get("build_R", cfg["build_R"])),
            str(params.get("build_L", cfg["build_L"])),
            str(build_cfg["B"]),
            str(build_cfg["M"]),
            str(build_cfg["T"]),
            str(params.get("alpha", cfg["alpha"])),
            str(params.get("nodes_to_cache", cfg.get("nodes_to_cache", 0))),
            str(params.get("merge_trigger_ratio", 0.03)),
            str(checkpoints),
            "1" if reuse_existing_index else "0",
        ]

    active_prefix = run_prefix
    reuse_existing_index = False
    if reuse_built_index:
        if workload == "query_only":
            active_prefix = shared_prefix
            ensure_dir(active_prefix.parent)
            with cache_build_lock(shared_prefix.parent / ".build.lock"):
                if force_rebuild_index and diskann_cache_ready(shared_prefix):
                    remove_prefix_family(shared_prefix)
                if not diskann_cache_ready(shared_prefix):
                    cache_result = run_dir / "results" / f"{assets.dataset_name}_cache_build_diskann.jsonl"
                    cache_log = run_dir / "logs" / f"{assets.dataset_name}_cache_build_diskann.log"
                    ensure_dir(shared_prefix.parent)
                    run_and_capture(make_cmd(shared_prefix, cache_result, False, "query_only", 1), cache_log,
                                    timeout=int(cfg.get("timeout_sec", 1800)))
            reuse_existing_index = True
        else:
            with cache_build_lock(shared_prefix.parent / ".build.lock"):
                if force_rebuild_index and diskann_cache_ready(shared_prefix):
                    remove_prefix_family(shared_prefix)
                if not diskann_cache_ready(shared_prefix):
                    cache_result = run_dir / "results" / f"{assets.dataset_name}_cache_build_diskann.jsonl"
                    cache_log = run_dir / "logs" / f"{assets.dataset_name}_cache_build_diskann.log"
                    ensure_dir(shared_prefix.parent)
                    run_and_capture(make_cmd(shared_prefix, cache_result, False, "query_only", 1), cache_log,
                                    timeout=int(cfg.get("timeout_sec", 1800)))
            copy_prefix_family(shared_prefix, run_prefix)
            active_prefix = run_prefix
            reuse_existing_index = True

    cmd = make_cmd(active_prefix, result_jsonl, reuse_existing_index, workload)
    log_path = run_dir / "logs" / f"{assets.dataset_name}_{workload}_diskann.log"
    run_and_capture(cmd, log_path, timeout=int(cfg.get("timeout_sec", 1800)))
    parsed_rows = parse_jsonl_rows(result_jsonl)
    extra = thread_fields(
        workload,
        params.get("query_threads", cfg["query_threads"]),
        params.get("insert_threads", cfg["insert_threads"]),
        params.get("delete_threads", cfg["delete_threads"]),
        max(params.get("insert_threads", cfg["insert_threads"]),
            params.get("delete_threads", cfg["delete_threads"])),
        0)
    extra.update({
        "system_label": "diskann",
        "adapter_label": "DiskANN",
        "warmup_enabled": 1 if cfg.get("warmup_enabled", True) else 0,
        "warmup_query_count": int(cfg.get("warmup_query_count", min(100, assets.query_count))),
        "search_L": eff["search_L"],
        "beamwidth": eff["beamwidth"],
        "maintenance_policy": cfg.get("maintenance_mode", "native_online"),
    })
    return normalize_rows(run_id, assets.dataset_name, profile, workload, assets, cmd, log_path,
                          result_jsonl, parsed_rows, extra)


def run_greator(run_id: str, profile: str, assets: PreparedAssets, workload: str,
                cfg: Dict, run_dir: Path) -> List[Dict]:
    if cfg.get("maintenance_mode", "native_online") == "native_online" and workload == "query_insert_round":
        reason = "greator query+insert native-online workload hangs after adapter validation"
        return [{
            "run_id": run_id,
            "dataset": assets.dataset_name,
            "profile": profile,
            "system": "greator",
            "workload": workload,
            "status": "unsupported",
            "checkpoint_id": 0,
            "unsupported_reason": reason,
            "notes": reason,
        }]
    build_type = cfg.get("build_type", "Release")
    build_dir = build_dir_for("Greator", build_type)
    ensure_target(project_root("Greator"), build_dir, ["benchmark_adapter"], build_cmake_args("Greator", build_type))
    binary = build_dir / "tests" / "benchmark_adapter"
    params = merged_cfg(cfg, "greator")
    eff = effective_search_cfg(cfg, "greator")
    build_cfg = cfg["disk_build"]
    reuse_built_index = bool(cfg.get("reuse_built_index", True))
    force_rebuild_index = bool(cfg.get("force_rebuild_index", False))
    id_map = int(params.get("id_map", 2))
    if workload == "query_only" and params.get("id_map_query_only") is not None:
        id_map = int(params["id_map_query_only"])
    if workload != "query_only" and params.get("id_map_updates") is not None:
        id_map = int(params["id_map_updates"])
    shared_prefix = shared_index_prefix("greator", assets.dataset_name, {
        "dtype": assets.dtype_name,
        "base_bin": str(assets.base_bin),
        "tag_file": str(assets.base_tags),
        "base_points": assets.base_points,
        "build_R": int(params.get("build_R", cfg["build_R"])),
        "build_L": int(params.get("build_L", cfg["build_L"])),
        "build_B": build_cfg["B"],
        "build_M": build_cfg["M"],
        "build_T": build_cfg["T"],
        "L_mem": int(params.get("L_mem", params.get("build_L", cfg["build_L"]))),
        "R_mem": int(params.get("R_mem", params.get("build_R", cfg["build_R"]))),
        "alpha_mem": float(params.get("alpha_mem", cfg["alpha"])),
        "L_disk": int(params.get("L_disk", params.get("build_L", cfg["build_L"]))),
        "R_disk": int(params.get("R_disk", params.get("build_R", cfg["build_R"]))),
        "alpha_disk": float(params.get("alpha_disk", cfg["alpha"])),
        "id_map": id_map,
    })
    run_prefix = INDICES_ROOT / run_id / "greator" / assets.dataset_name / profile / "index"
    ensure_dir(run_prefix.parent)
    result_jsonl = run_dir / "results" / f"{assets.dataset_name}_{workload}_greator.jsonl"
    def clean_temp(prefix: Path) -> None:
        temp_dir = Path(f"{prefix}_temp")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        ensure_dir(temp_dir)

    def make_cmd(index_prefix: Path, result_path: Path, reuse_existing_index: bool,
                 cmd_workload: str, query_only_checkpoints: Optional[int] = None,
                 mem_L_override: Optional[int] = None) -> List[str]:
        checkpoints = int(query_only_checkpoints if query_only_checkpoints is not None
                          else cfg.get("query_only_checkpoints", 1))
        effective_mem_L = int(params.get("mem_L", 0) if mem_L_override is None else mem_L_override)
        return [
            str(binary),
            assets.dtype_name,
            cmd_workload,
            str(assets.base_bin),
            str(assets.full_bin),
            str(assets.query_bin),
            str(trace_prefix_for_workload(assets, cmd_workload)),
            str(greator_gt_for_workload(assets, cmd_workload)),
            str(index_prefix),
            str(result_path),
            str(assets.base_tags),
            str(assets.base_points),
            str(assets.update_rounds),
            str(assets.recall_at),
            str(eff["search_L"]),
            str(eff["beamwidth"]),
            str(params.get("query_threads", cfg["query_threads"])),
            str(params.get("insert_threads", cfg["insert_threads"])),
            str(params.get("delete_threads", cfg["delete_threads"])),
            schedule_token(cfg.get("query_schedule", "dynamic")),
            schedule_token(cfg.get("update_schedule", "static")),
            "1" if cfg.get("warmup_enabled", True) else "0",
            str(int(cfg.get("warmup_query_count", min(100, assets.query_count)))),
            "1" if cfg.get("warmup_before_each_checkpoint", True) else "0",
            str(params.get("build_R", cfg["build_R"])),
            str(params.get("build_L", cfg["build_L"])),
            str(build_cfg["B"]),
            str(build_cfg["M"]),
            str(build_cfg["T"]),
            str(params.get("nodes_to_cache", cfg.get("nodes_to_cache", 0))),
            str(params.get("L_mem", params.get("build_L", cfg["build_L"]))),
            str(params.get("R_mem", params.get("build_R", cfg["build_R"]))),
            str(params.get("alpha_mem", cfg["alpha"])),
            str(params.get("L_disk", params.get("build_L", cfg["build_L"]))),
            str(params.get("R_disk", params.get("build_R", cfg["build_R"]))),
            str(params.get("alpha_disk", cfg["alpha"])),
            str(id_map),
            str(params.get("merge_trigger_ratio", 0.03)),
            str(checkpoints),
            "1" if reuse_existing_index else "0",
        ]

    active_prefix = run_prefix
    reuse_existing_index = False
    if reuse_built_index:
        if workload == "query_only":
            active_prefix = shared_prefix
            with cache_build_lock(shared_prefix.parent / ".build.lock"):
                if force_rebuild_index and greator_cache_ready(shared_prefix, id_map):
                    remove_prefix_family(shared_prefix)
                clean_temp(active_prefix)
                if not greator_cache_ready(shared_prefix, id_map):
                    cache_result = run_dir / "results" / f"{assets.dataset_name}_cache_build_greator.jsonl"
                    cache_log = run_dir / "logs" / f"{assets.dataset_name}_cache_build_greator.log"
                    ensure_dir(shared_prefix.parent)
                    run_and_capture(make_cmd(shared_prefix, cache_result, False, "query_only", 1), cache_log,
                                    timeout=int(cfg.get("timeout_sec", 1800)))
            reuse_existing_index = True
        else:
            with cache_build_lock(shared_prefix.parent / ".build.lock"):
                if force_rebuild_index and greator_cache_ready(shared_prefix, id_map):
                    remove_prefix_family(shared_prefix)
                if not greator_cache_ready(shared_prefix, id_map):
                    cache_result = run_dir / "results" / f"{assets.dataset_name}_cache_build_greator.jsonl"
                    cache_log = run_dir / "logs" / f"{assets.dataset_name}_cache_build_greator.log"
                    ensure_dir(shared_prefix.parent)
                    clean_temp(shared_prefix)
                    run_and_capture(make_cmd(shared_prefix, cache_result, False, "query_only", 1), cache_log,
                                    timeout=int(cfg.get("timeout_sec", 1800)))
            copy_prefix_family(shared_prefix, run_prefix)
            active_prefix = run_prefix
            clean_temp(active_prefix)
            reuse_existing_index = True
    else:
        clean_temp(active_prefix)

    cmd = make_cmd(active_prefix, result_jsonl, reuse_existing_index, workload)
    log_path = run_dir / "logs" / f"{assets.dataset_name}_{workload}_greator.log"
    run_and_capture(cmd, log_path, timeout=int(cfg.get("timeout_sec", 1800)))
    parsed_rows = parse_jsonl_rows(result_jsonl)
    extra = thread_fields(
        workload,
        params.get("query_threads", cfg["query_threads"]),
        params.get("insert_threads", cfg["insert_threads"]),
        params.get("delete_threads", cfg["delete_threads"]),
        max(params.get("insert_threads", cfg["insert_threads"]), params.get("delete_threads", cfg["delete_threads"])),
        1)
    extra.update({
        "system_label": "greator",
        "adapter_label": "Greator",
        "warmup_enabled": 1 if cfg.get("warmup_enabled", True) else 0,
        "warmup_query_count": int(cfg.get("warmup_query_count", min(100, assets.query_count))),
        "search_L": eff["search_L"],
        "beamwidth": eff["beamwidth"],
        "maintenance_policy": cfg.get("maintenance_mode", "native_online"),
        "notes": f"id_map={id_map};maintenance_mode={cfg.get('maintenance_mode', 'native_online')}",
    })
    return normalize_rows(run_id, assets.dataset_name, profile, workload, assets, cmd, log_path,
                          result_jsonl, parsed_rows, extra)


def run_odinann(run_id: str, profile: str, assets: PreparedAssets, workload: str,
                cfg: Dict, run_dir: Path) -> List[Dict]:
    build_type = cfg.get("build_type", "Release")
    build_dir = build_dir_for("PipeANN", build_type)
    ensure_target(project_root("PipeANN"), build_dir, ["benchmark_adapter"], build_cmake_args("PipeANN", build_type))
    binary = build_dir / "tests" / "benchmark_adapter"
    params = merged_cfg(cfg, "odinann")
    eff = effective_search_cfg(cfg, "odinann")
    if cfg.get("maintenance_mode", "native_online") == "native_online" and workload in {
        "update_only", "query_update_round", "query_delete_round", "query_insert_round"
    }:
        reasons = {
            "update_only": "odinann delete+insert online maintenance unsupported after native-online validation",
            "query_update_round": "odinann mixed delete+insert unsupported after native-online validation",
            "query_delete_round": "odinann delete-only online maintenance unsupported after native-online validation",
            "query_insert_round": "odinann query+insert online maintenance unsupported after native-online validation",
        }
        row = {field: "" for field in COMMON_FIELDS}
        row.update(thread_fields(
            workload,
            params.get("query_threads", cfg["query_threads"]),
            params.get("insert_threads", cfg["insert_threads"]),
            params.get("delete_threads", cfg["delete_threads"]),
            max(params.get("insert_threads", cfg["insert_threads"]),
                params.get("delete_threads", cfg["delete_threads"])),
            1))
        row.update({
            "run_id": run_id,
            "system": "odinann",
            "system_label": "odinann",
            "adapter_label": "PipeANN/OdinANN",
            "dataset": assets.dataset_name,
            "profile": profile,
            "workload": workload,
            "status": "unsupported",
            "maintenance_policy": cfg.get("maintenance_mode", "native_online"),
            "unsupported_reason": reasons[workload],
            "notes": reasons[workload],
            "search_L": eff["search_L"],
            "beamwidth": eff["beamwidth"],
            "warmup_enabled": 1 if cfg.get("warmup_enabled", True) else 0,
            "warmup_query_count": int(cfg.get("warmup_query_count", min(100, assets.query_count))),
        })
        return [row]
    reuse_built_index = bool(cfg.get("reuse_built_index", True))
    force_rebuild_index = bool(cfg.get("force_rebuild_index", False))
    use_mem_index = workload == "query_only" and int(params.get("mem_L", 0)) > 0
    shared_prefix = shared_index_prefix("odinann", assets.dataset_name, {
        "adapter_rev": 2,
        "dtype": assets.dtype_name,
        "base_bin": str(assets.base_bin),
        "tag_file": str(assets.base_tags),
        "base_points": assets.base_points,
        "R": int(params.get("R", cfg["build_R"])),
        "L_disk": int(params.get("L_disk", cfg["build_L"])),
        "pq_bytes": eff["pq_bytes"],
        "nbr_type": params.get("nbr_type", "pq"),
        "use_mem_index": use_mem_index,
        "mem_sample_rate": float(params.get("mem_sample_rate", 0.01)),
    })
    run_prefix = INDICES_ROOT / run_id / "odinann" / assets.dataset_name / profile / "index"
    ensure_dir(run_prefix.parent)
    result_jsonl = run_dir / "results" / f"{assets.dataset_name}_{workload}_odinann.jsonl"
    def make_cmd(index_prefix: Path, result_path: Path, reuse_existing_index: bool,
                 cmd_workload: str, query_only_checkpoints: Optional[int] = None,
                 mem_L_override: Optional[int] = None) -> List[str]:
        checkpoints = int(query_only_checkpoints if query_only_checkpoints is not None
                          else cfg.get("query_only_checkpoints", 1))
        effective_mem_L = int(params.get("mem_L", 0) if mem_L_override is None else mem_L_override)
        return [
            str(binary),
            assets.dtype_name,
            cmd_workload,
            str(assets.base_bin),
            str(assets.full_bin),
            str(assets.query_bin),
            str(trace_prefix_for_workload(assets, cmd_workload)),
            str(odin_gt_for_workload(assets, cmd_workload)),
            str(index_prefix),
            str(result_path),
            str(assets.base_tags),
            str(assets.base_points),
            str(assets.update_rounds),
            str(assets.recall_at),
            str(eff["search_L"]),
            str(effective_mem_L),
            str(eff["beamwidth"]),
            str(params.get("query_threads", cfg["query_threads"])),
            str(params.get("insert_threads", cfg["insert_threads"])),
            str(params.get("delete_threads", cfg["delete_threads"])),
            schedule_token(cfg.get("query_schedule", "dynamic")),
            schedule_token(cfg.get("update_schedule", "static")),
            "1" if cfg.get("warmup_enabled", True) else "0",
            str(int(cfg.get("warmup_query_count", min(100, assets.query_count)))),
            "1" if cfg.get("warmup_before_each_checkpoint", True) else "0",
            str(params.get("R", cfg["build_R"])),
            str(params.get("L_disk", cfg["build_L"])),
            str(eff["pq_bytes"]),
            str(params.get("merge_threads", 20)),
            str(params.get("merge_sampled_nbrs", 20)),
            str(params.get("nbr_type", "pq")),
            str(params.get("mem_sample_rate", 0.01)),
            str(params.get("merge_round", 20)),
            str(params.get("merge_io_threshold", 1.2)),
            str(checkpoints),
            "1" if reuse_existing_index else "0",
        ]

    active_prefix = run_prefix
    reuse_existing_index = False
    if reuse_built_index:
        if workload == "query_only":
            active_prefix = shared_prefix
            ensure_dir(active_prefix.parent)
            with cache_build_lock(shared_prefix.parent / ".build.lock"):
                if force_rebuild_index and odinann_cache_ready(shared_prefix, use_mem_index):
                    remove_prefix_family(shared_prefix)
                if not odinann_cache_ready(shared_prefix, use_mem_index):
                    cache_result = run_dir / "results" / f"{assets.dataset_name}_cache_build_odinann.jsonl"
                    cache_log = run_dir / "logs" / f"{assets.dataset_name}_cache_build_odinann.log"
                    ensure_dir(shared_prefix.parent)
                    run_and_capture(make_cmd(shared_prefix, cache_result, False, "build_only", 1),
                                    cache_log, timeout=int(cfg.get("timeout_sec", 1800)))
                    clean_odinann_runtime_artifacts(shared_prefix)
                else:
                    clean_odinann_runtime_artifacts(shared_prefix)
            reuse_existing_index = True
        else:
            with cache_build_lock(shared_prefix.parent / ".build.lock"):
                if force_rebuild_index and odinann_cache_ready(shared_prefix, False):
                    remove_prefix_family(shared_prefix)
                if not odinann_cache_ready(shared_prefix, False):
                    cache_result = run_dir / "results" / f"{assets.dataset_name}_cache_build_odinann.jsonl"
                    cache_log = run_dir / "logs" / f"{assets.dataset_name}_cache_build_odinann.log"
                    ensure_dir(shared_prefix.parent)
                    run_and_capture(make_cmd(shared_prefix, cache_result, False, "build_only", 1, mem_L_override=0),
                                    cache_log, timeout=int(cfg.get("timeout_sec", 1800)))
                    clean_odinann_runtime_artifacts(shared_prefix)
                else:
                    clean_odinann_runtime_artifacts(shared_prefix)
            copy_odinann_cache(shared_prefix, run_prefix, include_mem_index=False)
            clean_odinann_runtime_artifacts(run_prefix)
            active_prefix = run_prefix
            reuse_existing_index = True

    cmd = make_cmd(active_prefix, result_jsonl, reuse_existing_index, workload)
    log_path = run_dir / "logs" / f"{assets.dataset_name}_{workload}_odinann.log"
    run_and_capture(cmd, log_path, timeout=int(cfg.get("timeout_sec", 1800)))
    parsed_rows = parse_jsonl_rows(result_jsonl)
    extra = thread_fields(
        workload,
        params.get("query_threads", cfg["query_threads"]),
        params.get("insert_threads", cfg["insert_threads"]),
        params.get("delete_threads", cfg["delete_threads"]),
        max(params.get("insert_threads", cfg["insert_threads"]), params.get("delete_threads", cfg["delete_threads"])),
        1)
    extra.update({
        "system_label": "odinann",
        "adapter_label": "PipeANN/OdinANN",
        "warmup_enabled": 1 if cfg.get("warmup_enabled", True) else 0,
        "warmup_query_count": int(cfg.get("warmup_query_count", min(100, assets.query_count))),
        "search_L": eff["search_L"],
        "beamwidth": eff["beamwidth"],
        "maintenance_policy": cfg.get("maintenance_mode", "native_online"),
        "notes": f"maintenance_mode={cfg.get('maintenance_mode', 'native_online')};merge_threads={params.get('merge_threads', 20)};"
                 f"merge_sampled_nbrs={params.get('merge_sampled_nbrs', 20)};nbr_type={params.get('nbr_type', 'pq')};"
                 f"pq_bytes={eff['pq_bytes']}",
    })
    return normalize_rows(run_id, assets.dataset_name, profile, workload, assets, cmd, log_path,
                          result_jsonl, parsed_rows, extra)


def write_csv(path: Path, rows: List[Dict]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COMMON_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def rows_for_workload(rows: List[Dict], workload_key: str) -> List[Dict]:
    canonical = {
        "mix": "query_update_round",
        "query_insert": "query_insert_round",
        "query_delete": "query_delete_round",
    }.get(workload_key, workload_key)
    return [row for row in rows if row.get("workload") == canonical]


def write_json(path: Path, rows: List[Dict]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)


def write_workload_summaries(base_dir: Path, prefix: str, rows: List[Dict]) -> Dict[str, Dict[str, Path]]:
    outputs: Dict[str, Dict[str, Path]] = {}
    for workload_key in ("query_only", "update_only", "mix", "query_insert", "query_delete"):
        workload_rows = rows_for_workload(rows, workload_key)
        csv_path = base_dir / f"{prefix}_{workload_key}.csv"
        json_path = base_dir / f"{prefix}_{workload_key}.json"
        write_csv(csv_path, workload_rows)
        write_json(json_path, workload_rows)
        outputs[workload_key] = {"csv": csv_path, "json": json_path}
    return outputs


def notebook_markdown_cell(source: str) -> Dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def notebook_code_cell(source: str) -> Dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def make_report_notebook(notebook_path: Path, workload_csvs: Dict[str, Path], run_id: str, profile_name: str) -> None:
    rel_map = {}
    notebook_dir = notebook_path.parent
    for key, path in workload_csvs.items():
        rel_map[key] = os.path.relpath(path, notebook_dir)
    intro = """# ANN Update Experiment Report

This notebook reads workload-split CSV summaries from files next to it and compares all systems present in those CSVs.

Sections:
- effective configuration tables by dataset / workload
- progress-axis metric plots by workload
- cross-system comparison tables on progress axes
- workload summary tables
- unsupported / failed status tables

`update_only` has no query-side recall/latency/QPS by design, so it focuses on update-side metrics.

Native IO metrics come from each system's original query/update code paths:
- `disk_ios_native`, `query_io_us_native`, `query_cache_hit_rate` for DiskANN/Greator/OdinANN
- physical index read/write counters for PageANN

Reference configs:
- `../config/comparison_config.json`
- `../config/benchmark_datasets_config.json`
"""
    setup_code = f"""from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

WORKLOAD_FILES = {json.dumps(rel_map, indent=2)}
WORKLOAD_TITLES = {json.dumps(WORKLOAD_TITLES, indent=2)}

frames = {{}}
for workload_key, csv_path in WORKLOAD_FILES.items():
    path = Path(csv_path)
    if path.exists():
        frames[workload_key] = pd.read_csv(path)
    else:
        frames[workload_key] = pd.DataFrame()

{{key: frame.shape for key, frame in frames.items()}}
"""
    report_code = """DISPLAY_COLUMNS = [
    "dataset",
    "system",
    "checkpoint_id",
    "round_index",
    "trace_mode",
    "base_fraction",
    "update_fraction_per_round",
    "trace_seed",
    "status",
    "unsupported_reason",
    "recall",
    "lat_avg_us",
    "lat_p50_us",
    "lat_p95_us",
    "lat_p99_us",
    "qps",
    "disk_ios_native",
    "query_io_us_native",
    "native_index_bytes_read",
    "native_index_bytes_written",
    "cumulative_native_index_bytes_read",
    "cumulative_native_index_bytes_written",
    "native_pages_flushed",
    "cumulative_native_pages_flushed",
    "native_write_amplification",
    "rss_kb",
    "peak_rss_kb",
    "update_throughput",
    "cumulative_update_throughput",
    "query_wall_time_s",
    "foreground_update_wall_time_s",
    "maintenance_wall_time_s",
    "round_update_wall_time_s",
    "overlay_pending_edges",
    "overlay_pending_targets",
    "repair_queue_backlog",
    "oversized_node_backlog",
    "dirty_page_count",
    "tombstone_count",
]

INVARIANT_COLUMNS = [
    "dataset",
    "system",
    "source_points",
    "base_points",
    "update_rounds",
    "updates_per_round",
    "trace_mode",
    "base_fraction",
    "update_fraction_per_round",
    "trace_seed",
    "trace_continue_mode",
    "entry_init_mode",
    "query_pool_mode",
    "query_pool_slack",
    "disk_index_size_bytes",
    "aux_index_artifact_bytes",
    "index_num_nodes",
]

CONFIG_COLUMNS = [
    "dataset",
    "system",
    "query_threads",
    "insert_threads",
    "delete_threads",
    "update_threads",
    "background_threads",
    "warmup_enabled",
    "warmup_query_count",
    "search_L",
    "beamwidth",
    "entry_init_mode",
    "query_pool_mode",
    "query_pool_slack",
    "candidate_pool_L_effective",
    "pq_frontier_confirm_topk_effective",
    "buffer_pool_frames",
    "page_size",
    "insert_search_L",
    "prune_R",
    "prune_C",
]

PLOT_METRICS = {
    "query_only": ["recall", "lat_p50_us", "lat_p95_us", "lat_p99_us", "qps", "disk_ios_native", "query_io_us_native", "query_cache_hit_rate", "rss_kb", "peak_rss_kb", "native_index_bytes_read", "cumulative_native_index_bytes_read"],
    "mix": ["recall", "lat_p50_us", "lat_p95_us", "lat_p99_us", "qps", "disk_ios_native", "query_io_us_native", "query_cache_hit_rate", "rss_kb", "peak_rss_kb", "native_index_bytes_read", "native_index_bytes_written", "update_throughput", "cumulative_update_throughput", "foreground_update_wall_time_s", "maintenance_wall_time_s", "round_update_wall_time_s", "overlay_pending_edges", "repair_queue_backlog", "tombstone_count"],
    "query_insert": ["recall", "lat_p50_us", "lat_p95_us", "lat_p99_us", "qps", "disk_ios_native", "query_io_us_native", "query_cache_hit_rate", "rss_kb", "peak_rss_kb", "native_index_bytes_read", "native_index_bytes_written", "cumulative_update_throughput", "foreground_update_wall_time_s", "maintenance_wall_time_s", "round_update_wall_time_s", "overlay_pending_edges", "repair_queue_backlog"],
    "query_delete": ["recall", "lat_p50_us", "lat_p95_us", "lat_p99_us", "qps", "disk_ios_native", "query_io_us_native", "query_cache_hit_rate", "rss_kb", "peak_rss_kb", "native_index_bytes_read", "native_index_bytes_written", "cumulative_update_throughput", "foreground_update_wall_time_s", "maintenance_wall_time_s", "round_update_wall_time_s", "overlay_pending_edges", "repair_queue_backlog", "tombstone_count"],
    "update_only": ["update_throughput", "cumulative_update_throughput", "foreground_update_wall_time_s", "maintenance_wall_time_s", "round_update_wall_time_s", "rss_kb", "peak_rss_kb", "native_index_bytes_read", "native_index_bytes_written", "native_pages_flushed", "native_write_amplification", "overlay_pending_edges", "overlay_pending_targets", "repair_queue_backlog", "tombstone_count"],
}

COMPARE_METRICS = {
    "query_only": ["recall", "lat_p50_us", "lat_p95_us", "lat_p99_us", "qps", "disk_ios_native", "query_io_us_native", "query_cache_hit_rate", "rss_kb", "peak_rss_kb", "native_index_bytes_read"],
    "mix": ["recall", "lat_p50_us", "lat_p95_us", "lat_p99_us", "qps", "disk_ios_native", "query_io_us_native", "query_cache_hit_rate", "rss_kb", "peak_rss_kb", "native_index_bytes_read", "native_index_bytes_written", "update_throughput", "cumulative_update_throughput", "foreground_update_wall_time_s", "maintenance_wall_time_s", "round_update_wall_time_s", "overlay_pending_edges", "repair_queue_backlog", "tombstone_count"],
    "query_insert": ["recall", "lat_p50_us", "lat_p95_us", "lat_p99_us", "qps", "disk_ios_native", "query_io_us_native", "query_cache_hit_rate", "rss_kb", "peak_rss_kb", "native_index_bytes_read", "native_index_bytes_written", "cumulative_update_throughput", "foreground_update_wall_time_s", "maintenance_wall_time_s", "round_update_wall_time_s", "overlay_pending_edges", "repair_queue_backlog"],
    "query_delete": ["recall", "lat_p50_us", "lat_p95_us", "lat_p99_us", "qps", "disk_ios_native", "query_io_us_native", "query_cache_hit_rate", "rss_kb", "peak_rss_kb", "native_index_bytes_read", "native_index_bytes_written", "cumulative_update_throughput", "foreground_update_wall_time_s", "maintenance_wall_time_s", "round_update_wall_time_s", "overlay_pending_edges", "repair_queue_backlog", "tombstone_count"],
    "update_only": ["update_throughput", "cumulative_update_throughput", "foreground_update_wall_time_s", "maintenance_wall_time_s", "round_update_wall_time_s", "rss_kb", "peak_rss_kb", "native_index_bytes_read", "native_index_bytes_written", "native_pages_flushed", "native_write_amplification", "overlay_pending_edges", "overlay_pending_targets", "repair_queue_backlog", "tombstone_count"],
}

def ensure_elapsed_columns(df):
    if df.empty:
        return df
    out = df.copy()
    for col in ["query_wall_time_s", "foreground_update_wall_time_s", "maintenance_wall_time_s", "round_update_wall_time_s", "checkpoint_wall_time_s", "local_elapsed_time_s", "elapsed_time_s"]:
        if col not in out.columns:
            out[col] = 0.0
    if "qps" in out.columns and "query_count" in out.columns:
        mask = pd.to_numeric(out["query_wall_time_s"], errors="coerce").fillna(0.0) <= 0.0
        qps = pd.to_numeric(out["qps"], errors="coerce")
        qcount = pd.to_numeric(out["query_count"], errors="coerce")
        out.loc[mask & qps.gt(0) & qcount.gt(0), "query_wall_time_s"] = qcount / qps
    out["query_wall_time_s"] = pd.to_numeric(out["query_wall_time_s"], errors="coerce").fillna(0.0)
    out["foreground_update_wall_time_s"] = pd.to_numeric(out["foreground_update_wall_time_s"], errors="coerce").fillna(0.0)
    out["maintenance_wall_time_s"] = pd.to_numeric(out["maintenance_wall_time_s"], errors="coerce").fillna(0.0)
    out["round_update_wall_time_s"] = pd.to_numeric(out["round_update_wall_time_s"], errors="coerce").fillna(
        out["foreground_update_wall_time_s"] + out["maintenance_wall_time_s"]
    )
    if "insert_count" not in out.columns:
        out["insert_count"] = 0
    if "delete_count" not in out.columns:
        out["delete_count"] = 0
    out["insert_count"] = pd.to_numeric(out["insert_count"], errors="coerce").fillna(0.0)
    out["delete_count"] = pd.to_numeric(out["delete_count"], errors="coerce").fillna(0.0)
    out["round_update_ops"] = out["insert_count"] + out["delete_count"]
    out["checkpoint_wall_time_s"] = pd.to_numeric(out["checkpoint_wall_time_s"], errors="coerce").fillna(
        out["query_wall_time_s"] + out["round_update_wall_time_s"]
    )
    sort_cols = [c for c in ["dataset", "system", "checkpoint_id", "round_index"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).copy()
    group_cols = [c for c in ["dataset", "system", "workload"] if c in out.columns]
    if group_cols:
        out["local_elapsed_time_s"] = out.groupby(group_cols)["checkpoint_wall_time_s"].cumsum()
        out["cumulative_update_ops"] = out.groupby(group_cols)["round_update_ops"].cumsum()
        out["cumulative_update_wall_time_s"] = out.groupby(group_cols)["round_update_wall_time_s"].cumsum()
    else:
        out["local_elapsed_time_s"] = out["checkpoint_wall_time_s"].cumsum()
        out["cumulative_update_ops"] = out["round_update_ops"].cumsum()
        out["cumulative_update_wall_time_s"] = out["round_update_wall_time_s"].cumsum()
    out["elapsed_time_s"] = out["local_elapsed_time_s"]
    out["cumulative_update_throughput"] = np.where(
        out["cumulative_update_wall_time_s"] > 0,
        out["cumulative_update_ops"] / out["cumulative_update_wall_time_s"],
        np.nan,
    )
    return out


def plot_metric(ax, metric_df, x_col, metric, systems, xlabel):
    if metric_df.empty:
        ax.set_visible(False)
        return
    if metric_df[x_col].nunique() <= 1:
        bars = []
        vals = []
        for system in systems:
            one = metric_df[metric_df["system"].astype(str) == system]
            if not one.empty:
                bars.append(system)
                vals.append(float(one.iloc[-1][metric]))
        ax.bar(bars, vals)
        ax.set_xlabel("system")
    else:
        for system in systems:
            one = metric_df[metric_df["system"].astype(str) == system].sort_values(x_col)
            if not one.empty:
                ax.plot(one[x_col], one[metric], marker="o", label=system)
        ax.set_xlabel(xlabel)
        ax.legend()
    ax.set_ylabel(metric)


def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def plot_metric_grid(sub, metrics, x_col, systems, xlabel, title_prefix, ncols=3, max_panels=9):
    usable = []
    for metric in metrics:
        if metric not in sub.columns:
            continue
        metric_df = sub[["system", x_col, metric]].copy()
        metric_df[metric] = pd.to_numeric(metric_df[metric], errors="coerce")
        metric_df[x_col] = pd.to_numeric(metric_df[x_col], errors="coerce")
        metric_df = metric_df.dropna(subset=[metric, x_col])
        if metric_df.empty:
            continue
        usable.append((metric, metric_df))
    if not usable:
        return
    for group in chunked(usable, max_panels):
        nrows = (len(group) + ncols - 1) // ncols
        fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.8 * nrows))
        axs = np.array(axs).reshape(-1)
        for ax, (metric, metric_df) in zip(axs, group):
            plot_metric(ax, metric_df, x_col, metric, systems, xlabel)
            ax.set_title(metric)
        for ax in axs[len(group):]:
            ax.set_visible(False)
        fig.suptitle(title_prefix)
        plt.tight_layout()
        plt.show()


for workload_key in ["query_only", "update_only", "mix", "query_insert", "query_delete"]:
    df = ensure_elapsed_columns(frames.get(workload_key, pd.DataFrame()))
    print("=" * 100)
    print(WORKLOAD_TITLES[workload_key])
    if df.empty:
        print("No rows for this workload.")
        continue

    datasets = sorted(df["dataset"].dropna().astype(str).unique()) if "dataset" in df.columns else []
    for dataset in datasets:
        sub_all = df[df["dataset"].astype(str) == dataset].copy()
        sub_ok = sub_all[sub_all["status"].astype(str) == "ok"].copy() if "status" in sub_all.columns else sub_all.copy()
        print(f"Dataset: {dataset}")

        progress_x = "checkpoint_id" if workload_key == "query_only" else "round_index"
        xlabel = "checkpoint" if workload_key == "query_only" else "round"

        invariant_cols = [c for c in INVARIANT_COLUMNS if c in sub_all.columns]
        invariant_df = pd.DataFrame()
        if invariant_cols:
            invariant_df = sub_all[invariant_cols].drop_duplicates().sort_values([c for c in ["dataset", "system"] if c in invariant_cols])

        config_cols = [c for c in CONFIG_COLUMNS if c in sub_all.columns]
        if config_cols:
            print("Effective configuration")
            display(sub_all[config_cols].drop_duplicates().sort_values([c for c in ["dataset", "system"] if c in config_cols]))

        if not sub_ok.empty and progress_x in sub_ok.columns:
            systems = sorted(sub_ok["system"].dropna().astype(str).unique()) if "system" in sub_ok.columns else []
            progress_plot_metrics = [m for m in PLOT_METRICS[workload_key] if m in sub_ok.columns]
            title = f"{WORKLOAD_TITLES[workload_key]} | {dataset} | Progress ({xlabel})"
            plot_metric_grid(
                sub_ok,
                progress_plot_metrics,
                progress_x,
                systems,
                xlabel,
                title,
            )
        else:
            print("No ok rows for plotting.")

        if not sub_ok.empty and progress_x in sub_ok.columns and "system" in sub_ok.columns:
            progress_cols = [c for c in [progress_x, "system", *COMPARE_METRICS[workload_key]] if c in sub_ok.columns]
            progress_df = sub_ok[progress_cols].copy()
            for metric in COMPARE_METRICS[workload_key]:
                if metric not in progress_df.columns:
                    continue
                pivot = progress_df.pivot_table(index=progress_x, columns="system", values=metric, aggfunc="first")
                if not pivot.empty:
                    print(f"Comparison table: {metric}")
                    display(pivot.sort_index())

        if not invariant_df.empty:
            print("Invariant metrics")
            display(invariant_df)

        meta_cols = [c for c in ["dataset", "system", "source_points", "base_points", "update_rounds", "updates_per_round", "trace_mode", "base_fraction", "update_fraction_per_round", "trace_seed", "trace_continue_mode", "entry_init_mode", "query_pool_mode", "query_pool_slack"] if c in sub_all.columns]
        if meta_cols:
            print("Workload configuration")
            display(sub_all[meta_cols].drop_duplicates().sort_values([c for c in ["dataset", "system"] if c in meta_cols]))

        if "status" in sub_all.columns:
            bad = sub_all[sub_all["status"].astype(str) != "ok"].copy()
            if not bad.empty:
                print("Non-ok rows")
                display(bad[[c for c in ["dataset", "system", "checkpoint_id", "round_index", "status", "unsupported_reason", "notes"] if c in bad.columns]])

        display_df = sub_all[[c for c in DISPLAY_COLUMNS if c in sub_all.columns]].copy()
        sort_cols = [c for c in ["dataset", "system", "checkpoint_id", "round_index"] if c in display_df.columns]
        if sort_cols:
            display_df = display_df.sort_values(sort_cols)
        print("Detailed rows")
        display(display_df)
    """
    notebook = {
        "cells": [
            notebook_markdown_cell(intro),
            notebook_code_cell(setup_code),
            notebook_code_cell(report_code),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    write_json(notebook_path, notebook)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_DIR / "comparison_config.json"))
    parser.add_argument("--datasets", default="")
    parser.add_argument("--systems", default="")
    parser.add_argument("--workloads", default="")
    parser.add_argument("--profile", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--publish-latest", dest="publish_latest", action="store_true", default=True)
    parser.add_argument("--no-publish-latest", dest="publish_latest", action="store_false")
    parser.add_argument("--force-rebuild-index", dest="force_rebuild_index", action="store_true", default=False)
    parser.add_argument("--no-force-rebuild-index", dest="force_rebuild_index", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_json(Path(args.config))
    datasets_cfg = load_json(CONFIG_DIR / "benchmark_datasets_config.json")
    profile_name = args.profile or cfg["default_profile"]
    profile_cfg = json.loads(json.dumps(cfg["profiles"][profile_name]))
    profile_cfg["force_rebuild_index"] = bool(args.force_rebuild_index or profile_cfg.get("force_rebuild_index", False))

    selected_datasets = [d for d in (args.datasets.split(",") if args.datasets else profile_cfg["datasets"])]
    selected_systems = [s for s in (args.systems.split(",") if args.systems else profile_cfg["systems"])]
    selected_workloads = [w for w in (args.workloads.split(",") if args.workloads else profile_cfg["workloads"])]
    workload_aliases = {
        "query_update": "query_update_round",
        "query_insert": "query_insert_round",
        "query_delete": "query_delete_round",
    }
    selected_workloads = [workload_aliases.get(w, w) for w in selected_workloads]

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = THIS_DIR / "runs" / run_id
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "results")
    ensure_dir(INDICES_ROOT / run_id)
    ensure_dir(SHARED_TAG_ROOT)
    ensure_dir(Path(profile_cfg.get("prepared_cache_root", SHARED_PREPARED_ROOT)))

    rows: List[Dict] = []
    failures: List[str] = []

    runners = {
        "inplace": run_inplace,
        "diskann": run_diskann,
        "greator": run_greator,
        "odinann": run_odinann,
    }
    rng = random.Random(profile_cfg.get("shuffle_seed", 12345))

    for dataset_name in selected_datasets:
        if dataset_name not in datasets_cfg:
            raise KeyError(f"Unknown dataset: {dataset_name}")
        dataset_run_cfg = dataset_profile_cfg(profile_cfg, dataset_name, datasets_cfg[dataset_name])
        assets = prepare_assets(dataset_name, datasets_cfg[dataset_name], dataset_run_cfg)
        print(f"[prepared] dataset={dataset_name} cache={'hit' if assets.cache_reused else 'miss'} dir={assets.dataset_dir}")
        for workload in selected_workloads:
            systems_for_run = list(selected_systems)
            if dataset_run_cfg.get("shuffle_system_order", True):
                rng.shuffle(systems_for_run)
            for idx, system in enumerate(systems_for_run):
                if not system_supports(system, workload):
                    continue
                try:
                    rows.extend(runners[system](run_id, profile_name, assets, workload, dataset_run_cfg, run_dir))
                except Exception as exc:
                    if system == "odinann" and workload in {"update_only", "query_update_round", "query_delete_round", "query_insert_round"}:
                        reason = {
                            "update_only": "odinann delete+insert online maintenance unsupported after native-online validation",
                            "query_update_round": "odinann mixed delete+insert unsupported after native-online validation",
                            "query_delete_round": "odinann delete-only online maintenance unsupported after native-online validation",
                            "query_insert_round": "odinann query+insert online maintenance unsupported after native-online validation",
                        }[workload]
                        rows.append({
                            **{field: "" for field in COMMON_FIELDS},
                            "run_id": run_id,
                            "system": system,
                            "system_label": "odinann",
                            "adapter_label": "PipeANN/OdinANN",
                            "dataset": dataset_name,
                            "profile": profile_name,
                            "workload": workload,
                            "status": "unsupported",
                            "maintenance_policy": dataset_run_cfg.get("maintenance_mode", "native_online"),
                            "unsupported_reason": reason,
                            "notes": str(exc),
                        })
                    else:
                        failures.append(f"{system}/{dataset_name}/{workload}: {exc}")
                        failed_label = "pageann" if system == "inplace" else system
                        failed_adapter = {
                            "inplace": "PageANN",
                            "diskann": "DiskANN",
                            "greator": "Greator",
                            "odinann": "PipeANN/OdinANN",
                        }.get(system, system)
                        rows.append({
                            **{field: "" for field in COMMON_FIELDS},
                            "run_id": run_id,
                            "system": failed_label,
                            "system_label": failed_label,
                            "adapter_label": failed_adapter,
                            "dataset": dataset_name,
                            "profile": profile_name,
                            "workload": workload,
                            "status": "failed",
                            "notes": str(exc),
                        })
                if idx != len(systems_for_run) - 1:
                    if dataset_run_cfg.get("drop_os_cache_between_runs", False):
                        cmd = dataset_run_cfg.get("drop_os_cache_cmd", "").strip()
                        if cmd:
                            subprocess.run(shlex.split(cmd), check=True)
                    cooldown = int(dataset_run_cfg.get("cooldown_sec_between_systems", 0))
                    if cooldown > 0:
                        time.sleep(cooldown)

    enrich_elapsed_columns(rows)

    summary_json = run_dir / "results" / "summary.json"
    write_json(summary_json, rows)

    summary_csv = run_dir / "results" / "summary.csv"
    write_csv(summary_csv, rows)
    run_split_outputs = write_workload_summaries(run_dir / "results", "summary", rows)

    run_notebook = run_dir / "results" / "report.ipynb"
    make_report_notebook(
        run_notebook,
        {key: paths["csv"] for key, paths in run_split_outputs.items()},
        run_id,
        profile_name,
    )

    print(f"\nSummary CSV: {summary_csv}")
    print(f"Run Notebook: {run_notebook}")
    if args.publish_latest:
        ensure_dir(LATEST_DIR)
        for old_name in [
            "latest_summary.json",
            "latest_summary.csv",
            "latest_summary_query_only.csv",
            "latest_summary_query_only.json",
            "latest_summary_update_only.csv",
            "latest_summary_update_only.json",
            "latest_summary_mix.csv",
            "latest_summary_mix.json",
            "latest_report.ipynb",
        ]:
            old_path = THIS_DIR / old_name
            if old_path.exists():
                old_path.unlink()
        latest_summary_json = LATEST_DIR / "summary.json"
        latest_summary_csv = LATEST_DIR / "summary.csv"
        shutil.copy2(summary_json, latest_summary_json)
        shutil.copy2(summary_csv, latest_summary_csv)
        latest_split_outputs = write_workload_summaries(LATEST_DIR, "summary", rows)
        latest_notebook = LATEST_DIR / "report.ipynb"
        make_report_notebook(
            latest_notebook,
            {key: paths["csv"] for key, paths in latest_split_outputs.items()},
            run_id,
            profile_name,
        )
        print(f"Latest Summary CSV: {latest_summary_csv}")
        print(f"Latest Notebook: {latest_notebook}")
    else:
        print("Latest outputs unchanged (--no-publish-latest).")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
