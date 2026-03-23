#!/usr/bin/env python3
"""Convert .fvecs to DiskANN .bin format, with optional slicing.

DiskANN .bin format:
- int32: number of points
- int32: dimension
- payload: row-major vectors

This script reads an .fvecs file (each vector = int32 dim + dim float32 values),
and writes a new .bin file without modifying existing files.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .fvecs to DiskANN .bin with slicing")
    parser.add_argument("--input", required=True, help="Input .fvecs path")
    parser.add_argument("--output", required=True, help="Output DiskANN .bin path")
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=100_000_000,
        help="How many vectors to write from the start (default: 100000000)",
    )
    parser.add_argument(
        "--start-vector",
        type=int,
        default=0,
        help="Start offset in vectors from beginning (default: 0)",
    )
    parser.add_argument(
        "--chunk-vectors",
        type=int,
        default=200_000,
        help="Chunk size in vectors for streaming conversion (default: 200000)",
    )
    parser.add_argument(
        "--out-dtype",
        choices=["float", "uint8"],
        default="float",
        help="Output vector type in .bin payload (default: float)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists",
    )
    return parser.parse_args()


def discover_fvecs_layout(input_path: Path) -> tuple[int, int, int]:
    file_size = input_path.stat().st_size
    if file_size < 8:
        raise ValueError("Input file is too small to be a valid .fvecs file")

    with input_path.open("rb") as fin:
        raw_dim = fin.read(4)
        if len(raw_dim) != 4:
            raise ValueError("Could not read fvecs dimension header")
        dim = struct.unpack("<i", raw_dim)[0]

    if dim <= 0:
        raise ValueError(f"Invalid fvecs dimension: {dim}")

    record_bytes = 4 + dim * 4
    if file_size % record_bytes != 0:
        raise ValueError(
            "File size is not divisible by fvecs record size. "
            f"size={file_size}, record_bytes={record_bytes}, dim={dim}"
        )

    total_vectors = file_size // record_bytes
    return dim, record_bytes, total_vectors


def convert_fvecs_to_bin(
    input_path: Path,
    output_path: Path,
    start_vector: int,
    max_vectors: int,
    chunk_vectors: int,
    out_dtype: str,
) -> None:
    dim, record_bytes, total_vectors = discover_fvecs_layout(input_path)

    if start_vector < 0:
        raise ValueError("start-vector must be >= 0")
    if max_vectors <= 0:
        raise ValueError("max-vectors must be > 0")
    if chunk_vectors <= 0:
        raise ValueError("chunk-vectors must be > 0")
    if start_vector >= total_vectors:
        raise ValueError(
            f"start-vector {start_vector} is >= total vectors {total_vectors}"
        )

    n_to_write = min(max_vectors, total_vectors - start_vector)
    if n_to_write <= 0:
        raise ValueError("No vectors selected for output")

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Detected dim={dim}, total_vectors={total_vectors}")
    print(f"Slice: start={start_vector}, count={n_to_write}")
    print(f"Output payload dtype: {out_dtype}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    written = 0

    with input_path.open("rb") as fin, output_path.open("wb") as fout:
        # Write DiskANN header
        np.array([n_to_write, dim], dtype=np.int32).tofile(fout)

        fin.seek(start_vector * record_bytes, os.SEEK_SET)

        while written < n_to_write:
            take = min(chunk_vectors, n_to_write - written)
            raw = fin.read(take * record_bytes)
            if len(raw) != take * record_bytes:
                raise RuntimeError(
                    "Unexpected EOF while reading fvecs: "
                    f"expected {take * record_bytes}, got {len(raw)}"
                )

            # Reinterpret each record as (dim+1) int32 values.
            as_int = np.frombuffer(raw, dtype=np.int32).reshape(take, dim + 1)
            if not np.all(as_int[:, 0] == dim):
                bad = int(np.where(as_int[:, 0] != dim)[0][0])
                raise RuntimeError(
                    f"Dimension mismatch in chunk at local row {bad}: "
                    f"found {as_int[bad, 0]}, expected {dim}"
                )

            vecs_f32 = as_int[:, 1:].view(np.float32)

            if out_dtype == "float":
                out_chunk = vecs_f32.astype(np.float32, copy=False)
            else:
                # Most SIFT .fvecs values are near integer pixel-range values.
                # Clamp and round for uint8 payload when requested.
                out_chunk = np.rint(vecs_f32).clip(0, 255).astype(np.uint8)

            out_chunk.tofile(fout)
            written += take

            if written % max(chunk_vectors * 5, 1) == 0 or written == n_to_write:
                pct = written * 100.0 / n_to_write
                elapsed = time.time() - start_time
                rate = written / elapsed if elapsed > 0 else 0.0
                print(
                    f"Progress: {written}/{n_to_write} ({pct:.2f}%), "
                    f"{rate:,.0f} vec/s"
                )

    elapsed = time.time() - start_time
    print(f"Done. Wrote {n_to_write} vectors, dim {dim}, elapsed {elapsed:.1f}s")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input not found: {input_path}", file=sys.stderr)
        return 1

    if output_path.exists() and not args.force:
        print(
            f"Error: output already exists: {output_path} (use --force to overwrite)",
            file=sys.stderr,
        )
        return 1

    try:
        convert_fvecs_to_bin(
            input_path=input_path,
            output_path=output_path,
            start_vector=args.start_vector,
            max_vectors=args.max_vectors,
            chunk_vectors=args.chunk_vectors,
            out_dtype=args.out_dtype,
        )
    except Exception as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
