#!/usr/bin/env python3
import argparse
import numpy as np
import os

def read_bin_header(path):
    with open(path,'rb') as f:
        hdr = np.fromfile(f, dtype=np.int32, count=2)
    return int(hdr[0]), int(hdr[1])


def memmap_vectors(path, npts, dim, dtype=np.uint8):
    aligned_dim = ((dim + 7) // 8) * 8
    offset = 8  # two int32
    mm = np.memmap(path, dtype=dtype, mode='r', offset=offset)
    total = mm.size // aligned_dim
    mm = mm.reshape((total, aligned_dim))
    return mm, aligned_dim


def compute_exact_topk(base_file, query_file, k, sample_q=100, chunk_size=1000000):
    base_n, base_dim = read_bin_header(base_file)
    q_n, q_dim = read_bin_header(query_file)
    assert base_dim == q_dim
    dtype_size = 1  # uint8
    aligned_dim = ((base_dim + 7)//8)*8

    # memmap
    base_mm, aligned_dim = memmap_vectors(base_file, base_n, base_dim, dtype=np.uint8)
    query_mm, _ = memmap_vectors(query_file, q_n, q_dim, dtype=np.uint8)

    nq = min(sample_q, q_n)
    queries = np.asarray(query_mm[:nq, :base_dim], dtype=np.float32)

    # initialize topk arrays
    topk_dists = np.full((nq, k), np.inf, dtype=np.float64)
    topk_idx = np.full((nq, k), -1, dtype=np.int64)

    # precompute query squares
    q_sq = np.sum(queries.astype(np.float32)**2, axis=1).astype(np.float64)

    # iterate base in chunks
    for start in range(0, base_n, chunk_size):
        end = min(base_n, start + chunk_size)
        base_chunk = np.asarray(base_mm[start:end, :base_dim], dtype=np.float32)

        # compute base squares
        b_sq = np.sum(base_chunk**2, axis=1).astype(np.float64)
        # compute cross term
        # queries (nq x d) dot base_chunk.T (d x B) -> (nq x B)
        cross = queries.astype(np.float32) @ base_chunk.T
        # distances = q_sq[:,None] + b_sq[None,:] - 2*cross
        dists = q_sq[:, None] + b_sq[None, :] - 2.0 * cross

        # update topk per query
        # for each query, merge current topk_dists (k) with dists (B) and keep top k
        B = dists.shape[1]
        for i in range(nq):
            # combine
            combined_d = np.concatenate((topk_dists[i], dists[i]), axis=0)
            combined_idx = np.concatenate((topk_idx[i], np.arange(start, end, dtype=np.int64)), axis=0)
            # partition
            idx_part = np.argpartition(combined_d, kth=k-1)[:k]
            sel = idx_part[np.argsort(combined_d[idx_part])]
            topk_dists[i] = combined_d[sel]
            topk_idx[i] = combined_idx[sel]

        print(f"Scanned base {start}-{end} (kept topk for {nq} queries)")

    return topk_idx, topk_dists


def read_predicted(pred_file, k, sample_q=100):
    with open(pred_file,'rb') as f:
        hdr = np.fromfile(f, dtype=np.int32, count=2)
        npts, nd = int(hdr[0]), int(hdr[1])
        data = np.fromfile(f, dtype=np.int32)
    data = data.reshape((npts, nd))
    nq = min(sample_q, npts)
    return data[:nq, :k]


def recall_at_k(exact_idx, pred_idx):
    nq, k = pred_idx.shape
    hits = 0
    for i in range(nq):
        exact_set = set(int(x) for x in exact_idx[i])
        pred_set = set(int(x) for x in pred_idx[i])
        hits += len(exact_set & pred_set)
    recall = hits / (nq * k)
    return recall


def main():
    p = argparse.ArgumentParser()
    p.add_argument('base')
    p.add_argument('query')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--sample-queries', type=int, default=100)
    p.add_argument('--chunk-size', type=int, default=1000000)
    p.add_argument('--pred', type=str, default=None, help='predicted results bin to compare')
    args = p.parse_args()

    print('Base:', args.base)
    print('Query:', args.query)
    print('k:', args.k)
    print('sample_queries:', args.sample_queries)
    print('chunk_size:', args.chunk_size)

    exact_idx, exact_d = compute_exact_topk(args.base, args.query, args.k, sample_q=args.sample_queries, chunk_size=args.chunk_size)
    print('Computed exact top-k indices for sample queries')

    if args.pred:
        pred = read_predicted(args.pred, args.k, sample_q=args.sample_queries)
        rec = recall_at_k(exact_idx, pred)
        print(f'Recall@{args.k} (pred vs exact on first {min(args.sample_queries,pred.shape[0])} queries): {rec*100:.2f}%')

    # Optionally write exact to file
    out = 'exact_gt_sample.bin'
    with open(out, 'wb') as f:
        np.array(exact_idx.shape[0], dtype=np.int32).tofile(f)
        np.array(exact_idx.shape[1], dtype=np.int32).tofile(f)
        exact_idx.astype(np.int32).tofile(f)
    print('Wrote exact top-k to', out)

if __name__ == '__main__':
    main()
