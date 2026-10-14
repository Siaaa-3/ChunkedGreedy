"""Structural reordering-quality metrics for every (matrix, reorder method).

pillars   : P_mu(A) = sum over row windows of distinct columns (mu=16)
tc_blocks : sum over windows of ceil(pillars / 8)   (16x8 TC tiles)
bandwidth : max |i - j|
profile   : sum_i max(0, i - min_j a_ij != 0)
log_gap   : (1/nnz) sum_i sum_k log2(gap_ik), gaps between consecutive sorted
            column indices of row i (first gap measured from i), as in
            Dhulipala et al., KDD'16.
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import spmatkit

HASH = {
    "ORIGIN": None,
    "sporder-csg": "a609629886c6",
    "clusterwise-hier": "d4e368685f0d",
    "acc": "5c9e3da11c1e",
    "dtc-tca": "48301ddacf3a",
    "groot-py": "94e0ab87c116",
}
EXTRA_HASH = {
    "sporder-cg": "eefae4eb9269",
    "global-greedy": "00a81101f0d3",
    "rabbit-local-greedy": "8781fbd56fcb",
}
MU, NU = 16, 8
ORDERS = list(HASH)


def metrics(rows, cols, n):
    order = np.lexsort((cols, rows))
    r, c = rows[order], cols[order]
    win = r // MU
    key = win * n + c
    uk = np.unique(key)
    wins = uk // n
    per_win = np.bincount(wins)
    pillars = int(uk.size)
    tc_blocks = int(np.sum((per_win + NU - 1) // NU))
    bandwidth = int(np.max(np.abs(r - c))) if r.size else 0
    first = np.ones(r.size, dtype=bool)
    first[1:] = r[1:] != r[:-1]
    rmin = c[first]
    rid = r[first]
    profile = int(np.sum(np.maximum(0, rid - rmin)))
    gaps = np.empty(r.size, dtype=np.int64)
    gaps[1:] = c[1:] - c[:-1]
    gaps[first] = np.abs(c[first] - r[first])
    log_gap = float(np.log2(gaps + 1).sum() / r.size)
    return dict(pillars=pillars, tc_blocks=tc_blocks, bandwidth=bandwidth, profile=profile, log_gap=log_gap)


def _set_orders(orders):
    global ORDERS
    ORDERS = orders


def work(mat):
    try:
        A = spmatkit.load_matrix(mat, load_mode="pattern").tocoo()
    except Exception as e:  # noqa: BLE001
        return [dict(matrix=mat, order="ORIGIN", error=str(e))]
    n = A.shape[0]
    rows0, cols0 = A.row.astype(np.int64), A.col.astype(np.int64)
    out = []
    for order in ORDERS:
        h = {**HASH, **EXTRA_HASH}[order]
        try:
            if h is None:
                r, c = rows0, cols0
            else:
                perm = np.asarray(spmatkit._load_row_perm(mat, order, h), dtype=np.int64)
                inv = np.empty(n, dtype=np.int64)
                inv[perm] = np.arange(n)
                r, c = inv[rows0], inv[cols0]
            m = metrics(r, c, n)
            m.update(matrix=mat, order=order, n=n, nnz=int(A.nnz))
            out.append(m)
        except Exception as e:  # noqa: BLE001
            out.append(dict(matrix=mat, order=order, error=str(e)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-list", required=True)
    ap.add_argument("--out", default="quality_metrics.csv")
    ap.add_argument("--procs", type=int, default=24)
    ap.add_argument("--orders", nargs="+", default=list(HASH), choices=[*HASH, *EXTRA_HASH])
    args = ap.parse_args()
    mats = [m for m in open(args.matrix_list).read().split() if m]
    rows = []
    with Pool(args.procs, initializer=_set_orders, initargs=(args.orders,)) as pool:
        for i, res in enumerate(pool.imap_unordered(work, mats)):
            rows.extend(res)
            print(i + 1, len(mats), res[0]["matrix"], flush=True)
            if (i + 1) % 20 == 0:
                pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
