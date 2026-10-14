"""Space-Saving sketch accuracy and its effect on Chunked Greedy.

(a) gamma sweep: Chunked Greedy (C=1024, k=16) with oversampling gamma in
    {1,2,4,8,32}; M = gamma*k counters (gamma=32 -> M=512, the maximum).
    Reports reorder wall time and pillar count (mu=16).
(b) sketch vs exact top-k Dice (global scope, k=16): tie-aware set recall and
    weight recall on sampled rows, plus time of sketch vs exact SpGEMM.
"""
import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import spmatkit
import sporder
import torch

K = 16
GAMMAS = [1, 2, 4, 8, 32]


def gpu_csr(A):
    return (torch.tensor(A.indptr, dtype=torch.int32, device="cuda"),
            torch.tensor(A.indices, dtype=torch.int32, device="cuda"))


def sweep(mat, A, sym):
    p, i = gpu_csr(A)
    out = [dict(matrix=mat, part="sweep", variant="ORIGIN", pillars=int(sporder.count_pillars(p, i, 16)))]
    variants = [(f"gamma={g}", dict(k=K, oversampling=float(g))) for g in GAMMAS] + [("default", dict(k=0, oversampling=0.0))]
    for name, kw in variants:
        try:
            sporder.chunk_similarity_greedy(p, i, is_symmetric=sym, **kw)
            times = []
            for _ in range(3):
                o, t = sporder.chunk_similarity_greedy(p, i, is_symmetric=sym, with_timing=True, **kw)
                times.append(t["wall_ms"])
            p2, i2 = sporder.reorder(p, i, o, symmetric=True)[:2]
            out.append(dict(matrix=mat, part="sweep", variant=name, wall_ms=float(np.median(times)),
                            pillars=int(sporder.count_pillars(p2, i2, 16))))
        except Exception as e:  # noqa: BLE001
            out.append(dict(matrix=mat, part="sweep", variant=name, error=str(e)[:200]))
    return out


def _exact_worker(mat, sym, q):
    A = spmatkit.load_matrix(mat, load_mode="pattern").tocsr()
    A.sort_indices()
    p, i = gpu_csr(A)
    sporder.similarity_graph_exact(p, i, is_symmetric=sym, k=K)
    _, _, te = sporder.similarity_graph_exact(p, i, is_symmetric=sym, k=K, with_timing=True)
    q.put(float(te.get("wall_ms", np.nan)))


def exact_time_subprocess(mat, sym):
    """Exact SpGEMM top-k can abort the process (cuSPARSE out of resources)."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    pr = ctx.Process(target=_exact_worker, args=(mat, sym, q))
    pr.start()
    pr.join(timeout=1800)
    if pr.is_alive():
        pr.kill()
        return {"exact_ms": np.nan, "exact_error": "timeout"}
    if pr.exitcode != 0 or q.empty():
        return {"exact_ms": np.nan, "exact_error": f"failed (exit {pr.exitcode}, cuSPARSE insufficient resources)"}
    return {"exact_ms": q.get()}


def accuracy(mat, A, sym, nsample=4000, seed=0):
    p, i = gpu_csr(A)
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    rows = np.sort(rng.choice(n, size=min(nsample, n), replace=False))
    deg = np.diff(A.indptr).astype(np.float64)
    S = (A[rows] @ A.T).tocsr()
    out = []
    exact_top, kth = [], []
    for r_i, r in enumerate(rows):
        lo, hi = S.indptr[r_i], S.indptr[r_i + 1]
        cols = S.indices[lo:hi]
        inter = S.data[lo:hi]
        msk = cols != r
        cols, inter = cols[msk], inter[msk]
        dice = 2 * inter / (deg[r] + deg[cols])
        top = np.sort(dice)[::-1][:K]
        exact_top.append(dict(zip(cols.tolist(), dice.tolist())))
        kth.append((top.sum(), top[-1] if top.size else 0.0, top.size))
    res = exact_time_subprocess(mat, sym)
    for g in GAMMAS[:4]:
        sporder.similarity_graph(p, i, is_symmetric=sym, k=K, oversampling=float(g))
        idx, _, ts = sporder.similarity_graph(p, i, is_symmetric=sym, k=K, oversampling=float(g), with_timing=True)
        idx = idx[torch.tensor(rows, device="cuda")].cpu().numpy()
        set_rec, w_rec = [], []
        for r_i in range(len(rows)):
            tot, kw, cnt = kth[r_i]
            if cnt == 0:
                continue
            ex = exact_top[r_i]
            nb = [s for s in idx[r_i] if s >= 0]
            ws = [ex.get(int(s), 0.0) for s in nb]
            set_rec.append(sum(w >= kw - 1e-9 and w > 0 for w in ws) / cnt)
            w_rec.append(min(1.0, sum(sorted(ws, reverse=True)[:K]) / tot) if tot > 0 else 1.0)
        out.append(dict(matrix=mat, part="accuracy", variant=f"gamma={g}", sketch_ms=ts.get("wall_ms", np.nan),
                        set_recall=float(np.mean(set_rec)), weight_recall=float(np.mean(w_rec)), **res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-list", required=True)
    ap.add_argument("--accuracy-every", type=int, default=5)
    ap.add_argument("--out", default="sketch_accuracy.csv")
    args = ap.parse_args()
    mats = [m for m in open(args.matrix_list).read().split() if m]
    rows = pd.read_csv(args.out).to_dict("records") if os.path.exists(args.out) else []
    done = {r["matrix"] for r in rows}
    for j, mat in enumerate(mats):
        if mat in done:
            continue
        A = spmatkit.load_matrix(mat, load_mode="pattern").tocsr()
        A.sort_indices()
        A.data[:] = 1.0
        sym = (A != A.T).nnz == 0
        rows += sweep(mat, A, sym)
        if j % args.accuracy_every == 0:
            try:
                rows += accuracy(mat, A, sym)
            except Exception as e:  # noqa: BLE001
                rows.append(dict(matrix=mat, part="accuracy", error=str(e)[:200]))
        torch.cuda.empty_cache()
        print(j + 1, len(mats), mat, flush=True)
        if (j + 1) % 10 == 0:
            pd.DataFrame(rows).to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
