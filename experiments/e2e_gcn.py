"""End-to-end GCN training with Tensor-Core SpMM kernels on reordered graphs.

Total time = reorder + GPU permutation + kernel format preprocessing + E epochs.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import spmatkit
import torch

torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
from spmm_bench.base import SpMMMethod

HASH = {
    "ORIGIN": None,
    "sporder-csg": "a609629886c6",
    "clusterwise-hier": "d4e368685f0d",
    "acc": "5c9e3da11c1e",
    "dtc-tca": "48301ddacf3a",
    "groot-py": "94e0ab87c116",
}
REORDER_CSV = str(Path(__file__).resolve().parents[1] / "runs" / "reorder_bench.csv")


def sync_ms(t0):
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def gpu_permute(indptr, indices, perm):
    """Symmetric permutation P A P^T of a CSR pattern on GPU; perm[new] = old."""
    n = indptr.numel() - 1
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n, device=perm.device, dtype=perm.dtype)
    rows = torch.repeat_interleave(torch.arange(n, device=indptr.device), indptr[1:] - indptr[:-1])
    key = inv[rows].long() * n + inv[indices.long()].long()
    key, _ = torch.sort(key)
    new_rows = key // n
    new_cols = (key % n).int()
    counts = torch.bincount(new_rows, minlength=n)
    new_indptr = torch.zeros(n + 1, dtype=torch.int64, device=indptr.device)
    new_indptr[1:] = torch.cumsum(counts, 0)
    return new_indptr, new_cols


class SpMMFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, op):
        ctx.op = op
        return op(x)

    @staticmethod
    def backward(ctx, g):
        return ctx.op(g.contiguous()), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="+", required=True)
    ap.add_argument("--kernels", nargs="+", default=["dtc-spmm"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--classes", type=int, default=16)
    ap.add_argument("--out", default="e2e_gcn.csv")
    args = ap.parse_args()

    rb = pd.read_csv(REORDER_CSV)
    rb = rb[rb.success == 1]
    rows = []
    for kname in args.kernels:
        kern = SpMMMethod.create(kname)
        kern.build()
        for mat in args.matrices:
            A0 = spmatkit.load_matrix(mat, load_mode="pattern").tocsr()
            A0.sort_indices()
            n = A0.shape[0]
            ip0 = torch.tensor(A0.indptr, dtype=torch.int64, device="cuda")
            ix0 = torch.tensor(A0.indices, dtype=torch.int32, device="cuda")
            g = torch.Generator(device="cuda").manual_seed(0)
            X0 = torch.randn(n, args.hidden, device="cuda", generator=g)
            y0 = torch.randint(0, args.classes, (n,), device="cuda", generator=g)
            gpu_permute(ip0, ix0, torch.arange(n, device="cuda"))
            kern.prepare(A0.astype(np.float32))
            torch.cuda.synchronize()
            for order, h in HASH.items():
                if order == "ORIGIN":
                    t_r = 0.0
                    perm = None
                else:
                    sel = rb[(rb.matrix == mat) & (rb.reorder_method == order) & (rb.reorder_hash == h)]
                    if sel.empty:
                        continue
                    t_r = float(sel.reorder_time_ms.iloc[-1])
                    perm = torch.tensor(spmatkit._load_row_perm(mat, order, h), dtype=torch.int64, device="cuda")
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                if perm is None:
                    ip, ix, X, y = ip0, ix0, X0, y0
                else:
                    ip, ix = gpu_permute(ip0, ix0, perm)
                    X, y = X0[perm], y0[perm]
                t_perm = sync_ms(t0) if perm is not None else 0.0

                import scipy.sparse as sp
                A = sp.csr_matrix(
                    (np.ones(ix.numel(), dtype=np.float32), ix.cpu().numpy(), ip.cpu().numpy()), shape=(n, n)
                )
                t0 = time.perf_counter()
                P = kern.prepare(A)
                t_prep = sync_ms(t0)

                deg = torch.tensor(np.asarray(A.sum(1)).ravel(), device="cuda").clamp_min(1)
                dinv = deg.rsqrt().unsqueeze(1)
                spmm_ms = [0.0]

                def op(x):
                    out, ms = kern.run(P, (dinv * x).contiguous())
                    spmm_ms[0] += ms
                    return dinv * out

                torch.manual_seed(0)
                W1 = torch.nn.Linear(args.hidden, args.hidden).cuda()
                W2 = torch.nn.Linear(args.hidden, args.classes).cuda()
                opt = torch.optim.Adam(list(W1.parameters()) + list(W2.parameters()), lr=0.01)

                def epoch():
                    opt.zero_grad()
                    h1 = F.relu(SpMMFn.apply(W1(X), op))
                    z = W2(SpMMFn.apply(h1, op))
                    loss = -F.log_softmax(z, 1).gather(1, y[:, None]).mean()
                    loss.backward()
                    opt.step()
                    return loss

                for _ in range(3):
                    epoch()
                spmm_ms[0] = 0.0
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.epochs):
                    loss = epoch()
                t_train = sync_ms(t0)
                rows.append(dict(
                    matrix=mat, kernel=kname, order=order, n=n, nnz=A.nnz,
                    reorder_ms=t_r, perm_ms=t_perm, prep_ms=t_prep,
                    epoch_ms=t_train / args.epochs, spmm_ms_per_epoch=spmm_ms[0] / args.epochs,
                    train_ms=t_train, final_loss=float(loss),
                ))
                print(rows[-1], flush=True)
                del P
                torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
