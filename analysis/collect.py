"""Collect raw benchmark outputs into the result tables under results/.

Only the configurations in configs/paper.yaml and the matrix sets under datasets/
are kept. Defaults point at the outputs written by `reproduce.py`; every input can
be overridden on the command line.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from common import CONFIG, HASHES, KERNELS, RESULTS, ROOT, WIDTHS, matrix_set

RUNS = ROOT / "runs"
OOM_PATTERN = re.compile(r"OOM|out of memory|out_of_memory|bad_alloc|int32|exited with code -9", re.I)


def _read(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths if Path(p).exists()]
    if not frames:
        raise FileNotFoundError(f"none of {', '.join(map(str, paths))} exists")
    return pd.concat(frames, ignore_index=True)


def _pinned(df: pd.DataFrame, method_col: str = "reorder_method") -> pd.Series:
    return df["reorder_hash"] == df[method_col].map(HASHES)


def reorder_bench(paths, candidates) -> pd.DataFrame:
    df = _read(paths)
    df = df[_pinned(df) & (df["success"] == 1) & df["matrix"].isin(candidates)]
    df = df.sort_values("timestamp").groupby(["matrix", "reorder_method"], as_index=False).tail(1)
    cols = ["matrix", "reorder_method", "reorder_hash", "params_json", "success", "reorder_time_ms", "timestamp"]
    return df[cols].sort_values(["reorder_method", "matrix"])


def reorder_status(bench_paths, skip_paths, candidates) -> pd.DataFrame:
    """Timeout / OOM flags derived from the raw benchmark records."""
    df = _read(bench_paths)
    timeout_s = CONFIG["reorder"]["timeout_s"]
    timeouts = set()
    for p in skip_paths:
        if not Path(p).exists():
            continue
        for line in Path(p).read_text().splitlines():
            g = re.search(rf"matrix=(\S+) method=(\S+).*reason=timeout after {timeout_s}s", line)
            if g:
                timeouts.add((g.group(1), g.group(2)))
    failed = df[(df["success"] != 1) & df["error"].astype(str).str.contains(OOM_PATTERN)]
    ooms = set(zip(failed["matrix"], failed["reorder_method"]))
    rows = [
        dict(matrix=mat, method=m, timeout=int((mat, m) in timeouts), oom=int((mat, m) in ooms))
        for m in CONFIG["main_methods"]
        for mat in candidates
    ]
    return pd.DataFrame(rows)


def amortization(paths, final) -> pd.DataFrame:
    df = _read(paths)
    kernels = KERNELS + CONFIG["spmm"]["extra_kernels"]
    df = df[_pinned(df) & df["matrix"].isin(final) & df["spmm_method"].isin(kernels) & df["feature_dim"].isin(WIDTHS)]
    df = df.drop_duplicates(["matrix", "reorder_method", "spmm_method", "feature_dim"], keep="last")
    cols = ["matrix", "reorder_method", "reorder_hash", "spmm_method", "feature_dim",
            "reorder_time_ms", "spmm_orig_ms", "spmm_reord_ms", "amortization"]
    return df[cols].sort_values(["reorder_method", "spmm_method", "feature_dim", "matrix"])


def quality(paths, final, table1) -> pd.DataFrame:
    df = _read(paths)
    if "error" in df:
        df = df[df["error"].isna()]
    df = df.drop_duplicates(["matrix", "order"], keep="last")
    main = df["order"].isin(["ORIGIN", *CONFIG["main_methods"], "sporder-cg"]) & df["matrix"].isin(final)
    t1 = df["order"].isin(["ORIGIN", "global-greedy", "rabbit-local-greedy"]) & df["matrix"].isin(table1)
    cols = ["matrix", "order", "n", "nnz", "pillars", "tc_blocks", "bandwidth", "profile", "log_gap"]
    out = df[main | t1][cols]
    for c in ["n", "nnz", "pillars", "tc_blocks", "bandwidth", "profile"]:
        out[c] = out[c].astype("int64")
    return out.sort_values(["order", "matrix"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reorder-bench", nargs="+", type=Path, default=[RUNS / "reorder_bench.csv"])
    ap.add_argument("--skip", nargs="+", type=Path, default=[RUNS / "skip.txt"])
    ap.add_argument("--status", type=Path, default=None,
                    help="use an existing status table instead of deriving it from the raw records")
    ap.add_argument("--amortization", nargs="+", type=Path, default=[RUNS / "reorder_spmm_amortization.csv"])
    ap.add_argument("--quality", nargs="+", type=Path, default=[RUNS / "quality_metrics.csv"])
    ap.add_argument("--e2e", type=Path, default=RUNS / "e2e_gcn.csv")
    ap.add_argument("--perm-time", type=Path, default=RUNS / "perm_time_sporder.csv")
    ap.add_argument("--sketch", type=Path, default=RUNS / "sketch_accuracy.csv")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()

    candidates, final = matrix_set("candidates"), matrix_set("final")
    graphs, table1 = matrix_set("graphs"), matrix_set("table1")
    args.out.mkdir(parents=True, exist_ok=True)

    def write(df: pd.DataFrame, name: str) -> None:
        df.to_csv(args.out / name, index=False)
        print(f"{name}: {len(df)} rows")

    write(reorder_bench(args.reorder_bench, candidates), "reorder_bench.csv")
    status = pd.read_csv(args.status) if args.status else reorder_status(args.reorder_bench, args.skip, candidates)
    write(status, "reorder_status.csv")
    write(amortization(args.amortization, final), "reorder_spmm_amortization.csv")
    write(quality(args.quality, final, table1), "quality_metrics.csv")
    for src, name, mats in [(args.e2e, "e2e_gcn.csv", graphs), (args.perm_time, "perm_time_sporder.csv", graphs),
                            (args.sketch, "sketch_accuracy.csv", final)]:
        if src.exists():
            df = pd.read_csv(src)
            write(df[df["matrix"].isin(mats)], name)
        else:
            print(f"{name}: skipped ({src} not found)")


if __name__ == "__main__":
    main()
