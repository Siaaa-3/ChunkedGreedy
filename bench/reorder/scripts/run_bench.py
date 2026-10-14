#!/usr/bin/env python3
"""Row-reordering benchmark CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reorder_bench.config import METHOD_NAMES
from reorder_bench.loader import resolve_matrix_names
from reorder_bench.results import default_run_id
from reorder_bench.run_bench import RunConfig, run_benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "results" / "reorder_bench.csv"

# Reorder-stage wall-clock limit per method (hours). Use -1 for no limit.
TIMEOUT_H_BY_METHOD: dict[str, float] = {
    "acc": 0.3,
    "groot-py": 0.3,
    "groot-cpp": 0.3,
    "dtc-tca": 0.3,
    "clusterwise-hier": 0.3,
    "random": 0.2,
    "global-greedy": 0.3,
    "rabbit-local-greedy": 1,
    "rcm": 1,
    "amd": 1,
    "degree": 1,
    "gray": 1,
    "slashburn": 1,
    "nd": 1,
    "gp": 1,
    "sporder-cg": 0.3,
    "sporder-csg": 0.3,
    "sporder-hm": 0.3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark row-reordering methods on spmatkit matrices.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--matrices",
        nargs="+",
        metavar="NAME",
        help="Matrix base names (spmatkit)",
    )
    target.add_argument(
        "--dataset",
        metavar="NAME",
        help="spmatkit dataset name (matrices listed in datasets/<NAME>.txt)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["groot-py"],
        choices=list(METHOD_NAMES),
        help="Methods to run",
    )
    parser.add_argument(
        "--mode",
        choices=("tune", "save"),
        default="tune",
        help="tune: CSV only; save: also spmatkit.save_reorder_perm",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Output CSV path",
    )
    parser.add_argument("--run-id", default=None, help="Batch run id (default: auto)")
    parser.add_argument("--tc-h", type=int, default=16)
    parser.add_argument("--tc-w", type=int, default=8)
    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        help='Per-method params JSON, e.g. \'{"groot-py":{"knn":32}}\'',
    )
    parser.add_argument(
        "--timeout-h",
        type=float,
        default=None,
        help="Per-run timeout in hours for every method (negative disables; default: per-method table)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run even if results already exist; overwrite matching "
            "(matrix, method, hash) CSV rows (save: also overwrite perm)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params_by_method = json.loads(args.params_json) if args.params_json else None
    try:
        matrices = resolve_matrix_names(matrices=args.matrices, dataset=args.dataset)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    cfg = RunConfig(
        matrices=matrices,
        methods=args.methods,
        mode=args.mode,
        run_id=args.run_id or default_run_id(),
        output_csv=args.output_csv,
        tc_h=args.tc_h,
        tc_w=args.tc_w,
        params_by_method=params_by_method,
        timeout_h_by_method=(
            TIMEOUT_H_BY_METHOD
            if args.timeout_h is None
            else {m: args.timeout_h for m in args.methods}
        ),
        force=args.force,
        dataset=args.dataset,
    )
    try:
        out = run_benchmark(cfg)
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(f"Results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
