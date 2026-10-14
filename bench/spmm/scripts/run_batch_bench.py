#!/usr/bin/env python3
"""SpMM benchmark over a spmatkit dataset or explicit matrix names."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spmm_bench.batch_bench import (
    DEFAULT_FEATURE_DIMS,
    DEFAULT_METHODS,
    DEFAULT_OUTPUT_CSV,
    run_batch_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SpMM methods over feature dimensions. "
            "Matrices are loaded from a spmatkit dataset or by name."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="testDataset",
        help="spmatkit dataset name (e.g. testDataset, accDataset)",
    )
    parser.add_argument(
        "--matrices",
        nargs="+",
        metavar="NAME",
        help="matrix names to load via spmatkit (overrides --dataset)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=(
            f"Output CSV (default: {DEFAULT_OUTPUT_CSV}). "
            f"For --mode new-write a timestamped '<stem>_<timestamp>.csv' is "
            f"created next to it."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("append", "force", "new-write"),
        default="force",
        help=(
            "append: skip successful keys, retry failed rows, run missing ones; "
            "force: re-run and overwrite matching keys (default); "
            "new-write: create a new timestamped CSV and run fully"
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="SpMM methods to run",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=list(DEFAULT_FEATURE_DIMS),
        help=(
            "Feature dimensions for SpMM methods only. "
            "SpGEMM methods (uses_feature_dim=False) always write feature_dim=0 "
            "and ignore this list."
        ),
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-memory",
        type=float,
        default=128,
        help="spmatkit preload memory budget in GB (0 disables preloading)",
    )
    parser.add_argument(
        "--load-mode",
        choices=("pattern", "full"),
        default="full",
        help="spmatkit load mode (default: full)",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=1.0,
        help="Nonzero fill value when load_mode=pattern",
    )
    parser.add_argument(
        "--reorder-methods",
        nargs="+",
        default=None,
        metavar="METHOD",
        help=(
            "One or more reorder methods to benchmark "
            "(default: ORIGIN = original matrix; omit when using "
            "--reorder-hash alone to resolve the method from the hash)"
        ),
    )
    parser.add_argument(
        "--reorder-hash",
        default=None,
        help=(
            "Specific reorder variant hash (overrides --params; single method "
            "only). Alone, looks up reorder_method from registry/catalog. "
            "With --reorder-methods, the hash must belong to that method."
        ),
    )
    parser.add_argument(
        "--params",
        default=None,
        help=(
            "Reorder params as JSON (for a non-ORIGIN --reorder-methods entry; "
            "single method only)"
        ),
    )
    parser.add_argument(
        "--all-param-sets",
        action="store_true",
        help="Benchmark all stored param variants of each --reorder-methods entry",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    params = json.loads(args.params) if args.params else None
    if (
        args.reorder_methods is not None
        and len(args.reorder_methods) > 1
        and (args.reorder_hash is not None or params is not None)
    ):
        print(
            "Error: --reorder-hash / --params require a single --reorder-methods entry",
            file=sys.stderr,
        )
        return 2
    try:
        output = run_batch_benchmark(
            dataset=None if args.matrices else args.dataset,
            matrices=args.matrices,
            output_csv=args.output_csv,
            mode=args.mode,
            methods=args.methods,
            feature_dims=args.dims,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=args.seed,
            max_memory=args.max_memory,
            load_mode=args.load_mode,
            fill_value=args.fill_value,
            reorder_methods=args.reorder_methods,
            reorder_hash=args.reorder_hash,
            params=params,
            all_param_sets=args.all_param_sets,
        )
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1

    print(f"Results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
