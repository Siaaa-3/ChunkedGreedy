#!/usr/bin/env python3
"""Compute SpMM amortization and ROI into reorder_spmm_amortization.csv.

Candidate set is defined by CLI:
  matrices/dataset × reorder_method(s) × spmm_method(s) × dims

  amortization = T_reorder / ΔSpMM   (inf when ΔSpMM <= 0)
  roi          = ΔSpMM / T_reorder   (negative when SpMM got slower)

Three phases:
  1. Scan: collect missing ORIGIN SpMM keys and missing reorder SpMM keys.
  2. Fill: append-run missing ORIGIN / reorder SpMM into spmm_bench.csv.
  3. Compute: write amortization + ROI rows to reorder_spmm_amortization.csv.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import spmatkit

from spmm_bench.base import SpMMMethod
from spmm_bench.batch_bench import (
    AMORTIZATION_FIELDS,
    DEFAULT_AMORTIZATION_CSV,
    DEFAULT_FEATURE_DIMS,
    DEFAULT_METHODS,
    DEFAULT_OUTPUT_CSV,
    ResultKey,
    build_reorder_time_lookup,
    load_result_rows,
    result_key,
    result_key_from_row,
    run_batch_benchmark,
)

ORIGIN = "ORIGIN"

# (matrix, spmm_method, tc_h, tc_w, feature_dim)
OriginKey = tuple[str, str, int, int, int]
# (matrix, reorder_method, reorder_hash, spmm_method, tc_h, tc_w, feature_dim)
# == ResultKey


def format_amortization(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.10g}"


def format_roi(value: float) -> str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.10g}"


def compute_amortization(reorder_time_ms: float, spmm_reduction_ms: float) -> float:
    if spmm_reduction_ms <= 0:
        return math.inf
    return reorder_time_ms / spmm_reduction_ms


def compute_roi(reorder_time_ms: float, spmm_reduction_ms: float) -> float:
    """ROI = ΔSpMM / T_reorder. Larger is better; negative means SpMM got worse."""
    if reorder_time_ms <= 0:
        if spmm_reduction_ms > 0:
            return math.inf
        if spmm_reduction_ms < 0:
            return -math.inf
        return float("nan")
    return spmm_reduction_ms / reorder_time_ms


def resolve_matrix_names(
    *,
    matrices: list[str] | None,
    dataset: str | None,
) -> list[str]:
    if matrices:
        return list(matrices)
    if dataset:
        return list(spmatkit.get_dataset_names(dataset))
    raise ValueError("Either --matrices or --dataset is required")


def method_tc_tiles(methods: list[str]) -> dict[str, tuple[int, int]]:
    tiles: dict[str, tuple[int, int]] = {}
    for name in methods:
        wrapper = SpMMMethod.create(name)
        tiles[name] = (int(wrapper.tc_h), int(wrapper.tc_w))
    return tiles


def feature_dims_for_method(method: str, feature_dims: list[int]) -> list[int]:
    """Return CLI dims for SpMM methods; SpGEMM always uses feature_dim=0."""
    wrapper = SpMMMethod.create(method)
    if wrapper.uses_feature_dim:
        return list(feature_dims)
    return [0]


def load_amortization_rows(
    csv_path: Path,
) -> tuple[dict[ResultKey, dict[str, str]], bool]:
    """Load rows. Second return value is True if CSV lacked roi and needs rewrite."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {}, False
    required = set(AMORTIZATION_FIELDS) - {"roi"}
    rows: dict[ResultKey, dict[str, str]] = {}
    needs_rewrite = False
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Amortization CSV missing columns {sorted(missing)}: "
                f"{list(reader.fieldnames)!r} ({csv_path})"
            )
        has_roi_col = "roi" in reader.fieldnames
        if not has_roi_col:
            needs_rewrite = True
        for row in reader:
            key = result_key_from_row(row)
            entry = {field: str(row.get(field, "")) for field in AMORTIZATION_FIELDS}
            if not entry.get("roi"):
                try:
                    rt = float(entry["reorder_time_ms"])
                    red = float(entry["spmm_reduction_ms"])
                    entry["roi"] = format_roi(compute_roi(rt, red))
                    if has_roi_col:
                        needs_rewrite = True
                except (TypeError, ValueError):
                    entry["roi"] = ""
            rows.setdefault(key, entry)
    return rows, needs_rewrite


def write_amortization_rows(
    csv_path: Path, rows: dict[ResultKey, dict[str, str]]
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AMORTIZATION_FIELDS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow({field: row.get(field, "") for field in AMORTIZATION_FIELDS})
    tmp_path.replace(csv_path)


def _is_success(row: dict[str, str] | None) -> bool:
    return row is not None and str(row.get("success", "")).strip() == "1"


def build_candidates(
    matrices: list[str],
    reorder_methods: list[str],
    spmm_methods: list[str],
    feature_dims: list[int],
    *,
    all_param_sets: bool,
    tiles: dict[str, tuple[int, int]],
) -> list[ResultKey]:
    """Cartesian product of matrices × reorder variants × methods × per-method dims."""
    candidates: list[ResultKey] = []
    seen: set[ResultKey] = set()
    for matrix in matrices:
        for reorder_method in reorder_methods:
            entries = spmatkit.list_catalog_entries(
                matrix=matrix,
                reorder_method=reorder_method,
                all_param_sets=all_param_sets,
            )
            if not entries:
                print(
                    f"Warning: no catalog variant for {matrix!r} / {reorder_method!r}",
                    flush=True,
                )
                continue
            for entry in entries:
                rhash = str(entry["reorder_hash"])
                for method in spmm_methods:
                    tc_h, tc_w = tiles[method]
                    for dim in feature_dims_for_method(method, feature_dims):
                        key = result_key(
                            matrix,
                            reorder_method,
                            rhash,
                            method,
                            tc_h,
                            tc_w,
                            dim,
                        )
                        if key not in seen:
                            seen.add(key)
                            candidates.append(key)
    return candidates


def phase1_scan(
    candidates: list[ResultKey],
    bench_rows: dict[ResultKey, dict[str, str]],
    amort_rows: dict[ResultKey, dict[str, str]],
    *,
    force: bool,
) -> tuple[set[OriginKey], set[ResultKey], list[ResultKey]]:
    """Return (missing_origin, missing_reorder_spmm, todo_candidates)."""
    missing_origin: set[OriginKey] = set()
    missing_reorder_spmm: set[ResultKey] = set()
    todo: list[ResultKey] = []

    for key in candidates:
        matrix, reorder_method, reorder_hash, method, tc_h, tc_w, dim = key
        existing = amort_rows.get(key)
        if (
            not force
            and existing
            and existing.get("amortization")
            and existing.get("roi")
        ):
            continue
        todo.append(key)

        origin_row = _find_origin_row(
            bench_rows,
            matrix=matrix,
            spmm_method=method,
            spmm_tc_h=tc_h,
            spmm_tc_w=tc_w,
            feature_dim=dim,
        )
        if not _is_success(origin_row):
            missing_origin.add((matrix, method, tc_h, tc_w, dim))

        reorder_row = bench_rows.get(key)
        if not _is_success(reorder_row):
            missing_reorder_spmm.add(key)

    return missing_origin, missing_reorder_spmm, todo


def _find_origin_row(
    bench_rows: dict[ResultKey, dict[str, str]],
    *,
    matrix: str,
    spmm_method: str,
    spmm_tc_h: int,
    spmm_tc_w: int,
    feature_dim: int,
) -> dict[str, str] | None:
    for key, row in bench_rows.items():
        if (
            key[0] == matrix
            and key[1] == ORIGIN
            and key[3] == spmm_method
            and key[4] == spmm_tc_h
            and key[5] == spmm_tc_w
            and key[6] == feature_dim
            and _is_success(row)
        ):
            return row
    return None


def phase2_run_missing_origin(
    bench_path: Path,
    missing_origin: set[OriginKey],
) -> None:
    if not missing_origin:
        print("Phase 2a: no missing ORIGIN SpMM rows.", flush=True)
        return

    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for matrix, method, _tc_h, _tc_w, dim in missing_origin:
        groups[(method, dim)].add(matrix)

    print(
        f"Phase 2a: running {len(missing_origin)} missing ORIGIN SpMM key(s) "
        f"in {len(groups)} group(s) (warmup=30, iterations=100, mode=append).",
        flush=True,
    )
    for (method, dim), matrices in sorted(groups.items()):
        names = sorted(matrices)
        print(
            f"  append ORIGIN: method={method} dim={dim} matrices={len(names)}",
            flush=True,
        )
        run_batch_benchmark(
            dataset=None,
            matrices=names,
            output_csv=bench_path,
            mode="append",
            methods=[method],
            feature_dims=[dim],
            warmup=30,
            iterations=100,
            reorder_methods=ORIGIN,
        )


def phase2_run_missing_reorder(
    bench_path: Path,
    missing_reorder_spmm: set[ResultKey],
) -> None:
    if not missing_reorder_spmm:
        print("Phase 2b: no missing reorder SpMM rows.", flush=True)
        return

    # Group by (reorder_method, reorder_hash, spmm_method, dim).
    groups: dict[tuple[str, str, str, int], set[str]] = defaultdict(set)
    for matrix, reorder_method, reorder_hash, method, _tc_h, _tc_w, dim in missing_reorder_spmm:
        groups[(reorder_method, reorder_hash, method, dim)].add(matrix)

    print(
        f"Phase 2b: running {len(missing_reorder_spmm)} missing reorder SpMM key(s) "
        f"in {len(groups)} group(s) (warmup=30, iterations=100, mode=append).",
        flush=True,
    )
    for (reorder_method, reorder_hash, method, dim), matrices in sorted(groups.items()):
        names = sorted(matrices)
        print(
            f"  append reorder: method={reorder_method} hash={reorder_hash} "
            f"spmm={method} dim={dim} matrices={len(names)}",
            flush=True,
        )
        run_batch_benchmark(
            dataset=None,
            matrices=names,
            output_csv=bench_path,
            mode="append",
            methods=[method],
            feature_dims=[dim],
            warmup=30,
            iterations=100,
            reorder_methods=reorder_method,
            reorder_hash=reorder_hash,
        )


def phase3_compute(
    todo: list[ResultKey],
    bench_rows: dict[ResultKey, dict[str, str]],
    amort_rows: dict[ResultKey, dict[str, str]],
    reorder_time_lookup: dict[tuple[str, str, str], str],
    *,
    force: bool,
) -> tuple[int, int, list[str]]:
    """Returns (n_written, n_skipped, unresolved_messages)."""
    written = 0
    skipped = 0
    unresolved: list[str] = []

    for key in todo:
        matrix, reorder_method, reorder_hash, method, tc_h, tc_w, dim = key
        existing = amort_rows.get(key)
        if (
            not force
            and existing
            and existing.get("amortization")
            and existing.get("roi")
        ):
            skipped += 1
            continue

        reorder_row = bench_rows.get(key)
        if not _is_success(reorder_row):
            unresolved.append(
                f"{matrix}/{reorder_method}/{reorder_hash}/{method}/dim={dim}: "
                f"missing successful reorder SpMM"
            )
            continue
        assert reorder_row is not None

        rt_text = str(reorder_row.get("reorder_time_ms", "")).strip()
        if not rt_text:
            rt_text = reorder_time_lookup.get(
                (matrix, reorder_method, reorder_hash), ""
            )
        if not rt_text:
            unresolved.append(
                f"{matrix}/{reorder_method}/{reorder_hash}/{method}/dim={dim}: "
                f"missing reorder_time_ms"
            )
            continue

        origin = _find_origin_row(
            bench_rows,
            matrix=matrix,
            spmm_method=method,
            spmm_tc_h=tc_h,
            spmm_tc_w=tc_w,
            feature_dim=dim,
        )
        if origin is None:
            unresolved.append(
                f"{matrix}/{reorder_method}/{method}/dim={dim}: "
                f"missing successful ORIGIN SpMM"
            )
            continue

        reorder_time_ms = float(rt_text)
        spmm_orig_ms = float(origin["time_ms"])
        spmm_reord_ms = float(reorder_row["time_ms"])
        spmm_reduction_ms = spmm_orig_ms - spmm_reord_ms
        amort = compute_amortization(reorder_time_ms, spmm_reduction_ms)
        roi = compute_roi(reorder_time_ms, spmm_reduction_ms)

        amort_rows[key] = {
            "matrix": matrix,
            "reorder_method": reorder_method,
            "reorder_hash": reorder_hash,
            "spmm_method": method,
            "spmm_tc_h": str(tc_h),
            "spmm_tc_w": str(tc_w),
            "feature_dim": str(dim),
            "reorder_time_ms": rt_text,
            "spmm_orig_ms": f"{spmm_orig_ms:.10g}",
            "spmm_reord_ms": f"{spmm_reord_ms:.10g}",
            "spmm_reduction_ms": f"{spmm_reduction_ms:.10g}",
            "amortization": format_amortization(amort),
            "roi": format_roi(roi),
        }
        written += 1

    return written, skipped, unresolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute amortization and ROI for a CLI-defined candidate set and "
            "write results/reorder_spmm_amortization.csv. Missing ORIGIN / "
            "reorder SpMM times are append-run into spmm_bench.csv first."
        ),
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--matrices",
        nargs="+",
        metavar="NAME",
        help="matrix names (overrides --dataset)",
    )
    target.add_argument(
        "--dataset",
        metavar="NAME",
        help="spmatkit dataset name",
    )
    parser.add_argument(
        "--reorder-methods",
        nargs="+",
        required=True,
        metavar="METHOD",
        help="one or more reorder methods (non-ORIGIN)",
    )
    parser.add_argument(
        "--all-param-sets",
        action="store_true",
        help="include all catalog param variants of each --reorder-methods entry",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="SpMM methods (default: suite defaults)",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=list(DEFAULT_FEATURE_DIMS),
        help=(
            "Feature dimensions for SpMM methods only. "
            "SpGEMM methods always use feature_dim=0 and ignore this list."
        ),
    )
    parser.add_argument(
        "--bench-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"SpMM bench CSV (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_AMORTIZATION_CSV,
        help=f"Amortization CSV (default: {DEFAULT_AMORTIZATION_CSV})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute amortization even when already present",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bench_path = args.bench_csv
    amort_path = args.output_csv

    if any(m == ORIGIN for m in args.reorder_methods):
        print("Error: --reorder-methods must not include ORIGIN", file=sys.stderr)
        return 1

    try:
        matrices = resolve_matrix_names(matrices=args.matrices, dataset=args.dataset)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Failed to resolve matrices: {exc}", file=sys.stderr)
        return 1

    tiles = method_tc_tiles(args.methods)
    candidates = build_candidates(
        matrices,
        args.reorder_methods,
        args.methods,
        args.dims,
        all_param_sets=args.all_param_sets,
        tiles=tiles,
    )
    print(
        f"Candidate set: {len(matrices)} matrix(ces) × "
        f"{len(args.reorder_methods)} reorder method(s) × "
        f"{len(args.methods)} method(s) "
        f"(SpMM dims={list(args.dims)}; SpGEMM dim=0) "
        f"-> {len(candidates)} key(s)",
        flush=True,
    )
    if not candidates:
        print("No candidates (empty catalog match?).", flush=True)
        return 1

    if not bench_path.exists():
        # Allow starting from an empty bench file: create header-only via append runs.
        print(f"Bench CSV not found; will create via append: {bench_path}", flush=True)
        bench_rows: dict[ResultKey, dict[str, str]] = {}
    else:
        bench_rows = load_result_rows(bench_path)

    amort_rows, amort_needs_rewrite = load_amortization_rows(amort_path)

    # ---- Phase 1 ----
    missing_origin, missing_reorder_spmm, todo = phase1_scan(
        candidates, bench_rows, amort_rows, force=args.force
    )
    print(
        f"Phase 1: todo={len(todo)}; "
        f"missing ORIGIN SpMM={len(missing_origin)}; "
        f"missing reorder SpMM={len(missing_reorder_spmm)}",
        flush=True,
    )

    if (
        not todo
        and not missing_origin
        and not missing_reorder_spmm
        and not amort_needs_rewrite
    ):
        print("Nothing to do.", flush=True)
        return 0

    # ---- Phase 2 ----
    phase2_run_missing_origin(bench_path, missing_origin)
    phase2_run_missing_reorder(bench_path, missing_reorder_spmm)

    # ---- Phase 3 ----
    bench_rows = load_result_rows(bench_path)
    amort_rows, _ = load_amortization_rows(amort_path)
    reorder_time_lookup = build_reorder_time_lookup()
    written, skipped, unresolved = phase3_compute(
        todo,
        bench_rows,
        amort_rows,
        reorder_time_lookup,
        force=args.force,
    )
    if written or amort_needs_rewrite:
        write_amortization_rows(amort_path, amort_rows)

    print(
        f"Phase 3: wrote {written} amortization row(s), skipped {skipped}, "
        f"unresolved {len(unresolved)}",
        flush=True,
    )
    print(f"Amortization CSV: {amort_path}", flush=True)
    if unresolved:
        print("Unresolved:", flush=True)
        for msg in unresolved[:50]:
            print(f"  - {msg}", flush=True)
        if len(unresolved) > 50:
            print(f"  ... and {len(unresolved) - 50} more", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
