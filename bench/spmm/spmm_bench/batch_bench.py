"""Batch SpMM benchmark over multiple matrices, methods, and feature dims."""
from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Sequence

import spmatkit
import torch

from .base import SpMMMethod
from .paths import CURRENT_RUN_FILE, RESULTS_DIR
from .utils import generate_dense, recover_cuda_after_error

DEFAULT_METHODS: tuple[str, ...] = (
    "voltrix",
    "flashsparse",
    "dtc-spmm",
    "acc-spmm",
)

DEFAULT_FEATURE_DIMS: tuple[int, ...] = (32, 64, 128, 256)

DEFAULT_OUTPUT_CSV: Path = RESULTS_DIR / "spmm_bench.csv"
DEFAULT_AMORTIZATION_CSV: Path = RESULTS_DIR / "reorder_spmm_amortization.csv"

WriteMode = Literal["append", "force", "new-write"]

CSV_FIELDS: tuple[str, ...] = (
    "matrix",
    "reorder_method",
    "reorder_hash",
    "reorder_time_ms",
    "num_rows",
    "num_cols",
    "nnz",
    "spmm_method",
    "spmm_tc_h",
    "spmm_tc_w",
    "feature_dim",
    "time_ms",
    "prepare_time_ms",
    "success",
    "error",
    "warmup",
    "iterations",
)

AMORTIZATION_FIELDS: tuple[str, ...] = (
    "matrix",
    "reorder_method",
    "reorder_hash",
    "spmm_method",
    "spmm_tc_h",
    "spmm_tc_w",
    "feature_dim",
    "reorder_time_ms",
    "spmm_orig_ms",
    "spmm_reord_ms",
    "spmm_reduction_ms",
    "amortization",
    "roi",
)

# Catalog key -> reorder_time_ms text (ORIGIN always "").
ReorderTimeKey = tuple[str, str, str]

# Uniquely identifies a benchmark row:
# (matrix, reorder_method, reorder_hash, spmm_method, spmm_tc_h, spmm_tc_w, feature_dim).
ResultKey = tuple[str, str, str, str, int, int, int]

_FILENAME_TIMESTAMP_RE = re.compile(r"(\d{8}_\d{6})")


def result_key(
    matrix: str,
    reorder_method: str,
    reorder_hash: str,
    spmm_method: str,
    spmm_tc_h: int,
    spmm_tc_w: int,
    feature_dim: int,
) -> ResultKey:
    return (
        matrix,
        reorder_method,
        reorder_hash,
        spmm_method,
        int(spmm_tc_h),
        int(spmm_tc_w),
        int(feature_dim),
    )


def result_key_from_row(row: dict[str, object]) -> ResultKey:
    return result_key(
        str(row["matrix"]),
        str(row["reorder_method"]),
        str(row["reorder_hash"]),
        str(row["spmm_method"]),
        int(row["spmm_tc_h"]),
        int(row["spmm_tc_w"]),
        int(row["feature_dim"]),
    )


def _validate_csv_fields(fieldnames: Sequence[str] | None, path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"CSV has no header: {path}")
    if tuple(fieldnames) != CSV_FIELDS:
        raise ValueError(
            f"Existing CSV has unexpected columns: {list(fieldnames)!r}; "
            f"expected {list(CSV_FIELDS)} ({path})"
        )


def load_result_rows(csv_path: Path) -> dict[ResultKey, dict[str, str]]:
    """Load rows keyed by ResultKey (empty dict if file is missing/empty)."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {}

    rows: dict[ResultKey, dict[str, str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        _validate_csv_fields(reader.fieldnames, csv_path)
        for row in reader:
            key = result_key_from_row(row)
            # Within a single file, keep the first occurrence.
            rows.setdefault(key, {field: str(row.get(field, "")) for field in CSV_FIELDS})
    return rows


def load_completed_result_keys(csv_path: Path) -> set[ResultKey]:
    """Load keys of *successful* rows (``success == 1``) for append/skip logic.

    Failed rows are intentionally omitted so append mode can retry them.
    """
    return {
        key
        for key, row in load_result_rows(csv_path).items()
        if str(row.get("success", "")).strip() == "1"
    }

def write_result_rows(csv_path: Path, rows: dict[ResultKey, dict[str, str]]) -> None:
    """Atomically rewrite ``csv_path`` with the given keyed rows."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    tmp_path.replace(csv_path)


def _filename_timestamp(path: Path) -> str | None:
    match = _FILENAME_TIMESTAMP_RE.search(path.name)
    return match.group(1) if match else None


def iter_result_csv_files(results_dir: Path) -> list[Path]:
    """All ``*.csv`` under ``results_dir`` (recursive), excluding merge temps."""
    files = [
        p
        for p in results_dir.rglob("*.csv")
        if p.is_file() and not p.name.endswith(".tmp")
    ]
    return files


def merge_result_csvs(
    source_files: Sequence[Path],
    output_csv: Path,
) -> tuple[int, int]:
    """Merge CSVs into ``output_csv``.

    Files are applied oldest-first so later sources overwrite on key collision.
    Ordering: no-timestamp files by mtime ascending, then timestamped files by
    filename stamp ascending. Within one file, the first row for a key is kept.

    Returns ``(n_source_rows, n_unique_keys)``.
    """
    stamped: list[tuple[str, Path]] = []
    unstamped: list[tuple[float, Path]] = []
    for path in source_files:
        stamp = _filename_timestamp(path)
        if stamp is not None:
            stamped.append((stamp, path))
        else:
            unstamped.append((path.stat().st_mtime, path))

    ordered = [p for _, p in sorted(unstamped, key=lambda t: t[0])]
    ordered.extend(p for _, p in sorted(stamped, key=lambda t: t[0]))

    merged: dict[ResultKey, dict[str, str]] = {}
    n_source = 0
    for path in ordered:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            _validate_csv_fields(reader.fieldnames, path)
            seen_in_file: set[ResultKey] = set()
            for raw in reader:
                n_source += 1
                key = result_key_from_row(raw)
                if key in seen_in_file:
                    continue
                seen_in_file.add(key)
                merged[key] = {field: str(raw.get(field, "")) for field in CSV_FIELDS}

    write_result_rows(output_csv, merged)
    return n_source, len(merged)


@dataclass
class BatchBenchConfig:
    dataset: str | None = "testDataset"
    matrices: Sequence[str] | None = None
    output_dir: Path = RESULTS_DIR
    output_csv: Path | None = None
    mode: WriteMode = "force"
    methods: Sequence[str] = DEFAULT_METHODS
    feature_dims: Sequence[int] = DEFAULT_FEATURE_DIMS
    warmup: int = 30
    iterations: int = 100
    seed: int = 42
    max_memory: float = 128
    load_mode: str = "full"
    fill_value: float = 1.0
    # None + reorder_hash → method looked up from registry/catalog.
    # None without hash → ORIGIN. Method+hash mismatch raises at resolve time.
    reorder_methods: str | Sequence[str] | None = None
    reorder_hash: str | None = None
    params: dict | None = None
    all_param_sets: bool = False


@dataclass
class _MethodState:
    wrapper: SpMMMethod
    prepared: object | None = None
    prepare_time_ms: float | None = None


def build_reorder_time_lookup() -> dict[ReorderTimeKey, str]:
    """Map ``(matrix, reorder_method, reorder_hash)`` → ``reorder_time_ms`` text.

    ORIGIN variants are always ``""``. Missing catalog times are ``""``.
    """
    lookup: dict[ReorderTimeKey, str] = {}
    for entry in spmatkit.list_catalog_entries(all_param_sets=True):
        key = (
            str(entry["matrix"]),
            str(entry["reorder_method"]),
            str(entry["reorder_hash"]),
        )
        if key[1] == spmatkit.ORIGIN_METHOD:
            lookup[key] = ""
            continue
        rt = entry.get("reorder_time_ms")
        lookup[key] = "" if rt is None else f"{float(rt):.10g}"
    return lookup


def format_reorder_time_ms(value: float | None, *, reorder_method: str) -> str:
    if reorder_method == spmatkit.ORIGIN_METHOD or value is None:
        return ""
    return f"{float(value):.10g}"


@dataclass
class BatchBenchRunner:
    """Run benchmarks with minimal redundant I/O and preprocessing."""

    config: BatchBenchConfig
    completed_keys: set[ResultKey] | None = None
    existing_rows: dict[ResultKey, dict[str, str]] | None = None
    reorder_time_lookup: dict[ReorderTimeKey, str] | None = None

    def _output_path(self) -> Path:
        base = self.config.output_csv or DEFAULT_OUTPUT_CSV
        if self.config.mode == "new-write":
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return base.parent / f"{base.stem}_{stamp}{base.suffix}"
        return base

    def _reorder_time_ms(self, matrix: str, reorder_method: str, reorder_hash: str) -> str:
        if reorder_method == spmatkit.ORIGIN_METHOD:
            return ""
        assert self.reorder_time_lookup is not None
        return self.reorder_time_lookup.get((matrix, reorder_method, reorder_hash), "")

    def _is_completed(
        self,
        matrix: str,
        reorder_method: str,
        reorder_hash: str,
        spmm_method: str,
        spmm_tc_h: int,
        spmm_tc_w: int,
        feature_dim: int,
    ) -> bool:
        if self.config.mode != "append" or self.completed_keys is None:
            return False
        return (
            result_key(
                matrix,
                reorder_method,
                reorder_hash,
                spmm_method,
                spmm_tc_h,
                spmm_tc_w,
                feature_dim,
            )
            in self.completed_keys
        )

    def _pending_feature_dims(
        self,
        matrix: str,
        reorder_method: str,
        reorder_hash: str,
        spmm_method: str,
        spmm_tc_h: int,
        spmm_tc_w: int,
        wrapper: SpMMMethod,
    ) -> list[int]:
        dims = (
            self.config.feature_dims
            if wrapper.uses_feature_dim
            else (0,)
        )
        return [
            dim
            for dim in dims
            if not self._is_completed(
                matrix,
                reorder_method,
                reorder_hash,
                spmm_method,
                spmm_tc_h,
                spmm_tc_w,
                dim,
            )
        ]

    def _matrix_has_pending(
        self,
        matrix: str,
        reorder_method: str,
        reorder_hash: str,
        states: dict[str, _MethodState],
    ) -> bool:
        return any(
            self._pending_feature_dims(
                matrix,
                reorder_method,
                reorder_hash,
                method,
                state.wrapper.tc_h,
                state.wrapper.tc_w,
                state.wrapper,
            )
            for method, state in states.items()
        )

    def _expected_reorder_hash(self, reorder_method: str) -> str:
        """Best-effort hash for failure rows when a variant cannot be loaded."""
        if reorder_method == spmatkit.ORIGIN_METHOD:
            return spmatkit.ORIGIN_HASH
        if self.config.reorder_hash is not None:
            return self.config.reorder_hash
        params = spmatkit.resolve_params(reorder_method, self.config.params or {})
        env = spmatkit.resolve_env(reorder_method, {})
        return spmatkit.compute_reorder_hash(reorder_method, params, env)

    def _matrix_names(self) -> list[str]:
        if self.config.matrices is not None:
            return list(self.config.matrices)
        if not self.config.dataset:
            raise ValueError("dataset or matrices must be provided")
        return spmatkit.get_dataset_names(self.config.dataset)

    def _load_reorder_variants(
        self, matrix: str, reorder_method: str
    ) -> list[tuple[str, str, str, object]]:
        """Return ``[(matrix, reorder_method, reorder_hash, csr), ...]``."""
        if (
            reorder_method == spmatkit.ORIGIN_METHOD
            and not self.config.all_param_sets
            and self.config.reorder_hash is None
            and self.config.params is None
        ):
            csr = spmatkit.load_matrix(
                matrix,
                load_mode=self.config.load_mode,
                fill_value=self.config.fill_value,
            )
            return [(matrix, spmatkit.ORIGIN_METHOD, spmatkit.ORIGIN_HASH, csr)]

        # Resolve variants via spmatkit. Default mode uses schema defaults from
        # reordering-bench/methods.json (via spmatkit.resolve_params), not
        # ``all_param_sets=True`` + ``items[:1]`` — catalog order is by
        # reorder_hash, so the first entry is often a non-default param set.
        if self.config.all_param_sets:
            items = spmatkit.load_matrix(
                matrix,
                reorder_method=reorder_method,
                reorder_hash=self.config.reorder_hash,
                params=self.config.params,
                all_param_sets=True,
                load_mode=self.config.load_mode,
                fill_value=self.config.fill_value,
            )
            return [
                (item["matrix"], item["reorder_method"], item["reorder_hash"], item["csr"])
                for item in items
            ]

        reorder_hash = self.config.reorder_hash
        if reorder_hash is None:
            reorder_hash = self._expected_reorder_hash(reorder_method)
        csr = spmatkit.load_matrix(
            matrix,
            reorder_method=reorder_method,
            reorder_hash=reorder_hash,
            params=None if self.config.reorder_hash is not None else self.config.params,
            all_param_sets=False,
            load_mode=self.config.load_mode,
            fill_value=self.config.fill_value,
        )
        return [(matrix, reorder_method, reorder_hash, csr)]

    def _iter_load_failure_rows(
        self,
        matrix: str,
        reorder_method: str,
        reorder_hash: str,
        err: str,
        states: dict[str, _MethodState],
    ) -> Iterable[dict[str, object]]:
        num_rows: object = ""
        num_cols: object = ""
        nnz: object = ""
        try:
            base = spmatkit.load_matrix(
                matrix,
                load_mode=self.config.load_mode,
                fill_value=self.config.fill_value,
            )
            num_rows, num_cols = base.shape
            nnz = int(base.nnz)
        except Exception:  # noqa: BLE001
            pass

        for method_name, state in states.items():
            tc_h = int(state.wrapper.tc_h)
            tc_w = int(state.wrapper.tc_w)
            pending_dims = self._pending_feature_dims(
                matrix,
                reorder_method,
                reorder_hash,
                method_name,
                tc_h,
                tc_w,
                state.wrapper,
            )
            for dim in pending_dims:
                key = result_key(
                    matrix,
                    reorder_method,
                    reorder_hash,
                    method_name,
                    tc_h,
                    tc_w,
                    dim,
                )
                yield {
                    "matrix": matrix,
                    "reorder_method": reorder_method,
                    "reorder_hash": reorder_hash,
                    "reorder_time_ms": self._reorder_time_ms(
                        matrix, reorder_method, reorder_hash
                    ),
                    "num_rows": num_rows,
                    "num_cols": num_cols,
                    "nnz": nnz,
                    "spmm_method": method_name,
                    "spmm_tc_h": tc_h,
                    "spmm_tc_w": tc_w,
                    "feature_dim": dim,
                    "time_ms": "",
                    "prepare_time_ms": "",
                    "success": 0,
                    "error": err,
                    "warmup": self.config.warmup,
                    "iterations": self.config.iterations,
                }
                if self.completed_keys is not None:
                    self.completed_keys.add(key)

    def _build_methods(self) -> dict[str, _MethodState]:
        states: dict[str, _MethodState] = {}
        for name in self.config.methods:
            wrapper = SpMMMethod.create(name)
            wrapper.build()
            states[name] = _MethodState(wrapper=wrapper)
        return states

    @staticmethod
    def _time_run(
        wrapper: SpMMMethod,
        prepared: object,
        B: torch.Tensor | None,
        warmup: int,
        iterations: int,
    ) -> float:
        for _ in range(warmup):
            wrapper.run(prepared, B)
        total_ms = 0.0
        for _ in range(iterations):
            _, kernel_ms = wrapper.run(prepared, B)
            total_ms += kernel_ms
        return total_ms / iterations

    def _prepare_method(self, state: _MethodState, A_csr) -> None:
        if state.prepared is not None:
            return
        use_cuda = state.wrapper.uses_cuda
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        state.prepared = state.wrapper.prepare(A_csr)
        if use_cuda:
            torch.cuda.synchronize()
        state.prepare_time_ms = (time.perf_counter() - t0) * 1000.0

    def _reset_prepare_cache(self, states: dict[str, _MethodState]) -> None:
        for state in states.values():
            state.prepared = None
            state.prepare_time_ms = None

    @staticmethod
    def _announce_current_run(info: dict[str, object]) -> None:
        text = "\n".join(f"{k}: {v}" for k, v in info.items()) + "\n"
        print(text, end="", flush=True)
        CURRENT_RUN_FILE.write_text(text, encoding="utf-8")

    @staticmethod
    def _is_cuda_runtime_error(exc: BaseException) -> bool:
        if not isinstance(exc, RuntimeError):
            return False
        msg = str(exc).lower()
        return "cuda error" in msg or "illegal memory access" in msg

    def _handle_run_failure(
        self,
        exc: BaseException,
        states: dict[str, _MethodState],
    ) -> None:
        if self._is_cuda_runtime_error(exc):
            recover_cuda_after_error(exc)
            self._reset_prepare_cache(states)

    def _iter_rows_for_matrix(
        self,
        matrix: str,
        reorder_method: str,
        reorder_hash: str,
        A_csr,
        states: dict[str, _MethodState],
    ) -> Iterable[dict[str, object]]:
        num_rows, num_cols = A_csr.shape
        nnz = int(A_csr.nnz)

        self._reset_prepare_cache(states)

        for method_name, state in states.items():
            tc_h = int(state.wrapper.tc_h)
            tc_w = int(state.wrapper.tc_w)
            pending_dims = self._pending_feature_dims(
                matrix, reorder_method, reorder_hash, method_name, tc_h, tc_w,
                state.wrapper,
            )
            if not pending_dims:
                continue

            base_row = {
                "matrix": matrix,
                "reorder_method": reorder_method,
                "reorder_hash": reorder_hash,
                "reorder_time_ms": self._reorder_time_ms(
                    matrix, reorder_method, reorder_hash
                ),
                "num_rows": num_rows,
                "num_cols": num_cols,
                "nnz": nnz,
                "spmm_tc_h": tc_h,
                "spmm_tc_w": tc_w,
                "spmm_method": method_name,
                "warmup": self.config.warmup,
                "iterations": self.config.iterations,
            }

            try:
                try:
                    self._announce_current_run(
                        {
                            "phase": "prepare",
                            "matrix": matrix,
                            "reorder_method": reorder_method,
                            "reorder_hash": reorder_hash,
                            "num_rows": num_rows,
                            "num_cols": num_cols,
                            "nnz": nnz,
                            "spmm_method": method_name,
                            "spmm_tc_h": tc_h,
                            "spmm_tc_w": tc_w,
                            "warmup": self.config.warmup,
                            "iterations": self.config.iterations,
                        }
                    )
                    self._prepare_method(state, A_csr)
                    prepare_str = f"{state.prepare_time_ms:.6f}"
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {exc}"
                    self._handle_run_failure(exc, states)
                    for dim in pending_dims:
                        key = result_key(
                            matrix,
                            reorder_method,
                            reorder_hash,
                            method_name,
                            tc_h,
                            tc_w,
                            dim,
                        )
                        yield {
                            **base_row,
                            "feature_dim": dim,
                            "time_ms": "",
                            "prepare_time_ms": "",
                            "success": 0,
                            "error": err,
                        }
                        if self.completed_keys is not None:
                            self.completed_keys.add(key)
                    continue

                for i, dim in enumerate(pending_dims):
                    row: dict[str, object] = {
                        **base_row,
                        "feature_dim": dim,
                        "time_ms": "",
                        "prepare_time_ms": prepare_str if i == 0 else "",
                        "success": 0,
                        "error": "",
                    }
                    try:
                        self._announce_current_run(
                            {
                                "phase": "run",
                                "matrix": matrix,
                                "reorder_method": reorder_method,
                                "reorder_hash": reorder_hash,
                                "num_rows": num_rows,
                                "num_cols": num_cols,
                                "nnz": nnz,
                                "spmm_method": method_name,
                                "feature_dim": dim,
                                "spmm_tc_h": tc_h,
                                "spmm_tc_w": tc_w,
                                "warmup": self.config.warmup,
                                "iterations": self.config.iterations,
                            }
                        )
                        if state.wrapper.uses_feature_dim:
                            B = generate_dense(num_rows, dim, seed=self.config.seed)
                            elapsed = self._time_run(
                                state.wrapper,
                                state.prepared,
                                B,
                                self.config.warmup,
                                self.config.iterations,
                            )
                        else:
                            elapsed = self._time_run(
                                state.wrapper,
                                state.prepared,
                                None,
                                self.config.warmup,
                                self.config.iterations,
                            )
                        row["time_ms"] = f"{elapsed:.6f}"
                        row["success"] = 1
                    except Exception as exc:  # noqa: BLE001
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        self._handle_run_failure(exc, states)
                    finally:
                        if "B" in locals() and B is not None:
                            del B
                    yield row
                    if self.completed_keys is not None:
                        self.completed_keys.add(
                            result_key(
                                matrix,
                                reorder_method,
                                reorder_hash,
                                method_name,
                                tc_h,
                                tc_w,
                                dim,
                            )
                        )
            finally:
                # Release per-method GPU buffers before the next method on this matrix.
                state.prepared = None
                state.prepare_time_ms = None
                torch.cuda.empty_cache()

    def _upsert_row(self, row: dict[str, object]) -> None:
        """Buffer a row into the in-memory table (all write modes).

        Re-inserts the key so updated/new rows are written at the end of the CSV
        (Python dict preserves insertion order).
        """
        assert self.existing_rows is not None
        key = result_key_from_row(row)
        self.existing_rows.pop(key, None)
        self.existing_rows[key] = {field: str(row.get(field, "")) for field in CSV_FIELDS}

    def run(self) -> Path:
        """Execute the full batch benchmark and write results incrementally."""
        mode = self.config.mode
        output_path = self._output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.reorder_time_lookup = build_reorder_time_lookup()

        if mode == "new-write":
            self.completed_keys = set()
            self.existing_rows = {}
            print(f"New-write mode: creating {output_path}", flush=True)
        elif mode == "append":
            self.existing_rows = load_result_rows(output_path)
            self.completed_keys = {
                key
                for key, row in self.existing_rows.items()
                if str(row.get("success", "")).strip() == "1"
            }
            n_failed = len(self.existing_rows) - len(self.completed_keys)
            print(
                f"Append mode {output_path}: skipping "
                f"{len(self.completed_keys)} successful result(s)"
                + (f"; will retry {n_failed} failed row(s)" if n_failed else ""),
                flush=True,
            )
        else:  # force
            self.completed_keys = set()  # never skip
            self.existing_rows = load_result_rows(output_path)
            print(
                f"Force mode {output_path}: "
                f"{len(self.existing_rows)} existing row(s); "
                f"matching keys will be overwritten",
                flush=True,
            )

        states = self._build_methods()
        rows_written = 0
        reorder_methods, reorder_hash = resolve_reorder_selection(
            self.config.reorder_methods,
            self.config.reorder_hash,
            self.config.params,
        )
        self.config.reorder_methods = reorder_methods
        self.config.reorder_hash = reorder_hash

        names = self._matrix_names()
        for rm in reorder_methods:
            for name in names:
                try:
                    variants = self._load_reorder_variants(name, rm)
                except Exception as exc:  # noqa: BLE001
                    err = f"{type(exc).__name__}: {exc}"
                    reorder_hash = self._expected_reorder_hash(rm)
                    if not self._matrix_has_pending(name, rm, reorder_hash, states):
                        continue
                    print(
                        f"Warning: skip {name!r} / {rm!r}: {err}",
                        flush=True,
                    )
                    matrix_rows = 0
                    for row in self._iter_load_failure_rows(
                        name, rm, reorder_hash, err, states
                    ):
                        self._upsert_row(row)
                        rows_written += 1
                        matrix_rows += 1
                    if matrix_rows:
                        write_result_rows(output_path, self.existing_rows)
                    continue

                for matrix, reorder_method, reorder_hash, A_csr in variants:
                    if not self._matrix_has_pending(
                        matrix, reorder_method, reorder_hash, states
                    ):
                        continue
                    matrix_rows = 0
                    for row in self._iter_rows_for_matrix(
                        matrix, reorder_method, reorder_hash, A_csr, states
                    ):
                        self._upsert_row(row)
                        rows_written += 1
                        matrix_rows += 1
                    # Flush once per matrix variant (all SpMM methods/dims).
                    if matrix_rows:
                        write_result_rows(output_path, self.existing_rows)

        if mode == "new-write" and rows_written == 0 and self.existing_rows is not None:
            # Ensure an empty header-only file exists for a no-op write run.
            write_result_rows(output_path, self.existing_rows)

        if mode == "append" and rows_written == 0:
            print("No new benchmarks to run.", flush=True)
        elif rows_written:
            print(f"Wrote {rows_written} result row(s).", flush=True)

        return output_path


def _normalize_reorder_methods(
    reorder_methods: str | Sequence[str] | None,
) -> list[str] | None:
    if reorder_methods is None:
        return None
    if isinstance(reorder_methods, str):
        methods = [reorder_methods]
    else:
        methods = [str(m) for m in reorder_methods]
    if not methods:
        raise ValueError("reorder_methods must contain at least one method name")
    if any(not m for m in methods):
        raise ValueError("reorder_methods entries must be non-empty strings")
    return methods


def resolve_reorder_selection(
    reorder_methods: str | Sequence[str] | None,
    reorder_hash: str | None,
    params: dict | None = None,
) -> tuple[list[str], str | None]:
    """Resolve CLI/API reorder selection; raise on method/hash mismatch.

    * ``reorder_methods is None`` and no hash → ``[ORIGIN]``
    * hash only → look up method from registry/catalog
    * method + hash → require the hash to belong to that method
    """
    methods = _normalize_reorder_methods(reorder_methods)
    if methods is not None and len(methods) > 1 and (
        reorder_hash is not None or params is not None
    ):
        raise ValueError(
            "reorder_hash / params apply to a single reorder method; "
            "pass one method, or omit hash/params when listing several"
        )

    if reorder_hash is None:
        if methods is None:
            return [spmatkit.ORIGIN_METHOD], None
        return methods, None

    method_arg = None if methods is None else methods[0]
    resolved_method, resolved_hash = spmatkit.resolve_reorder_method_and_hash(
        method_arg, reorder_hash
    )
    return [resolved_method], resolved_hash


def run_batch_benchmark(
    dataset: str | None = "testDataset",
    matrices: Sequence[str] | None = None,
    output_dir: str | Path = RESULTS_DIR,
    output_csv: str | Path | None = None,
    mode: WriteMode = "force",
    methods: Sequence[str] = DEFAULT_METHODS,
    feature_dims: Sequence[int] = DEFAULT_FEATURE_DIMS,
    warmup: int = 30,
    iterations: int = 100,
    seed: int = 42,
    max_memory: float = 128,
    load_mode: str = "full",
    fill_value: float = 1.0,
    reorder_methods: str | Sequence[str] | None = None,
    reorder_hash: str | None = None,
    params: dict | None = None,
    all_param_sets: bool = False,
) -> Path:
    """Run benchmark on a spmatkit dataset or an explicit list of matrix names.

    By default the original (``ORIGIN``) matrices are benchmarked. Pass
    ``reorder_methods`` (string or list; with ``params``/``reorder_hash`` or
    ``all_param_sets``) to benchmark reordered variants. Hash-only resolves
    the method from the registry/catalog. ``params`` / ``reorder_hash``
    require a single method; method+hash mismatches raise.

    Modes (all keyed off ``output_csv``, default ``results/spmm_bench.csv``):
    - ``force`` (default): write to ``output_csv``; re-run requested keys and
      overwrite matching rows (moved to the end of the CSV). Flushed per matrix.
    - ``append``: skip keys already present with ``success=1``; retry failed
      rows and overwrite them (also moved to the end); append truly missing
      keys. Flushed per matrix.
    - ``new-write``: create a new timestamped file next to ``output_csv``
      (``<stem>_<timestamp>.csv``) and run fully.

    ``output_dir`` is retained for backward compatibility but is no longer used
    to derive the output path.
    """
    if mode not in ("append", "force", "new-write"):
        raise ValueError(f"Invalid mode {mode!r}; expected append|force|new-write")

    resolved_csv = Path(output_csv) if output_csv is not None else DEFAULT_OUTPUT_CSV

    runner = BatchBenchRunner(
        BatchBenchConfig(
            dataset=dataset,
            matrices=matrices,
            output_dir=Path(output_dir),
            output_csv=resolved_csv,
            mode=mode,
            methods=methods,
            feature_dims=feature_dims,
            warmup=warmup,
            iterations=iterations,
            seed=seed,
            max_memory=max_memory,
            load_mode=load_mode,
            fill_value=fill_value,
            reorder_methods=reorder_methods,
            reorder_hash=reorder_hash,
            params=params,
            all_param_sets=all_param_sets,
        )
    )
    return runner.run()
