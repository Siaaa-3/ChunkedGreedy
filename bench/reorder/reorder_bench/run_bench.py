from __future__ import annotations

import json
import multiprocessing as mp
import queue
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import scipy.sparse

import spmatkit

from reorder_bench.config import (
    DEFAULT_TC_H,
    DEFAULT_TC_W,
    METHOD_NAMES,
    merge_params,
    method_env_json,
    env_for_hash,
    params_for_hash,
    resolve_effective_params,
    apply_runtime_env,
    resolve_runtime_env,
    sync_method_schemas_to_spmatkit,
)
from reorder_bench.metrics import compute_metrics
from reorder_bench.results import (
    append_text_line,
    default_run_id,
    load_completed_keys,
    make_timestamp,
    source_commit,
    write_result_row,
)
from reorder_bench.utils import recover_gpu_after_error
from reorder_bench.types import ReorderResult

METHOD_MODULES: dict[str, Callable] = {}

# Native / C-extension methods: SIGALRM cannot interrupt them promptly, so
# reorder runs in a child process that can be terminate/kill'd on timeout.
# Parent still loads each matrix once and passes the in-memory CSR.
PROCESS_TIMEOUT_METHODS: frozenset[str] = frozenset(
    {
        "acc",
        "rabbit-local-greedy",
        "groot-cpp",
        "clusterwise-hier",
        "rcm",
        "amd",
        "degree",
        "gray",
        "slashburn",
        "nd",
        "gp",
    }
)


class _ReorderTimeout(BaseException):
    """Raised when the reorder stage exceeds the configured time limit."""


@contextmanager
def _time_limit(seconds: float) -> Iterator[None]:
    """Soft time limit using SIGALRM (Linux/Unix; Python-level only)."""

    def _handler(signum: int, frame: Any) -> None:  # noqa: ARG001
        raise _ReorderTimeout(f"timeout after {seconds:.0f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _mp_context() -> mp.context.BaseContext:
    # fork: share parent CSR pages (COW); safe here because these methods are CPU-only.
    # spawn: used off-Linux where fork is unavailable.
    if sys.platform.startswith("linux"):
        return mp.get_context("fork")
    return mp.get_context("spawn")


def _reorder_process_worker(
    method: str,
    csr: scipy.sparse.csr_matrix,
    params: dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    """Child entry: run one reorder and put (ok, result) or (err, message)."""
    try:
        apply_runtime_env()
        _register_methods()
        result = METHOD_MODULES[method](csr, params)
        result_queue.put(("ok", result))
    except BaseException as exc:  # noqa: BLE001 — surface any child failure to parent
        result_queue.put(("err", f"{type(exc).__name__}: {exc}"))


def _run_with_process_timeout(
    method: str,
    csr: scipy.sparse.csr_matrix,
    params: dict[str, Any],
    seconds: float,
) -> ReorderResult:
    """Run reorder in a child process; kill it if wall time exceeds ``seconds``."""
    ctx = _mp_context()
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_reorder_process_worker,
        args=(method, csr, params, result_queue),
    )
    proc.start()
    proc.join(timeout=seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=10.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=10.0)
        raise _ReorderTimeout(f"timeout after {seconds:.0f}s")

    try:
        kind, payload = result_queue.get(timeout=60.0)
    except queue.Empty as exc:
        code = proc.exitcode
        raise RuntimeError(
            f"reorder subprocess exited with code {code} and returned no result"
        ) from exc

    if kind == "ok":
        return payload
    raise RuntimeError(f"reorder subprocess failed: {payload}")


def _call_reorder(
    method: str,
    csr: scipy.sparse.csr_matrix,
    params: dict[str, Any],
    timeout_h: float | None,
) -> ReorderResult:
    """Dispatch reorder with the appropriate timeout mechanism."""
    if timeout_h is None or timeout_h < 0:
        return METHOD_MODULES[method](csr, params)

    seconds = timeout_h * 3600.0
    if method in PROCESS_TIMEOUT_METHODS:
        return _run_with_process_timeout(method, csr, params, seconds)

    with _time_limit(seconds):
        return METHOD_MODULES[method](csr, params)


@contextmanager
def _print_stage(
    label: str,
    *,
    matrix: str,
    method: str | None = None,
) -> Iterator[None]:
    ctx = f"{matrix} / {method}" if method else matrix
    print(f"[{ctx}] {label} ...", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    except BaseException:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[{ctx}] {label} 中断，用时 {elapsed_ms:.2f} ms", flush=True)
        raise
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"[{ctx}] {label} 完成，用时 {elapsed_ms:.2f} ms", flush=True)


_METHOD_MODULE_NAMES = {
    "acc": "acc",
    "groot-py": "groot_py",
    "groot-cpp": "groot_cpp",
    "dtc-tca": "dtc_tca",
    "clusterwise-hier": "clusterwise_hier",
    "rcm": "rcm",
    "amd": "amd",
    "degree": "degree",
    "gray": "gray",
    "slashburn": "slashburn",
    "nd": "nd",
    "gp": "gp",
    "random": "random_reorder",
    "global-greedy": "global_greedy",
    "rabbit-local-greedy": "rabbit_local_greedy",
    "sporder-cg": "sporder_cg",
    "sporder-csg": "sporder_csg",
    "sporder-hm": "sporder_hm",
}


def _register_methods() -> None:
    if METHOD_MODULES:
        return
    import importlib

    # Method modules absent from a trimmed checkout are skipped, not errors.
    for method, module in _METHOD_MODULE_NAMES.items():
        try:
            mod = importlib.import_module(f"reorder_bench.methods.{module}")
        except ModuleNotFoundError as exc:
            if exc.name != f"reorder_bench.methods.{module}":
                raise
            continue
        METHOD_MODULES[method] = mod.reorder


@dataclass
class RunConfig:
    matrices: list[str]
    methods: list[str]
    mode: str  # tune | save
    run_id: str
    output_csv: Path
    tc_h: int = DEFAULT_TC_H
    tc_w: int = DEFAULT_TC_W
    params_by_method: dict[str, dict[str, Any]] | None = None
    force: bool = False
    dataset: str | None = None
    timeout_h_by_method: dict[str, float] = field(default_factory=dict)


def run_single(
    base_name: str,
    method: str,
    csr: scipy.sparse.csr_matrix,
    *,
    load_matrix_ms: float,
    cfg: RunConfig,
    timeout_h: float | None = None,
) -> dict[str, Any]:
    _register_methods()
    if method not in METHOD_NAMES:
        raise ValueError(f"Unknown method: {method}")

    runtime = resolve_runtime_env()
    user_params = (cfg.params_by_method or {}).get(method, {})
    params = merge_params(method, user_params)
    hash_params = params_for_hash(method, params)
    nrows = csr.shape[0]
    hash_env = env_for_hash(method, runtime, nrows=nrows, params=params)
    params_hash = spmatkit.compute_reorder_hash(method, hash_params, hash_env)
    params_json = spmatkit.canonicalize_params(hash_params)

    row: dict[str, Any] = {
        "timestamp": make_timestamp(),
        "matrix": base_name,
        "reorder_method": method,
        "params_json": params_json,
        "reorder_hash": params_hash,
        "source_commit": source_commit(),
        "load_matrix_ms": f"{load_matrix_ms:.4f}",
    }

    try:
        with _print_stage("重排序", matrix=base_name, method=method):
            result: ReorderResult = _call_reorder(method, csr, params, timeout_h)
        effective = resolve_effective_params(method, params, csr.shape[0])
        if result.effective_params:
            effective.update(result.effective_params)

        env_extra = dict(result.env_extra)
        row["effective_params_json"] = json.dumps(effective, sort_keys=True)
        row["env_json"] = method_env_json(method, runtime, env_extra)
        hash_env = env_for_hash(method, runtime, env_extra, nrows=nrows, params=params)
        params_hash = spmatkit.compute_reorder_hash(method, hash_params, hash_env)
        row["reorder_hash"] = params_hash
        row["prep_ms"] = f"{result.prep_time_ms:.4f}"
        row["reorder_time_ms"] = f"{result.reorder_time_ms:.4f}"
        row["perm_trans_ms"] = f"{result.perm_trans_ms:.4f}"
        row["timing_breakdown_json"] = json.dumps(
            result.timing_breakdown, sort_keys=True
        )

        with _print_stage("计算指标", matrix=base_name, method=method):
            metric_vals, eval_ms = compute_metrics(
                csr, result.row_perm, tc_h=cfg.tc_h, tc_w=cfg.tc_w
            )
        row["eval_time_ms"] = f"{eval_ms:.4f}"
        for key in ("bandwidth", "profile", "tc_h", "tc_w", "tc_num", "tc_fill_rate"):
            if key in metric_vals:
                row[key] = metric_vals[key]

        row["success"] = "1"
        row["error"] = ""

        if cfg.mode == "save":
            with _print_stage("存储矩阵", matrix=base_name, method=method):
                spmatkit.save_reorder_perm(
                    base_name,
                    method,
                    hash_params,
                    result.row_perm,
                    env=hash_env,
                    source=f"reordering-bench@{row['source_commit']}",
                    reorder_time_ms=result.reorder_time_ms,
                    force=cfg.force,
                )
    except Exception as exc:
        recover_gpu_after_error(exc)
        row["effective_params_json"] = ""
        row["env_json"] = method_env_json(method, runtime)
        row["prep_ms"] = ""
        row["reorder_time_ms"] = ""
        row["perm_trans_ms"] = ""
        row["eval_time_ms"] = ""
        row["timing_breakdown_json"] = "{}"
        row["success"] = "0"
        row["error"] = str(exc)

    return row


def _hash_params_for_method(method: str, cfg: RunConfig) -> dict[str, Any]:
    user_params = (cfg.params_by_method or {}).get(method, {})
    params = merge_params(method, user_params)
    return params_for_hash(method, params)


def _hash_env_for_matrix(
    method: str,
    cfg: RunConfig,
    *,
    nrows: int,
    env_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = resolve_runtime_env()
    user_params = (cfg.params_by_method or {}).get(method, {})
    params = merge_params(method, user_params)
    return env_for_hash(method, runtime, env_extra, nrows=nrows, params=params)


def _variant_hash_for_matrix(
    method: str,
    cfg: RunConfig,
    *,
    nrows: int,
    env_extra: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    hash_params = _hash_params_for_method(method, cfg)
    hash_env = _hash_env_for_matrix(method, cfg, nrows=nrows, env_extra=env_extra)
    params_hash = spmatkit.compute_reorder_hash(method, hash_params, hash_env)
    return params_hash, hash_params, hash_env


def _should_skip(
    base_name: str,
    method: str,
    params_hash: str,
    hash_params: dict[str, Any],
    hash_env: dict[str, Any],
    cfg: RunConfig,
    completed_csv: set[tuple[str, str, str]],
) -> bool:
    if cfg.force:
        return False
    if cfg.mode == "tune":
        return (base_name, method, params_hash) in completed_csv
    return spmatkit.has_reorder_perm(base_name, method, hash_params, hash_env)


def _task_total(cfg: RunConfig) -> int:
    return len(cfg.matrices) * len(cfg.methods)


def _count_already_done(
    cfg: RunConfig,
    completed_csv: set[tuple[str, str, str]],
) -> int:
    from reorder_bench.loader import load_pattern_matrix

    done = 0
    for base_name in cfg.matrices:
        csr, _load_ms = load_pattern_matrix(base_name)
        nrows = csr.shape[0]
        for method in cfg.methods:
            params_hash, hash_params, hash_env = _variant_hash_for_matrix(
                method, cfg, nrows=nrows
            )
            if _should_skip(
                base_name,
                method,
                params_hash,
                hash_params,
                hash_env,
                cfg,
                completed_csv,
            ):
                done += 1
    return done


def _print_dataset_progress(
    dataset: str, done: int, total: int, matrix: str | None = None
) -> None:
    if matrix:
        print(
            f"[dataset {dataset}] reorder progress: {done}/{total} "
            f"(current: {matrix})"
        )
    else:
        print(f"[dataset {dataset}] reorder progress: {done}/{total}")


def run_benchmark(cfg: RunConfig) -> Path:
    from reorder_bench.loader import load_pattern_matrix

    apply_runtime_env()
    # Mirror methods.json defaults into reorder_methods.json display schemas
    # (resolve/catalog already read methods.json via spmatkit).
    sync_method_schemas_to_spmatkit(cfg.methods)
    completed_csv = load_completed_keys(cfg.output_csv) if cfg.mode == "tune" else set()

    skip_path = cfg.output_csv.parent / "skip.txt"
    # Make sure directory exists (e.g. when output-csv is overridden).
    skip_path.parent.mkdir(parents=True, exist_ok=True)

    timeout_h_by_method = cfg.timeout_h_by_method

    total_tasks = _task_total(cfg)
    done_tasks = 0
    if cfg.dataset:
        done_tasks = _count_already_done(cfg, completed_csv)
        _print_dataset_progress(cfg.dataset, done_tasks, total_tasks)

    for base_name in cfg.matrices:
        with _print_stage("读取矩阵", matrix=base_name):
            csr, load_ms = load_pattern_matrix(base_name)
        nrows = csr.shape[0]
        skip_matrix = False
        for method in cfg.methods:
            params_hash, hash_params, hash_env = _variant_hash_for_matrix(
                method, cfg, nrows=nrows
            )
            if _should_skip(
                base_name, method, params_hash, hash_params, hash_env, cfg, completed_csv
            ):
                if cfg.mode == "tune":
                    reason = f"already in {cfg.output_csv}"
                else:
                    reason = "already in spmatkit reorder catalog"
                print(
                    f"Skip {base_name} / {method} "
                    f"(params_hash={params_hash}) — {reason}"
                )
                if cfg.dataset:
                    _print_dataset_progress(
                        cfg.dataset, done_tasks, total_tasks, matrix=base_name
                    )
                continue
            if cfg.dataset:
                _print_dataset_progress(
                    cfg.dataset, done_tasks, total_tasks, matrix=base_name
                )
            timeout_h = timeout_h_by_method.get(method)
            t0 = time.perf_counter()
            try:
                row = run_single(
                    base_name,
                    method,
                    csr,
                    load_matrix_ms=load_ms,
                    cfg=cfg,
                    timeout_h=timeout_h,
                )
            except _ReorderTimeout as exc:
                elapsed_h = (time.perf_counter() - t0) / 3600.0
                # Stop sorting the current matrix and proceed with the next one.
                skip_matrix = True
                append_text_line(
                    skip_path,
                    f"{make_timestamp()} matrix={base_name} method={method} "
                    f"elapsed_h={elapsed_h:.4f} timeout_h={timeout_h:.2f} "
                    f"reason={exc}\n",
                )
                print(
                    f"Timeout: {base_name} / {method} 超过 {timeout_h:.2f} 小时，跳过该矩阵（继续下一个）",
                    flush=True,
                )
                recover_gpu_after_error(exc)
                break

            write_result_row(cfg.output_csv, row, overwrite=cfg.force)
            if cfg.mode == "tune" and row.get("success") == "1":
                completed_csv.add((base_name, method, params_hash))
            if cfg.dataset and row.get("success") == "1":
                done_tasks += 1
                _print_dataset_progress(
                    cfg.dataset, done_tasks, total_tasks, matrix=base_name
                )

        if skip_matrix:
            continue

    return cfg.output_csv
