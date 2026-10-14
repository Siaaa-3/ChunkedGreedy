from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse

from reorder_bench.loader import prep_clusterwise_csr, row_perm_from_order
from reorder_bench.timing import timed_ms
from reorder_bench.types import ReorderResult

# Matches clusterwise native HashSpGEMMTopK preflight (int64 index + double value).
_ELEM_BYTES = 8 + 8
_PREFLIGHT_HEADROOM_BYTES = 1 << 30


def _read_mem_available_bytes() -> int:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return 0
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            kb = int(line.split()[1])
            return kb * 1024
    return 0


def _max_column_degree(indices: np.ndarray, ncols: int) -> int:
    col_ids = np.asarray(indices, dtype=np.int64)
    if col_ids.size == 0:
        return 0
    counts = np.bincount(col_ids, minlength=ncols)
    return int(counts.max())


def _preflight_lower_bound(indices: np.ndarray, nrows: int, topk: int) -> None:
    """Reject only when a proven lower bound on A*A^T symbolic nnz implies OOM."""
    d_max = _max_column_degree(indices, nrows)
    if d_max == 0:
        return

    symbolic_nnz_lb = d_max * d_max
    c_bytes = symbolic_nnz_lb * _ELEM_BYTES
    need_bytes = c_bytes + c_bytes // 4 + _PREFLIGHT_HEADROOM_BYTES
    avail_bytes = _read_mem_available_bytes()
    if avail_bytes <= 0 or need_bytes <= avail_bytes:
        return

    raise RuntimeError(
        "HashSpGEMMTopK OOM (early preflight, lower bound): "
        f"A*A^T symbolic_nnz_lb={symbolic_nnz_lb} "
        f"needs ~{need_bytes // (1024**3)} GiB "
        f"(C only ~{c_bytes // (1024**3)} GiB; "
        f"rows={nrows}, topK={topk}, d_max={d_max}, "
        f"avail~={avail_bytes // (1024**3)} GiB)"
    )


def _load_native():
    try:
        return importlib.import_module("reorder_bench.clusterwise_reorder_core")
    except ImportError:
        try:
            return importlib.import_module("clusterwise_reorder_core")
        except ImportError as exc:
            raise ImportError(
                "clusterwise-hier requires native extension clusterwise_reorder_core. "
                "Run: ./scripts/build_native.sh clusterwise"
            ) from exc


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    prep_csr, prep_ms = prep_clusterwise_csr(csr)
    nrows = prep_csr.shape[0]
    topk = int(params["topk"])
    cluster_size = int(params["cluster_size"])

    _preflight_lower_bound(prep_csr.indices, nrows, topk)

    clusterwise_reorder_core = _load_native()

    breakdown: dict[str, float] = {}
    with timed_ms() as bucket:
        reordered_dict, parts = clusterwise_reorder_core.reorder(
            prep_csr.indptr.astype("int64"),
            prep_csr.indices.astype("int64"),
            prep_csr.data.astype("float32"),
            nrows,
            topk,
            cluster_size,
        )
        breakdown.update(parts)
    reorder_time_ms = sum(breakdown.values()) if breakdown else bucket[0]

    with timed_ms() as bucket:
        order = clusterwise_reorder_core.clusters_to_row_perm(reordered_dict)
        row_perm = row_perm_from_order(order, nrows)
    perm_trans_ms = bucket[0]

    return ReorderResult(
        row_perm=row_perm,
        reorder_time_ms=reorder_time_ms,
        prep_time_ms=prep_ms,
        perm_trans_ms=perm_trans_ms,
        timing_breakdown=breakdown,
    )
