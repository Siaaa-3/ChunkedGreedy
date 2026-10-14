from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import scipy.sparse

from reorder_bench.config import resolve_groot_cpp_effective
from reorder_bench.loader import prep_groot_csr, ori2new_to_row_perm
from reorder_bench.timing import timed_ms
from reorder_bench.types import ReorderResult


def _load_native():
    try:
        return importlib.import_module("reorder_bench.groot_reorder_core")
    except ImportError:
        try:
            return importlib.import_module("groot_reorder_core")
        except ImportError as exc:
            raise ImportError(
                "groot-cpp requires native extension groot_reorder_core. "
                "Run: ./scripts/build_native.sh groot-cpp"
            ) from exc


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    prep_csr, prep_ms = prep_groot_csr(csr)
    nrows = prep_csr.shape[0]
    effective = resolve_groot_cpp_effective(params, nrows)

    groot_reorder_core = _load_native()

    breakdown: dict[str, float] = {}
    with timed_ms() as bucket:
        ori2new, parts = groot_reorder_core.reorder(
            prep_csr.indptr.astype("int64"),
            prep_csr.indices.astype("int64"),
            nrows,
            effective["i_k"],
            effective["i_l"],
            effective["s_k"],
            effective["iterations"],
        )
        breakdown.update(parts)
    reorder_time_ms = sum(breakdown.values()) if breakdown else bucket[0]

    with timed_ms() as bucket:
        row_perm = ori2new_to_row_perm(np.asarray(ori2new, dtype=np.int64))
    perm_trans_ms = bucket[0]

    return ReorderResult(
        row_perm=row_perm,
        reorder_time_ms=reorder_time_ms,
        prep_time_ms=prep_ms,
        perm_trans_ms=perm_trans_ms,
        timing_breakdown=breakdown,
        effective_params=effective,
    )
