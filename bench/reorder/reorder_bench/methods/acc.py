from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import scipy.sparse

from reorder_bench.loader import csr_to_acc_adjacency, row_perm_from_order
from reorder_bench.timing import timed_ms
from reorder_bench.types import ReorderResult


def _load_native():
    try:
        return importlib.import_module("reorder_bench.acc_reorder_core")
    except ImportError:
        try:
            return importlib.import_module("acc_reorder_core")
        except ImportError as top_level_exc:
            raise ImportError(
                "acc requires native extension acc_reorder_core. "
                "Run: ./scripts/build_native.sh acc"
            ) from top_level_exc


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    adj, prep_ms = csr_to_acc_adjacency(csr)
    nrows = csr.shape[0]

    acc_reorder_core = _load_native()

    breakdown: dict[str, float] = {}
    with timed_ms() as bucket:
        order = acc_reorder_core.reorder_vertex(adj)
    breakdown["re2order_vertex_ms"] = bucket[0]

    with timed_ms() as bucket:
        row_perm = row_perm_from_order(np.asarray(order, dtype=np.int64), nrows)
    perm_trans_ms = bucket[0]

    return ReorderResult(
        row_perm=row_perm,
        reorder_time_ms=breakdown["re2order_vertex_ms"],
        prep_time_ms=prep_ms,
        perm_trans_ms=perm_trans_ms,
        timing_breakdown=breakdown,
    )
