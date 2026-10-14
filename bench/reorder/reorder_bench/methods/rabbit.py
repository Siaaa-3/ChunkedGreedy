from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import scipy.sparse

from reorder_bench.loader import csr_to_int64_arrays, ori2new_to_row_perm
from reorder_bench.timing import timed_ms
from reorder_bench.types import ReorderResult


def _load_rabbit():
    try:
        return importlib.import_module("reorder_bench.rabbit_reorder_core")
    except ImportError:
        try:
            return importlib.import_module("rabbit_reorder_core")
        except ImportError as exc:
            raise ImportError(
                "rabbit requires native extension rabbit_reorder_core. "
                "Run: ./scripts/build_native.sh rabbit"
            ) from exc


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    del params
    nrows = csr.shape[0]
    row_ptr, col, prep_ms = csr_to_int64_arrays(csr)
    ext = _load_rabbit()

    with timed_ms() as bucket:
        ori2new = ext.rabbit_reorder(row_ptr, col, nrows)
    reorder_ms = bucket[0]

    with timed_ms() as bucket:
        row_perm = ori2new_to_row_perm(np.asarray(ori2new, dtype=np.int64))
    perm_trans_ms = bucket[0]

    return ReorderResult(
        row_perm=row_perm,
        reorder_time_ms=reorder_ms,
        prep_time_ms=prep_ms,
        perm_trans_ms=perm_trans_ms,
    )
