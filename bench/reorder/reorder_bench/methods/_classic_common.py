from __future__ import annotations

import importlib
from collections import defaultdict
from typing import Any

import numpy as np
import scipy.sparse

from reorder_bench.loader import csr_to_int64_arrays, ori2new_to_row_perm
from reorder_bench.timing import timed_ms
from reorder_bench.types import ReorderResult


def _load_sparsebase():
    try:
        return importlib.import_module("reorder_bench.sparsebase_reorder_core")
    except ImportError:
        try:
            return importlib.import_module("sparsebase_reorder_core")
        except ImportError as exc:
            raise ImportError(
                "SparseBase methods require native extension sparsebase_reorder_core. "
                "Run: ./scripts/build_native.sh sparsebase"
            ) from exc


def _reorder_from_ori2new(
    csr: scipy.sparse.csr_matrix,
    ori2new: np.ndarray,
    *,
    prep_ms: float,
    reorder_ms: float,
    effective_params: dict[str, Any] | None = None,
) -> ReorderResult:
    with timed_ms() as bucket:
        row_perm = ori2new_to_row_perm(np.asarray(ori2new, dtype=np.int64))
    return ReorderResult(
        row_perm=row_perm,
        reorder_time_ms=reorder_ms,
        prep_time_ms=prep_ms,
        perm_trans_ms=bucket[0],
        effective_params=effective_params or {},
    )


def reorder_sparsebase(
    csr: scipy.sparse.csr_matrix,
    *,
    method: str,
    reorder_fn,
    prep_ms: float,
    effective_params: dict[str, Any] | None = None,
) -> ReorderResult:
    row_ptr, col, cast_ms = csr_to_int64_arrays(csr)
    n, m = csr.shape
    ext = _load_sparsebase()
    with timed_ms() as bucket:
        ori2new = reorder_fn(ext, row_ptr, col, n, m)
    return _reorder_from_ori2new(
        csr,
        ori2new,
        prep_ms=prep_ms + cast_ms,
        reorder_ms=bucket[0],
        effective_params=effective_params,
    )


def partition_to_row_perm(membership: list[int] | np.ndarray) -> np.ndarray:
    parts: dict[int, list[int]] = defaultdict(list)
    for old_i, part_id in enumerate(membership):
        parts[int(part_id)].append(int(old_i))
    row_perm: list[int] = []
    for part_id in sorted(parts.keys()):
        row_perm.extend(parts[part_id])
    return np.asarray(row_perm, dtype=np.int64)


def csr_to_symmetric_csr_arrays(
    csr: scipy.sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Symmetrize CSR (pattern) and return int64 indptr/indices for METIS."""
    with timed_ms() as bucket:
        sym = csr + csr.T
        sym.sum_duplicates()
        # METIS expects no self-loops
        sym.setdiag(0)
        sym.eliminate_zeros()
        sym.sort_indices()
        xadj = np.asarray(sym.indptr, dtype=np.int64)
        adjncy = np.asarray(sym.indices, dtype=np.int64)
    return xadj, adjncy, bucket[0]


def csr_to_symmetric_adjacency(
    csr: scipy.sparse.csr_matrix,
) -> tuple[list[list[int]], float]:
    xadj, adjncy, prep_ms = csr_to_symmetric_csr_arrays(csr)
    n = xadj.shape[0] - 1
    adj = [adjncy[xadj[i] : xadj[i + 1]].astype(int).tolist() for i in range(n)]
    return adj, prep_ms
