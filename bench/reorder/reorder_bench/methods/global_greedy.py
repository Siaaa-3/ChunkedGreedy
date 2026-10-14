"""Global / local greedy chain reorder by row-pattern Jaccard similarity.

Global: start at a vertex; repeatedly append the remaining vertex with highest
Jaccard similarity to the current tip (ties → smaller id).

Local: same, but candidates restricted to an ``allowed`` mask (e.g. one rabbit
community).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

from reorder_bench.config import CLASSIC_FIXED
from reorder_bench.timing import timed_ms
from reorder_bench.types import ReorderResult

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _local_greedy_numpy(
    indptr: np.ndarray,
    indices: np.ndarray,
    csc_indptr: np.ndarray,
    csc_indices: np.ndarray,
    allowed: np.ndarray,
    start: int,
) -> np.ndarray:
    n = int(indptr.shape[0] - 1)
    n_members = int(np.count_nonzero(allowed))
    if n_members == 0:
        return np.empty(0, dtype=np.int64)
    if not allowed[start]:
        raise ValueError(f"start={start} is not in the allowed set")

    degrees = np.diff(indptr).astype(np.int64, copy=False)
    remaining = allowed.copy()
    order = np.empty(n_members, dtype=np.int64)
    remaining[start] = False
    order[0] = start
    current = start

    marks = np.zeros(n, dtype=np.int64)
    intersects = np.zeros(n, dtype=np.int64)
    cand_buf = np.empty(n, dtype=np.int64)
    stamp = 0

    for pos in range(1, n_members):
        stamp += 1
        n_cand = 0
        deg_c = int(degrees[current])
        best = -1
        best_inter = -1
        best_union = 1

        if deg_c == 0:
            for r in range(n):
                if remaining[r] and degrees[r] == 0:
                    best = r
                    best_inter = 1
                    best_union = 1
                    break
        else:
            row_start = int(indptr[current])
            row_end = int(indptr[current + 1])
            for k in range(row_start, row_end):
                col = int(indices[k])
                c_start = int(csc_indptr[col])
                c_end = int(csc_indptr[col + 1])
                for p in range(c_start, c_end):
                    r = int(csc_indices[p])
                    if not remaining[r]:
                        continue
                    if marks[r] != stamp:
                        marks[r] = stamp
                        intersects[r] = 1
                        cand_buf[n_cand] = r
                        n_cand += 1
                    else:
                        intersects[r] += 1

            for i in range(n_cand):
                r = int(cand_buf[i])
                inter = int(intersects[r])
                union = deg_c + int(degrees[r]) - inter
                if union <= 0:
                    continue
                better = inter * best_union > best_inter * union
                tied = inter * best_union == best_inter * union
                if better or (tied and (best < 0 or r < best)):
                    best_inter = inter
                    best_union = union
                    best = r

        if best < 0:
            for r in range(n):
                if remaining[r]:
                    best = r
                    break

        remaining[best] = False
        order[pos] = best
        current = best

    return order


if _HAS_NUMBA:

    @njit(cache=True)
    def _local_greedy_numba(
        indptr: np.ndarray,
        indices: np.ndarray,
        csc_indptr: np.ndarray,
        csc_indices: np.ndarray,
        allowed: np.ndarray,
        start: int,
    ) -> np.ndarray:
        n = indptr.shape[0] - 1
        n_members = 0
        for i in range(n):
            if allowed[i]:
                n_members += 1
        order = np.empty(n_members, dtype=np.int64)
        if n_members == 0:
            return order

        degrees = np.empty(n, dtype=np.int64)
        for i in range(n):
            degrees[i] = indptr[i + 1] - indptr[i]

        remaining = allowed.copy()
        remaining[start] = False
        order[0] = start
        current = start

        marks = np.zeros(n, dtype=np.int64)
        intersects = np.zeros(n, dtype=np.int64)
        cand_buf = np.empty(n, dtype=np.int64)
        stamp = 0

        for pos in range(1, n_members):
            stamp += 1
            n_cand = 0
            deg_c = degrees[current]
            best = -1
            best_inter = -1
            best_union = 1

            if deg_c == 0:
                for r in range(n):
                    if remaining[r] and degrees[r] == 0:
                        best = r
                        best_inter = 1
                        best_union = 1
                        break
            else:
                row_start = indptr[current]
                row_end = indptr[current + 1]
                for k in range(row_start, row_end):
                    col = indices[k]
                    c_start = csc_indptr[col]
                    c_end = csc_indptr[col + 1]
                    for p in range(c_start, c_end):
                        r = csc_indices[p]
                        if not remaining[r]:
                            continue
                        if marks[r] != stamp:
                            marks[r] = stamp
                            intersects[r] = 1
                            cand_buf[n_cand] = r
                            n_cand += 1
                        else:
                            intersects[r] += 1

                for i in range(n_cand):
                    r = cand_buf[i]
                    inter = intersects[r]
                    union = deg_c + degrees[r] - inter
                    if union <= 0:
                        continue
                    better = inter * best_union > best_inter * union
                    tied = inter * best_union == best_inter * union
                    if better or (tied and (best < 0 or r < best)):
                        best_inter = inter
                        best_union = union
                        best = r

            if best < 0:
                for r in range(n):
                    if remaining[r]:
                        best = r
                        break

            remaining[best] = False
            order[pos] = best
            current = best

        return order


def run_local_greedy(
    indptr: np.ndarray,
    indices: np.ndarray,
    csc_indptr: np.ndarray,
    csc_indices: np.ndarray,
    allowed: np.ndarray,
    start: int,
) -> np.ndarray:
    """Greedy chain over vertices with ``allowed[v]``; start must be allowed."""
    if _HAS_NUMBA:
        return _local_greedy_numba(
            indptr, indices, csc_indptr, csc_indices, allowed, start
        )
    return _local_greedy_numpy(
        indptr, indices, csc_indptr, csc_indices, allowed, start
    )


def prepare_csr_arrays(
    csr: scipy.sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """CSR + CSC int64 arrays for greedy; returns (*, prep_ms)."""
    with timed_ms() as prep_bucket:
        mat = csr.tocsr()
        if not mat.has_sorted_indices:
            mat.sort_indices()
        indptr = np.asarray(mat.indptr, dtype=np.int64)
        indices = np.asarray(mat.indices, dtype=np.int64)
        csc = mat.tocsc()
        if not csc.has_sorted_indices:
            csc.sort_indices()
        csc_indptr = np.asarray(csc.indptr, dtype=np.int64)
        csc_indices = np.asarray(csc.indices, dtype=np.int64)
    return indptr, indices, csc_indptr, csc_indices, prep_bucket[0]


def has_numba() -> bool:
    return _HAS_NUMBA


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    del params
    start = int(CLASSIC_FIXED["global-greedy"]["start_node"])
    nrows = int(csr.shape[0])
    if nrows == 0:
        return ReorderResult(
            row_perm=np.empty(0, dtype=np.int64),
            reorder_time_ms=0.0,
            prep_time_ms=0.0,
            effective_params={"start_node": start, "backend": "empty"},
        )
    if start < 0 or start >= nrows:
        raise ValueError(f"global-greedy start_node={start} out of range [0, {nrows})")

    indptr, indices, csc_indptr, csc_indices, prep_ms = prepare_csr_arrays(csr)
    allowed = np.ones(nrows, dtype=np.bool_)

    backend = "numba" if _HAS_NUMBA else "numpy"
    if _HAS_NUMBA:
        run_local_greedy(indptr, indices, csc_indptr, csc_indices, allowed, start)

    with timed_ms() as core_bucket:
        row_perm = run_local_greedy(
            indptr, indices, csc_indptr, csc_indices, allowed, start
        )
    reorder_ms = core_bucket[0]

    return ReorderResult(
        row_perm=np.asarray(row_perm, dtype=np.int64),
        reorder_time_ms=reorder_ms,
        prep_time_ms=prep_ms,
        effective_params={"start_node": start, "backend": backend},
    )
