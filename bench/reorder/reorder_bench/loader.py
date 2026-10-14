from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

import spmatkit

from reorder_bench.timing import timed_ms


def resolve_matrix_names(
    *,
    matrices: list[str] | None = None,
    dataset: str | None = None,
) -> list[str]:
    """Resolve matrix base names from an explicit list or a spmatkit dataset."""
    if matrices and dataset:
        raise ValueError("Specify either --matrices or --dataset, not both")
    if matrices:
        return list(matrices)
    if dataset:
        names = spmatkit.get_dataset_names(dataset)
        if not names:
            raise FileNotFoundError(f"Dataset '{dataset}' is empty or not found")
        return names
    raise ValueError("Either --matrices or --dataset must be provided")


def load_pattern_matrix(name: str) -> scipy.sparse.csr_matrix:
    with timed_ms() as bucket:
        csr = spmatkit.load_matrix(name, load_mode="pattern")
    assert scipy.sparse.isspmatrix_csr(csr)
    assert csr.shape[0] == csr.shape[1]
    return csr, bucket[0]


def prep_groot_csr(csr: scipy.sparse.csr_matrix) -> tuple[scipy.sparse.csr_matrix, float]:
    with timed_ms() as bucket:
        out = csr.copy()
        out.sort_indices()
        if out.data.dtype != np.float32:
            out.data = np.ones(out.nnz, dtype=np.float32)
    return out, bucket[0]


def prep_dtc_csr(csr: scipy.sparse.csr_matrix) -> tuple[scipy.sparse.csr_matrix, float]:
    with timed_ms() as bucket:
        out = csr.copy()
        out.sort_indices()
        out.data = np.ones(out.nnz, dtype=np.float32)
    return out, bucket[0]


def prep_clusterwise_csr(csr: scipy.sparse.csr_matrix) -> tuple[scipy.sparse.csr_matrix, float]:
    with timed_ms() as bucket:
        out = csr.copy()
        out.sort_indices()
    return out, bucket[0]


def prep_sporder_gpu(
    csr: scipy.sparse.csr_matrix,
    *,
    detect_symmetric: bool = True,
) -> tuple[Any, Any, bool, float]:
    """Sort CSR, optionally detect pattern symmetry, upload int32 ptr/ind to CuPy.

    When ``detect_symmetric`` is False, skip ``(mat != mat.T).nnz`` and return
    ``is_symmetric=False`` (forces sporder to run the csr2csc path).
    """
    import cupy

    int32_max = np.iinfo(np.int32).max
    nrows = int(csr.shape[0])
    nnz = int(csr.nnz)
    if nrows > int32_max or nnz > int32_max:
        raise ValueError(
            f"sporder requires n and nnz within int32; got n={nrows}, nnz={nnz}"
        )

    with timed_ms() as bucket:
        out = csr.copy()
        out.sort_indices()
        if detect_symmetric:
            # Same check as sporder scripts: pattern (+values) equals transpose.
            is_symmetric = bool((out != out.T).nnz == 0)
        else:
            is_symmetric = False
        ptr = cupy.asarray(out.indptr, dtype=cupy.int32)
        ind = cupy.asarray(out.indices, dtype=cupy.int32)
    return ptr, ind, is_symmetric, bucket[0]


def csr_to_acc_adjacency(
    csr: scipy.sparse.csr_matrix,
) -> tuple[list[list[tuple[int, float]]], float]:
    """Python equivalent of Acc make_adj_list (symmetrize, dedup, no self-loops)."""
    with timed_ms() as bucket:
        n = csr.shape[0]
        coo = csr.tocoo()
        edges: list[tuple[int, int, float]] = []
        for r, c, v in zip(coo.row, coo.col, coo.data):
            ri, ci = int(r), int(c)
            if ri == ci:
                continue
            w = float(v)
            edges.append((ri, ci, w))
            edges.append((ci, ri, w))
        edges.sort()
        adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        i = 0
        while i < len(edges):
            s, t, w = edges[i]
            while i < len(edges) and edges[i][0] == s and edges[i][1] == t:
                w += float(edges[i][2])
                i += 1
            if w > 0:
                adj[s].append((t, w))
    return adj, bucket[0]


def row_perm_from_order(order: list[int] | np.ndarray, nrows: int) -> np.ndarray:
    perm = np.asarray(order, dtype=np.int64)
    if perm.shape[0] != nrows:
        raise ValueError(f"perm length {perm.shape[0]} != nrows {nrows}")
    return perm


def csr_to_int64_arrays(
    csr: scipy.sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray, float]:
    with timed_ms() as bucket:
        row_ptr = np.asarray(csr.indptr, dtype=np.int64)
        col = np.asarray(csr.indices, dtype=np.int64)
    return row_ptr, col, bucket[0]


def ori2new_to_row_perm(ori2new: np.ndarray) -> np.ndarray:
    ori2new = np.asarray(ori2new, dtype=np.int64)
    n = ori2new.shape[0]
    row_perm = np.empty(n, dtype=np.int64)
    for old_i, new_i in enumerate(ori2new):
        row_perm[new_i] = old_i
    return row_perm
