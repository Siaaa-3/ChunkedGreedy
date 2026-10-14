"""Rabbit community partition + per-community Jaccard greedy reorder.

1. ACC rabbit ``aggregate`` / ``trace_com`` → communities (root, members)
2. Communities concatenated in ascending root / comm_id order
3. Inside each community: local greedy starting at the community root
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import scipy.sparse

from reorder_bench.loader import csr_to_acc_adjacency
from reorder_bench.methods.global_greedy import (
    has_numba,
    prepare_csr_arrays,
    run_local_greedy,
)
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
                "rabbit-local-greedy requires native extension acc_reorder_core. "
                "Run: ./scripts/build_native.sh acc"
            ) from top_level_exc


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    del params
    nrows = int(csr.shape[0])
    if nrows == 0:
        return ReorderResult(
            row_perm=np.empty(0, dtype=np.int64),
            reorder_time_ms=0.0,
            prep_time_ms=0.0,
            effective_params={"backend": "empty", "n_communities": 0},
        )

    adj, adj_prep_ms = csr_to_acc_adjacency(csr)
    indptr, indices, csc_indptr, csc_indices, csr_prep_ms = prepare_csr_arrays(csr)
    prep_ms = adj_prep_ms + csr_prep_ms

    acc_reorder_core = _load_native()
    backend = "numba" if has_numba() else "numpy"

    breakdown: dict[str, float] = {}
    with timed_ms() as rabbit_bucket:
        communities = acc_reorder_core.rabbit_communities(adj)
    breakdown["rabbit_ms"] = rabbit_bucket[0]

    # Warmup local-greedy compile on first non-trivial community.
    if has_numba():
        for root, members in communities:
            members_arr = np.asarray(members, dtype=np.int64)
            if members_arr.size <= 1:
                continue
            allowed = np.zeros(nrows, dtype=np.bool_)
            allowed[members_arr] = True
            start = int(root)
            if not allowed[start]:
                start = int(members_arr.min())
            run_local_greedy(
                indptr, indices, csc_indptr, csc_indices, allowed, start
            )
            break

    placed = np.zeros(nrows, dtype=np.bool_)
    with timed_ms() as greedy_bucket:
        chunks: list[np.ndarray] = []
        for root, members in communities:
            members_arr = np.asarray(members, dtype=np.int64)
            if members_arr.size == 0:
                continue
            allowed = np.zeros(nrows, dtype=np.bool_)
            allowed[members_arr] = True
            start = int(root)
            if not allowed[start]:
                # Defensive: root should be in members; fall back to min id.
                start = int(members_arr.min())
            if members_arr.size == 1:
                local = members_arr
            else:
                local = run_local_greedy(
                    indptr, indices, csc_indptr, csc_indices, allowed, start
                )
            chunks.append(np.asarray(local, dtype=np.int64))
            placed[local] = True

        leftovers = np.flatnonzero(~placed)
        n_leftover = int(leftovers.size)
        if leftovers.size:
            chunks.append(leftovers.astype(np.int64, copy=False))
        row_perm = (
            np.concatenate(chunks)
            if chunks
            else np.arange(nrows, dtype=np.int64)
        )
    breakdown["local_greedy_ms"] = greedy_bucket[0]

    if row_perm.shape[0] != nrows:
        raise RuntimeError(
            f"rabbit-local-greedy perm length {row_perm.shape[0]} != nrows {nrows}"
        )

    return ReorderResult(
        row_perm=row_perm,
        reorder_time_ms=breakdown["rabbit_ms"] + breakdown["local_greedy_ms"],
        prep_time_ms=prep_ms,
        timing_breakdown=breakdown,
        effective_params={
            "backend": backend,
            "n_communities": int(len(communities)),
            "n_leftover": n_leftover,
        },
    )
