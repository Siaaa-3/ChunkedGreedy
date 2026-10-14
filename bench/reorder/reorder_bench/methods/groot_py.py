from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import minimum_spanning_tree

from reorder_bench.config import GROOT_PY_FIXED, resolve_runtime_env
from reorder_bench.loader import prep_groot_csr, row_perm_from_order
from reorder_bench.timing import timed_ms
from reorder_bench.types import ReorderResult

_SPARSE_METRICS_DIR = (
    Path(__file__).resolve().parents[2]
    / "Reordering_methods"
    / "Groot-EuroSys25"
)


def _import_groot_deps():
    try:
        from pynndescent import NNDescent
    except ImportError as exc:
        raise ImportError(
            "groot-py requires pynndescent and numba: "
            "pip install pynndescent numba"
        ) from exc
    return NNDescent


def _import_sparse_metrics():
    if str(_SPARSE_METRICS_DIR) not in sys.path:
        sys.path.insert(0, str(_SPARSE_METRICS_DIR))
    try:
        from sparse import sparse_hamming, sparse_jaccard
    except ImportError as exc:
        raise ImportError(
            f"groot-py requires sparse.py metrics under {_SPARSE_METRICS_DIR}"
        ) from exc
    return sparse_jaccard, sparse_hamming


def _resolve_metric(metric_name: str, n_features: int) -> Callable:
    sparse_jaccard, sparse_hamming = _import_sparse_metrics()
    name = metric_name.lower()
    if name == "jaccard":
        return sparse_jaccard
    if name == "hamming":
        import numba

        @numba.njit()
        def hamming_impl(ind1, data1, ind2, data2):
            return sparse_hamming(ind1, data1, ind2, data2, n_features)

        return hamming_impl
    raise ValueError(
        f"groot-py fixed_defaults.similarity_metric must be "
        f"'jaccard' or 'hamming', got {metric_name!r}"
    )


def _select_start_node(
    strategy: str,
    graph: list[list[tuple[int, float]]],
    nrows: int,
    seed: int,
) -> int:
    name = strategy.lower()
    if name == "max_degree":
        degrees = [len(nbrs) for nbrs in graph]
        return int(np.argmax(degrees))
    if name == "random":
        rng = random.Random(seed)
        return int(rng.randint(0, nrows - 1))
    if name == "first":
        return 0
    raise ValueError(
        f"groot-py fixed_defaults.start_node must be "
        f"'max_degree', 'random', or 'first', got {strategy!r}"
    )


def _traverse(
    traversal: str,
    graph: list[list[tuple[int, float]]],
    start_node: int,
    nrows: int,
) -> list[int]:
    visited: set[int] = set()
    order: list[int] = []
    name = traversal.lower()

    def _sorted_nbrs(node: int) -> list[tuple[int, float]]:
        return sorted(graph[node], key=lambda x: x[1], reverse=True)

    if name == "dfs":

        def dfs_from(seed: int) -> None:
            stack = [seed]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                order.append(node)
                for neighbor_id, _sim in reversed(_sorted_nbrs(node)):
                    if neighbor_id not in visited:
                        stack.append(neighbor_id)

        dfs_from(start_node)
        for node in range(nrows):
            if node not in visited:
                dfs_from(node)
        return order

    if name == "bfs":
        queue: deque[int] = deque([start_node])
        visited.add(start_node)
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor_id, _sim in _sorted_nbrs(node):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)
        for node in range(nrows):
            if node not in visited:
                queue.append(node)
                visited.add(node)
                while queue:
                    cur = queue.popleft()
                    order.append(cur)
                    for neighbor_id, _sim in _sorted_nbrs(cur):
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            queue.append(neighbor_id)
        return order

    raise ValueError(
        f"groot-py fixed_defaults.traversal must be "
        f"'dfs' or 'bfs', got {traversal!r}"
    )


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    NNDescent = _import_groot_deps()
    runtime = resolve_runtime_env()
    prep_csr, prep_ms = prep_groot_csr(csr)
    nrows = prep_csr.shape[0]
    knn = int(params["knn"])
    n_jobs = runtime["n_jobs"]

    seed = int(GROOT_PY_FIXED["seed"])
    use_mst = bool(GROOT_PY_FIXED["use_mst"])
    similarity_metric = str(GROOT_PY_FIXED["similarity_metric"])
    traversal = str(GROOT_PY_FIXED["traversal"])
    start_strategy = str(GROOT_PY_FIXED["start_node"])

    metric_func = _resolve_metric(similarity_metric, prep_csr.shape[1])
    random.seed(seed)
    np.random.seed(seed)

    breakdown: dict[str, float] = {}
    knn_graph: list[list[tuple[int, float]]] = [[] for _ in range(nrows)]

    with timed_ms() as bucket:
        index = NNDescent(
            prep_csr,
            metric=metric_func,
            n_neighbors=knn + 1,
            random_state=seed,
            verbose=False,
            n_jobs=n_jobs,
        )
        neighbors, distances = index.neighbor_graph
        for i in range(nrows):
            for neighbor_id, dist in zip(neighbors[i], distances[i]):
                nid = int(neighbor_id)
                if nid >= 0 and nid != i:
                    sim = 1.0 - float(dist) if dist < 1.0 else 0.0
                    knn_graph[i].append((nid, sim))
    breakdown["knn_ms"] = bucket[0]

    if use_mst:
        with timed_ms() as bucket:
            from scipy.sparse import lil_matrix

            adj = lil_matrix((nrows, nrows), dtype=np.float32)
            for i, nbrs in enumerate(knn_graph):
                for neighbor_id, sim in nbrs:
                    dist = 1.0 - sim
                    j = int(neighbor_id)
                    if adj[i, j] == 0 or dist < adj[i, j]:
                        adj[i, j] = dist
                    if adj[j, i] == 0 or dist < adj[j, i]:
                        adj[j, i] = dist
            mst = minimum_spanning_tree(adj.tocsr())
            mst_coo = mst.tocoo()
            graph: list[list[tuple[int, float]]] = [[] for _ in range(nrows)]
            for i, j, dist in zip(mst_coo.row, mst_coo.col, mst_coo.data):
                sim = 1.0 - float(dist)
                ii, jj = int(i), int(j)
                graph[ii].append((jj, sim))
                graph[jj].append((ii, sim))
            graph_to_traverse = graph
        breakdown["mst_ms"] = bucket[0]
    else:
        breakdown["mst_ms"] = 0.0
        undirected: list[list[tuple[int, float]]] = [[] for _ in range(nrows)]
        for i, nbrs in enumerate(knn_graph):
            for neighbor_id, sim in nbrs:
                undirected[i].append((int(neighbor_id), sim))
                reverse = any(n == i for n, _ in knn_graph[int(neighbor_id)])
                if not reverse:
                    undirected[int(neighbor_id)].append((i, sim))
        graph_to_traverse = undirected

    start_node = _select_start_node(
        start_strategy, graph_to_traverse, nrows, seed
    )

    with timed_ms() as bucket:
        order = _traverse(traversal, graph_to_traverse, start_node, nrows)
    breakdown["traversal_ms"] = bucket[0]

    if len(order) != nrows or len(set(order)) != nrows:
        raise RuntimeError(
            f"groot-py traversal produced invalid order: "
            f"len={len(order)} unique={len(set(order))} nrows={nrows}"
        )

    reorder_time_ms = sum(breakdown.values())

    with timed_ms() as bucket:
        row_perm = row_perm_from_order(order, nrows)
    perm_trans_ms = bucket[0]

    return ReorderResult(
        row_perm=row_perm,
        reorder_time_ms=reorder_time_ms,
        prep_time_ms=prep_ms,
        perm_trans_ms=perm_trans_ms,
        timing_breakdown=breakdown,
    )
