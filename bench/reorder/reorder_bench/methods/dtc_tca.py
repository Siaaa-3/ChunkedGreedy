from __future__ import annotations

import queue as Q
from typing import Any, Iterator

import numpy as np
import scipy.sparse
from datasketch import MinHash, MinHashLSH, WeightedMinHash, WeightedMinHashGenerator

from reorder_bench.config import (
    DTC_FIXED,
    DTC_GPU_ROW_THRESHOLD,
)
from reorder_bench.loader import prep_dtc_csr, row_perm_from_order
from reorder_bench.timing import sync_gpu_if_needed, timed_ms
from reorder_bench.types import ReorderResult


class _Pair:
    def __init__(self, p1: int, p2: int, similarity: float) -> None:
        self.p1 = p1
        self.p2 = p2
        self.simi = similarity

    def __lt__(self, other: "_Pair") -> bool:
        return self.simi > other.simi


def _resolve_minhash_backend(nrows: int, params: dict[str, Any]) -> str:
    mode = DTC_FIXED.get("use_gpu_minhash", "auto")
    if mode == "cpu":
        return "cpu"
    if mode == "gpu":
        return "gpu" if _gpu_minhash_available() else "cpu"
    if nrows < DTC_GPU_ROW_THRESHOLD and _gpu_minhash_available():
        return "gpu"
    return "cpu"


def _gpu_minhash_available() -> bool:
    try:
        import libMHCUDA  # noqa: F401

        return True
    except ImportError:
        return False


def _cugraph_available() -> bool:
    try:
        import cugraph  # noqa: F401
        import cudf  # noqa: F401

        return True
    except ImportError:
        return False


def _root(cluster_id: list[int], i: int) -> int:
    while i != cluster_id[i]:
        cluster_id[i] = cluster_id[cluster_id[i]]
        i = cluster_id[i]
    return i


def _jd(l1: list, l2: list) -> float:
    if not l1 or not l2:
        return 0.0
    s1, s2 = set(l1), set(l2)
    return len(s1 & s2) / len(s1 | s2)


def _makenum(a: int, b: int, stride: int) -> int:
    if a > b:
        a, b = b, a
    return a * stride + b


def _cluster_from_queue(
    que: Q.PriorityQueue,
    sset: set[int],
    lists: list[list],
    cluster_id: list[int],
    cluster_sz: list[int],
    deleted: list[int],
    thres: int,
    stride: int,
) -> None:
    num_cluster = len(cluster_id)
    while not que.empty() and num_cluster > 0:
        item = que.get()
        p1, p2 = item.p1, item.p2
        sset.discard(_makenum(p1, p2, stride))
        if p1 == cluster_id[p1] and p2 == cluster_id[p2]:
            if deleted[p1] or deleted[p2]:
                continue
            num_cluster -= 1
            if cluster_sz[p1] < cluster_sz[p2]:
                cluster_id[p1] = p2
                cluster_sz[p2] += cluster_sz[p1]
                if cluster_sz[p2] >= thres:
                    deleted[p2] = 1
                    num_cluster -= 1
            else:
                cluster_id[p2] = p1
                cluster_sz[p1] += cluster_sz[p2]
                if cluster_sz[p1] >= thres:
                    deleted[p1] = 1
                    num_cluster -= 1
        else:
            p1 = _root(cluster_id, p1)
            p2 = _root(cluster_id, p2)
            if deleted[p1] or deleted[p2]:
                continue
            key = _makenum(p1, p2, stride)
            if p1 != p2 and key not in sset:
                que.put(_Pair(p1, p2, _jd(lists[p1], lists[p2])))
                sset.add(key)


def _collect_clusters(cluster_id: list[int], n: int) -> dict[int, list[int]]:
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        ro = _root(cluster_id, i)
        clusters.setdefault(ro, []).append(i)
    return clusters


def _flatten_perm(clusters: dict[int, list[int]], clusters_c: dict[int, list[int]], key: list[int]) -> list[int]:
    order: list[int] = []
    for j in clusters_c:
        for k in clusters_c[j]:
            for item in clusters[key[k]]:
                order.append(item)
    return order


def _jaccard_rows(graph: Any, pairs: Any) -> Iterator[tuple[Any, Any, Any]]:
    import cugraph

    result = cugraph.jaccard(graph, pairs).to_arrow().to_pydict()
    required = ("first", "second", "jaccard_coeff")
    missing = [name for name in required if name not in result]
    if missing:
        raise KeyError(f"cugraph.jaccard missing columns: {', '.join(missing)}")
    return zip(result["first"], result["second"], result["jaccard_coeff"])


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    prep_csr, prep_ms = prep_dtc_csr(csr)
    nrows = prep_csr.shape[0]
    per = int(params["per"])
    thres = int(params["thres"])
    lsh_thres = float(params["lsh_thres"])
    cluster_thres = float(params["cluster_thres"])
    c_thres = int(params["c_thres"])
    seed = int(DTC_FIXED["seed"])

    ptr = prep_csr.indptr
    idx = prep_csr.indices
    backend = _resolve_minhash_backend(nrows, params)
    used_gpu = backend == "gpu"
    breakdown: dict[str, float] = {}

    row_ind = prep_csr.tocoo().row.astype(np.int64)
    col_ind = prep_csr.tocoo().col.astype(np.int64)

    with timed_ms() as bucket:
        lsh = MinHashLSH(threshold=lsh_thres, num_perm=per)
        allver = []
        lists: list[list[int]] = [[] for _ in range(nrows)]
        if backend == "gpu":
            import libMHCUDA

            bgen = WeightedMinHashGenerator(nrows)
            gen = libMHCUDA.minhash_cuda_init(nrows, per, seed=1, verbosity=0)
            try:
                libMHCUDA.minhash_cuda_assign_vars(gen, bgen.rs, bgen.ln_cs, bgen.betas)
                hashes = libMHCUDA.minhash_cuda_calc(gen, prep_csr)
            finally:
                libMHCUDA.minhash_cuda_fini(gen)
            for i in range(nrows):
                m = WeightedMinHash(seed, hashes[i])
                lsh.insert(str(i), m)
                allver.append(m)
                for j in range(int(ptr[i]), int(ptr[i + 1])):
                    lists[i].append(int(idx[j]))
        else:
            for i in range(nrows):
                m = MinHash(num_perm=per)
                for j in range(int(ptr[i]), int(ptr[i + 1])):
                    col = int(idx[j])
                    m.update(str(col).encode("utf-8"))
                    lists[i].append(col)
                lsh.insert(str(i), m)
                allver.append(m)
    breakdown["minhash_ms"] = bucket[0]

    if not _cugraph_available():
        raise ImportError(
            "dtc-tca requires cugraph/cudf (and cupy): install RAPIDS stack or use CPU-only path "
            "after implementing CPU jaccard fallback"
        )

    import cudf
    import cugraph

    G = cugraph.Graph()
    df = cudf.DataFrame({"src": cudf.Series(row_ind), "dst": cudf.Series(col_ind)})
    G.from_cudf_edgelist(df, source="src", destination="dst")

    with timed_ms() as bucket:
        que: Q.PriorityQueue = Q.PriorityQueue()
        sset: set[int] = set()
        first = np.empty((0,), dtype=np.int64)
        second = np.empty((0,), dtype=np.int64)
        for i in range(nrows):
            if ptr[i] == ptr[i + 1]:
                continue
            res = lsh.query(allver[i])
            first = np.concatenate((first, np.full(len(res), i, dtype=np.int64)))
            second = np.concatenate((second, np.array(res, dtype=np.int64)))
            if i % 1000 == 0 or i == nrows - 1:
                df_pairs = cudf.DataFrame({"first": first, "second": second})
                for source_raw, item_raw, score_raw in _jaccard_rows(G, df_pairs):
                    source = int(source_raw)
                    item = int(item_raw)
                    if item == source:
                        continue
                    key = _makenum(source, item, nrows)
                    if key in sset:
                        continue
                    que.put(_Pair(source, item, float(score_raw)))
                    sset.add(key)
                first = np.empty((0,), dtype=np.int64)
                second = np.empty((0,), dtype=np.int64)
    breakdown["lsh_query_l1_ms"] = bucket[0]

    cluster_id = list(range(nrows))
    cluster_sz = [1] * nrows
    deleted = [0] * nrows
    with timed_ms() as bucket:
        _cluster_from_queue(que, sset, lists, cluster_id, cluster_sz, deleted, thres, nrows)
    breakdown["cluster_l1_ms"] = bucket[0]

    clusters = _collect_clusters(cluster_id, nrows)
    cluster_num = len(clusters)
    key = list(clusters.keys())

    with timed_ms() as bucket:
        lsh_c = MinHashLSH(threshold=cluster_thres, num_perm=per)
        allver_c = []
        lists_c: list[list[int]] = [[] for _ in range(cluster_num)]
        cnt = 0
        for i in clusters:
            m = MinHash(num_perm=per)
            list_cluster_i: list[int] = []
            for node in clusters[i]:
                list_cluster_i.extend(lists[node])
            list_cluster_i = list(set(list_cluster_i))
            lists_c[cnt] = list_cluster_i
            for col in list_cluster_i:
                m.update(str(col).encode("utf-8"))
            lsh_c.insert(str(cnt), m)
            allver_c.append(m)
            cnt += 1
        que_c: Q.PriorityQueue = Q.PriorityQueue()
        sset_c: set[int] = set()
        for i in range(cluster_num):
            if not lists_c[i]:
                continue
            res = lsh_c.query(allver_c[i])
            for item in res:
                item_i = int(item)
                if item_i == i:
                    continue
                key_c = _makenum(i, item_i, cluster_num)
                if key_c in sset_c:
                    continue
                if not lists_c[item_i]:
                    continue
                que_c.put(_Pair(i, item_i, _jd(lists_c[i], lists_c[item_i])))
                sset_c.add(key_c)
    breakdown["lsh_query_l2_ms"] = bucket[0]

    cluster_id_c = list(range(cluster_num))
    cluster_sz_c = [1] * cluster_num
    deleted_c = [0] * cluster_num
    with timed_ms() as bucket:
        _cluster_from_queue(
            que_c, sset_c, lists_c, cluster_id_c, cluster_sz_c, deleted_c, c_thres, cluster_num
        )
    breakdown["cluster_l2_ms"] = bucket[0]

    clusters_c = _collect_clusters(cluster_id_c, cluster_num)

    sync_gpu_if_needed(used_gpu)
    reorder_time_ms = sum(breakdown.values())

    with timed_ms() as bucket:
        order = _flatten_perm(clusters, clusters_c, key)
        row_perm = row_perm_from_order(order, nrows)
    perm_trans_ms = bucket[0]

    return ReorderResult(
        row_perm=row_perm,
        reorder_time_ms=reorder_time_ms,
        prep_time_ms=prep_ms,
        perm_trans_ms=perm_trans_ms,
        timing_breakdown=breakdown,
        env_extra={"dtc_minhash_backend": backend},
    )
