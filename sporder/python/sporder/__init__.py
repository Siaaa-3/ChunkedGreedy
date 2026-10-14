"""GPU sparse matrix reordering via hierarchical matching."""

from ._sporder import (
    Context,
    bench_cusparse_spmm as _bench_cusparse_spmm,
    chunk_greedy as _chunk_greedy,
    chunk_similarity_greedy as _chunk_similarity_greedy,
    count_pillars as _count_pillars,
    hierarchical_matching as _hierarchical_matching,
    reorder as _reorder,
    similarity_graph as _similarity_graph,
    similarity_graph_exact as _similarity_graph_exact,
)

__all__ = [
    "bench_cusparse_spmm",
    "Context",
    "chunk_greedy",
    "chunk_similarity_greedy",
    "count_pillars",
    "hierarchical_matching",
    "reorder",
    "similarity_graph",
    "similarity_graph_exact",
]


def _asarray_like(array, reference):
    """Convert a DLPack-capable array to the framework of `reference`."""
    root = type(reference).__module__.partition(".")[0]
    if root == "torch":
        import torch

        return torch.from_dlpack(array)
    if root == "cupy":
        import cupy

        return cupy.from_dlpack(array)
    return array


def hierarchical_matching(
    ptr,
    ind,
    *,
    is_symmetric=False,
    num_levels=1,
    k=0,
    oversampling=0.0,
    max_rounds=0,
    with_timing=False,
    with_stats=False,
    ctx=None,
):
    """Compute a fill-reducing ordering of a CSR matrix on the GPU.

    Arguments:
        ptr: int32 CUDA array of length n+1 (CSR row pointers)
        ind: int32 CUDA array of length nnz (CSR column indices)
        is_symmetric: whether the sparsity pattern is symmetric
        num_levels: number of hierarchy levels
        k: similarity top-k (0 = auto)
        oversampling: oversampling factor (0 = auto)
        max_rounds: matching round limit (0 = run to convergence)
        with_timing: also return a dict of timing measurements
        with_stats: also return a dict of algorithm diagnostics
            (k_used, m_used, avg_degree, avg_edge_weight,
            zero_degree_rows, and per-level level_size, matched_pairs,
            rounds); excluded
            from stage timings, only wall_ms includes their cost.
            Useful derived quantities: the matched fraction at level l
            is 2 * matched_pairs[l] / level_size[l]; the isolated
            fraction is zero_degree_rows / n
        ctx: Context instance; None uses a shared default context
            created on first use and bound to the then-current device

    Accepts any DLPack-compatible GPU array (PyTorch, CuPy, ...).

    Returns:
        new_order: int32 CUDA array of length n in new2old form:
            the output's row p contains the input's row new_order[p],
            in the same framework as `ptr` (PyTorch or CuPy;
            otherwise a DLPack-capable array)
        timing: dict of timing measurements, appended to the result
            only when with_timing is True
        stats: dict of algorithm diagnostics, appended to the result
            only when with_stats is True

    Like numpy's return_* flags, each requested extra appends one more
    return value: `new_order = ...`, `new_order, timing = ...`,
    `new_order, stats = ...`, or `new_order, timing, stats = ...`.
    """
    new_order, timing, stats = _hierarchical_matching(
        ptr,
        ind,
        is_symmetric=is_symmetric,
        num_levels=num_levels,
        k=k,
        oversampling=oversampling,
        max_rounds=max_rounds,
        with_timing=with_timing,
        with_stats=with_stats,
        ctx=ctx,
    )
    result = (_asarray_like(new_order, ptr),)
    if with_timing:
        result += (timing,)
    if with_stats:
        result += (stats,)
    return result if len(result) > 1 else result[0]


def reorder(ptr, ind, new_order, *, val=None, symmetric=False, ctx=None):
    """Apply a row (or symmetric row+column) permutation to a CSR pattern.

    The output's row p contains the input's row new_order[p]. With
    symmetric=True, column indices are remapped by the same permutation
    (new[p[i], p[j]] = old[i, j]; the pattern must be square) and each
    output row is re-sorted.  Pass val to reorder a value array
    alongside the column indices.

    Arguments:
        ptr: int32 CUDA array of length n+1 (CSR row pointers)
        ind: int32 CUDA array of length nnz (CSR column indices)
        new_order: int32 CUDA array of length n, a permutation of 0..n-1
        val: optional float32 CUDA array of length nnz (CSR values)
        symmetric: also permute columns (default: rows only)
        ctx: Context instance; None uses the shared default context

    Accepts any DLPack-compatible GPU array (PyTorch, CuPy, ...).

    Returns:
        (ptr_new, ind_new) or (ptr_new, ind_new, val_new): the reordered
            CSR pattern (and values, when val is provided), in the same
            framework as `ptr`; column indices are sorted within each row
    """
    if val is not None:
        ptr_new, ind_new, val_new = _reorder(
            ptr, ind, val=val, new_order=new_order, symmetric=symmetric, ctx=ctx
        )
        return (
            _asarray_like(ptr_new, ptr),
            _asarray_like(ind_new, ptr),
            _asarray_like(val_new, ptr),
        )
    ptr_new, ind_new = _reorder(
        ptr, ind, val=None, new_order=new_order, symmetric=symmetric, ctx=ctx
    )
    return _asarray_like(ptr_new, ptr), _asarray_like(ind_new, ptr)


def count_pillars(ptr, ind, height, *, ctx=None):
    """Count nonzero columns over consecutive row windows.

    Rows are grouped into ceil(n / height) consecutive windows of
    `height` rows (the last window may be shorter). A pillar is a
    column holding at least one nonzero inside a window; the total over
    all windows is the column-tile count of a blocked SpMM with tile
    width 1 -- the quantity row reordering tries to reduce.

    Arguments:
        ptr: int32 CUDA array of length n+1 (CSR row pointers)
        ind: int32 CUDA array of length nnz (CSR column indices)
        height: row window height, >= 1
        ctx: Context instance; None uses the shared default context

    Accepts any DLPack-compatible GPU array (PyTorch, CuPy, ...).

    Returns:
        int: total number of pillars over all windows
    """
    return _count_pillars(ptr, ind, height, ctx=ctx)


def bench_cusparse_spmm(ptr, ind, cols, *, warmup=3, repeat=10, seed=0, ctx=None):
    """Benchmark cuSPARSE SpMM throughput on a square CSR pattern.

    Generates reproducible pseudorandom float values for A and ``cols``-column
    dense B (column-major) from ``seed``, then runs cusparseSpMM
    ``warmup`` + ``repeat`` times.  Returns the average wall-clock
    time per repeat in milliseconds.

    Arguments:
        ptr:    int32 CUDA array of length n+1 (CSR row pointers)
        ind:    int32 CUDA array of length nnz (CSR column indices)
        cols:   number of columns in dense B, >= 1
        warmup: number of dry-run iterations (default 3)
        repeat: number of timed iterations    (default 10)
        seed:   value-generation seed         (default 0)
        ctx:    Context instance; None uses the shared default context

    Returns:
        float: average cusparseSpMM wall-time in milliseconds
    """
    return _bench_cusparse_spmm(
        ptr, ind, cols, warmup=warmup, repeat=repeat, seed=seed, ctx=ctx
    )


def similarity_graph(
    ptr,
    ind,
    *,
    is_symmetric=False,
    k=0,
    oversampling=0.0,
    with_timing=False,
    with_stats=False,
    ctx=None,
):
    """Build the weighted top-k Dice-similarity graph.

    For each row r, finds up to k neighbors s with the largest Dice
    similarity of their column sets, 2|r&s|/(|r|+|s|), approximated
    with a space-saving sketch.

    Arguments:
        ptr: int32 CUDA array of length n+1 (CSR row pointers)
        ind: int32 CUDA array of length nnz (CSR column indices)
        is_symmetric: whether the sparsity pattern is symmetric
        k: number of neighbors per row (0 = auto)
        oversampling: space-saving sketch oversampling (0 = auto)
        with_timing: also return a dict of timing measurements
        with_stats: also return a dict of graph diagnostics
            (k_used, m_used, avg_degree, avg_edge_weight,
            zero_degree_rows)
        ctx: Context instance; None uses the shared default context

    Returns:
        (indices, weights): n x k arrays in the same framework as
            `ptr`.  Entries are sorted by descending weight per row;
            unused entries are marked with index = -1 and weight = 0.
        timing: dict of timing measurements, appended to the result
            only when with_timing is True
        stats: dict of graph diagnostics, appended to the result
            only when with_stats is True
    """
    idx, wgt, timing, stats = _similarity_graph(
        ptr,
        ind,
        is_symmetric=is_symmetric,
        k=k,
        oversampling=oversampling,
        with_timing=with_timing,
        with_stats=with_stats,
        ctx=ctx,
    )
    n = ptr.shape[0] - 1
    if k <= 0:
        nnz = ind.shape[0] if hasattr(ind, "shape") else len(ind)
        k = min(32, max(16, 2 * nnz // n))
    idx = _asarray_like(idx, ptr).reshape(n, k)
    wgt = _asarray_like(wgt, ptr).reshape(n, k)
    result = (idx, wgt)
    if with_timing:
        result += (timing,)
    if with_stats:
        result += (stats,)
    return result if len(result) > 2 else result


def similarity_graph_exact(
    ptr, ind, *, is_symmetric=False, k=0, with_timing=False, with_stats=False, ctx=None
):
    """Exact top-k Dice-similarity graph via cuSPARSE SpGEMM.

    Computes the exact pairwise Dice similarity for every row pair
    that shares a column.  Intended as a correctness baseline and for
    small-to-medium validation; the exact computation is dramatically
    more expensive than the sketch.

    Arguments:
        ptr: int32 CUDA array of length n+1 (CSR row pointers)
        ind: int32 CUDA array of length nnz (CSR column indices)
        is_symmetric: whether the sparsity pattern is symmetric
        k: number of neighbors per row (0 = auto)
        with_timing: also return a dict of timing measurements
        with_stats: also return a dict of graph diagnostics
        ctx: Context instance; None uses the shared default context

    Returns:
        (indices, weights): n x k arrays; entries are sorted by
            descending weight per row; unused entries are marked with
            index = -1 and weight = 0.
        timing / stats: dicts appended only when the corresponding
            with_* flag is set
    """
    idx, wgt, timing, stats = _similarity_graph_exact(
        ptr,
        ind,
        is_symmetric=is_symmetric,
        k=k,
        with_timing=with_timing,
        with_stats=with_stats,
        ctx=ctx,
    )
    n = ptr.shape[0] - 1
    if k <= 0:
        nnz = ind.shape[0] if hasattr(ind, "shape") else len(ind)
        k = min(32, max(16, 2 * nnz // n))
    idx = _asarray_like(idx, ptr).reshape(n, k)
    wgt = _asarray_like(wgt, ptr).reshape(n, k)
    result = (idx, wgt)
    if with_timing:
        result += (timing,)
    if with_stats:
        result += (stats,)
    return result if len(result) > 2 else result


def chunk_greedy(
    ptr,
    ind,
    *,
    is_symmetric=False,
    k=0,
    oversampling=0.0,
    chunk_size=0,
    with_timing=False,
    with_stats=False,
    ctx=None,
):
    """One-shot chunked greedy seriation.

    Computes Dice-similarity top-k internally, partitions rows into
    fixed-size chunks, runs a warp-cooperative nearest-neighbour greedy
    chain inside each chunk, concatenates the results, and returns a row
    permutation (new2old format) suitable for sporder.reorder().

    Arguments:
        ptr: int32 CUDA array of length n+1 (CSR row pointers)
        ind: int32 CUDA array of length nnz (CSR column indices)
        is_symmetric: whether the sparsity pattern is symmetric
        k: similarity top-k (0 = auto)
        oversampling: space-saving sketch oversampling (0 = auto)
        chunk_size: rows per chunk (0 = auto, default 1024)
        with_timing: also return a dict of timing measurements
        with_stats: also return a dict of algorithm diagnostics
            (k_used, m_used, avg_degree, avg_edge_weight,
            zero_degree_rows, chunk_size_used, chain_restarts,
            avg_followed_weight, cut_weight_fraction); similarity collection
            is excluded from stage timings, only wall_ms includes it.
            Useful derived quantities: break rate is
            chain_restarts / (n - ceil(n / chunk_size_used)); selection
            advantage is avg_followed_weight / avg_edge_weight (> 1 is
            healthy); the random-order baseline for cut_weight_fraction
            is 1 - (chunk_size_used - 1) / (n - 1)
        ctx: Context instance; None uses the shared default context

    Accepts any DLPack-compatible GPU array (PyTorch, CuPy, ...).

    Returns:
        new_order: int32 CUDA array of length n in new2old form:
            the output's row p contains the input's row new_order[p],
            in the same framework as `ptr`
        timing: dict of timing measurements, appended to the result
            only when with_timing is True
        stats: dict of algorithm diagnostics, appended to the result
            only when with_stats is True
    """
    new_order, timing, stats = _chunk_greedy(
        ptr,
        ind,
        is_symmetric=is_symmetric,
        k=k,
        oversampling=oversampling,
        chunk_size=chunk_size,
        with_timing=with_timing,
        with_stats=with_stats,
        ctx=ctx,
    )
    result = (_asarray_like(new_order, ptr),)
    if with_timing:
        result += (timing,)
    if with_stats:
        result += (stats,)
    return result if len(result) > 1 else result[0]


def chunk_similarity_greedy(
    ptr,
    ind,
    *,
    is_symmetric=False,
    k=0,
    oversampling=0.0,
    chunk_size=0,
    with_timing=False,
    with_stats=False,
    ctx=None,
):
    """One-shot per-chunk similarity + greedy seriation.

    Partitions rows into fixed-size chunks, computes the top-k
    Dice-similarity graph within each chunk independently, then runs
    warp-cooperative greedy chaining inside each chunk.  The similarity
    work is O(chunk_size * d^2) per chunk -- a fraction C/n of the
    global similarity cost used by chunk_greedy.

    Arguments match chunk_greedy.  Chunk quality is bounded: rows
    only find neighbours within their own chunk.

    Returns:
        new_order: int32 CUDA array of length n in new2old form
        timing: dict (wall_ms, csr2csc_ms, similarity_ms, chunk_greedy_ms)
        stats: dict (same diagnostics as chunk_greedy)
    """
    new_order, timing, stats = _chunk_similarity_greedy(
        ptr,
        ind,
        is_symmetric=is_symmetric,
        k=k,
        oversampling=oversampling,
        chunk_size=chunk_size,
        with_timing=with_timing,
        with_stats=with_stats,
        ctx=ctx,
    )
    result = (_asarray_like(new_order, ptr),)
    if with_timing:
        result += (timing,)
    if with_stats:
        result += (stats,)
    return result if len(result) > 1 else result[0]
