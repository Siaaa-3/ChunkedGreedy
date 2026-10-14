"""Utilities for dense matrix generation and sparse format conversions."""
import ctypes
import gc
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

# Kernel dense linear-index limits (row * feature_dim + col).
INT32_LINEAR_INDEX_LIMIT = 2_147_483_647
UINT32_LINEAR_INDEX_LIMIT = 4_294_967_295
# FlashSparse dense offsets use C long (64-bit on Linux); launch dims are int.
INT64_LINEAR_INDEX_LIMIT = (1 << 63) - 1


@dataclass(frozen=True)
class DenseIndexLimits:
    """Dense-matrix limits aligned with a vendored kernel's index types."""

    kernel_index_type: str
    max_linear_elems: int
    max_num_nodes: int
    max_feature_dim: int


# Host checks mirror kernel-side index arithmetic (see method wrappers).
METHOD_DENSE_INDEX_LIMITS: dict[str, DenseIndexLimits] = {
    "voltrix": DenseIndexLimits(
        kernel_index_type="int32_t",
        max_linear_elems=INT32_LINEAR_INDEX_LIMIT,
        max_num_nodes=INT32_LINEAR_INDEX_LIMIT,
        max_feature_dim=INT32_LINEAR_INDEX_LIMIT,
    ),
    "flashsparse": DenseIndexLimits(
        kernel_index_type="int32_t launch / long dense offset",
        max_linear_elems=INT64_LINEAR_INDEX_LIMIT,
        max_num_nodes=INT32_LINEAR_INDEX_LIMIT,
        max_feature_dim=INT32_LINEAR_INDEX_LIMIT,
    ),
    "dtc-spmm": DenseIndexLimits(
        kernel_index_type="uint32_t / unsigned",
        max_linear_elems=UINT32_LINEAR_INDEX_LIMIT,
        max_num_nodes=UINT32_LINEAR_INDEX_LIMIT,
        max_feature_dim=UINT32_LINEAR_INDEX_LIMIT,
    ),
    "dtc-spmm-balance": DenseIndexLimits(
        kernel_index_type="uint32_t / unsigned",
        max_linear_elems=UINT32_LINEAR_INDEX_LIMIT,
        max_num_nodes=UINT32_LINEAR_INDEX_LIMIT,
        max_feature_dim=UINT32_LINEAR_INDEX_LIMIT,
    ),
    "acc-spmm": DenseIndexLimits(
        kernel_index_type="vint (uint32_t)",
        max_linear_elems=UINT32_LINEAR_INDEX_LIMIT,
        max_num_nodes=UINT32_LINEAR_INDEX_LIMIT,
        max_feature_dim=UINT32_LINEAR_INDEX_LIMIT,
    ),
    "acc-spmm-balance": DenseIndexLimits(
        kernel_index_type="vint (uint32_t)",
        max_linear_elems=UINT32_LINEAR_INDEX_LIMIT,
        max_num_nodes=UINT32_LINEAR_INDEX_LIMIT,
        max_feature_dim=UINT32_LINEAR_INDEX_LIMIT,
    ),
    "cusparse-spmm": DenseIndexLimits(
        kernel_index_type="int32_t (CSR indices)",
        max_linear_elems=INT32_LINEAR_INDEX_LIMIT,
        max_num_nodes=INT32_LINEAR_INDEX_LIMIT,
        max_feature_dim=INT32_LINEAR_INDEX_LIMIT,
    ),
    "cusparse-spgemm": DenseIndexLimits(
        kernel_index_type="int32_t (CSR indices)",
        max_linear_elems=INT32_LINEAR_INDEX_LIMIT,
        max_num_nodes=INT32_LINEAR_INDEX_LIMIT,
        max_feature_dim=INT32_LINEAR_INDEX_LIMIT,
    ),
    "gustavson-spgemm": DenseIndexLimits(
        kernel_index_type="int64_t (CSR indices)",
        max_linear_elems=INT64_LINEAR_INDEX_LIMIT,
        max_num_nodes=INT64_LINEAR_INDEX_LIMIT,
        max_feature_dim=INT64_LINEAR_INDEX_LIMIT,
    ),
}


def dense_index_limits(method: str) -> DenseIndexLimits:
    """Return dense indexing limits for a registered bench method."""
    try:
        return METHOD_DENSE_INDEX_LIMITS[method]
    except KeyError as exc:
        known = ", ".join(sorted(METHOD_DENSE_INDEX_LIMITS))
        raise ValueError(f"Unknown method {method!r}. Known: {known}") from exc


def check_method_dense_limits(
    method: str,
    num_nodes: int,
    feature_dim: int,
) -> None:
    """Raise ``RuntimeError`` if dense shape exceeds kernel indexing limits.

    Called from each wrapper's ``run()`` immediately before the first kernel
    launch so batch CSV records a clear error instead of CUDA illegal access.
    """
    limits = dense_index_limits(method)
    if num_nodes > limits.max_num_nodes:
        raise RuntimeError(
            f"{method} num_nodes overflow: num_nodes({num_nodes}) > "
            f"{limits.max_num_nodes} ({limits.kernel_index_type})"
        )
    if feature_dim > limits.max_feature_dim:
        raise RuntimeError(
            f"{method} feature_dim overflow: feature_dim({feature_dim}) > "
            f"{limits.max_feature_dim} ({limits.kernel_index_type})"
        )
    linear_elems = num_nodes * feature_dim
    if linear_elems > limits.max_linear_elems:
        raise RuntimeError(
            f"{method} dense linear index overflow: num_nodes({num_nodes}) * "
            f"feature_dim({feature_dim}) = {linear_elems} > "
            f"{limits.max_linear_elems} ({limits.kernel_index_type})"
        )


def estimate_tc_block_count(
    csr: sp.csr_matrix,
    tc_h: int,
    tc_w: int,
) -> int:
    """Total TC tile count across row windows (spmatkit ``tc_num`` / Voltrix ``block_counter``)."""
    if not sp.isspmatrix_csr(csr):
        csr = csr.tocsr()
    num_rows = csr.shape[0]
    indptr = csr.indptr
    indices = csr.indices
    num_windows = (num_rows + tc_h - 1) // tc_h
    tc_num = 0
    for window in range(num_windows):
        r0 = window * tc_h
        r1 = min((window + 1) * tc_h, num_rows)
        start, end = indptr[r0], indptr[r1]
        if start >= end:
            continue
        n = len(np.unique(indices[start:end]))
        tc_num += (n + tc_w - 1) // tc_w
    return tc_num


def check_voltrix_prepare_limits(
    csr: sp.csr_matrix,
    tc_h: int,
    tc_w: int,
) -> None:
    """Raise if Voltrix ``csr_preprocess`` would overflow int32 linear indices.

    Voltrix allocates ``hspa`` with ``total_tc_blocks * tc_h * tc_w`` elements
    and indexes it with ``int32_t`` in host/GPU preprocess kernels.
    """
    tc_num = estimate_tc_block_count(csr, tc_h, tc_w)
    padded_elems = tc_num * tc_h * tc_w
    if padded_elems > INT32_LINEAR_INDEX_LIMIT:
        raise RuntimeError(
            f"voltrix prepare int32 overflow: tc_num({tc_num}) * "
            f"tc_h({tc_h}) * tc_w({tc_w}) = {padded_elems} > "
            f"{INT32_LINEAR_INDEX_LIMIT} (int32_t hspa linear index)"
        )


def check_dense_linear_limit(
    num_nodes: int,
    feature_dim: int,
    *,
    method: str,
    limit: int = INT32_LINEAR_INDEX_LIMIT,
) -> None:
    """Backward-compatible alias; prefer :func:`check_method_dense_limits`."""
    _ = limit
    check_method_dense_limits(method, num_nodes, feature_dim)


def recover_cuda_after_error(exc: BaseException | None = None) -> None:
    """Best-effort CUDA recovery after a kernel failure.

    After illegal memory access the device context is often poisoned; resetting
    the device clears sticky errors so later cases can run in the same process.
    Prepared GPU handles must be discarded after calling this.
    """
    err_msg = str(exc).lower() if exc is not None else ""
    signals = (
        "illegal memory access",
        "illegaladdress",
        "device-side assert",
        "cuda error",
        "failed to allocate",
        "out_of_memory",
        "out of memory",
        "bad_alloc",
    )
    should_reset = exc is None or any(signal in err_msg for signal in signals)
    try:
        torch.cuda.synchronize()
    except RuntimeError:
        pass
    sticky_error = False
    try:
        libcudart = ctypes.CDLL("libcudart.so")
        sticky_error = int(libcudart.cudaGetLastError()) != 0
        if sticky_error or should_reset:
            libcudart.cudaDeviceReset()
    except OSError:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        pass


def generate_dense(num_nodes: int, feature_dim: int,
                   device: str = "cuda", seed: int = 42,
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Generate a random dense matrix on the specified device."""
    gen = torch.Generator(device=device).manual_seed(seed)
    return torch.randn((num_nodes, feature_dim), generator=gen,
                       dtype=dtype, device=device)


def csr_to_torch(csr: sp.csr_matrix, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert scipy CSR to three torch tensors: row_ptr, col_idx, values."""
    row_ptr = torch.from_numpy(csr.indptr.astype(np.int32)).to(device)
    col_idx = torch.from_numpy(csr.indices.astype(np.int32)).to(device)
    values = torch.from_numpy(csr.data.astype(np.float32)).to(device)
    return row_ptr, col_idx, values


def csr_to_coo_edges(csr: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Return COO edge list (src, dst) as numpy arrays."""
    coo = csr.tocoo()
    return coo.row.astype(np.int32), coo.col.astype(np.int32)


def compute_degrees(csr: sp.csr_matrix) -> torch.Tensor:
    """Compute D^-1/2 as used by GNN normalization (FlashSparse style)."""
    degrees = np.array(csr.sum(axis=1)).flatten().astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    return torch.from_numpy(inv_sqrt.astype(np.float32))


def pad_csr_for_flashsparse(
    csr: sp.csr_matrix,
    multiple: int = 16,
) -> tuple[sp.csr_matrix, int, int]:
    """Pad CSR rows/cols to a multiple of ``multiple`` (FlashSparse requirement)."""
    num_nodes_ori = csr.shape[0]
    if num_nodes_ori % multiple == 0:
        return csr, num_nodes_ori, num_nodes_ori

    num_nodes_pad = num_nodes_ori + multiple - (num_nodes_ori % multiple)
    pad_rows = num_nodes_pad - num_nodes_ori
    indptr = np.concatenate(
        [csr.indptr, np.full(pad_rows, csr.indptr[-1], dtype=csr.indptr.dtype)]
    )
    padded = sp.csr_matrix(
        (csr.data, csr.indices, indptr),
        shape=(num_nodes_pad, num_nodes_pad),
    )
    return padded, num_nodes_ori, num_nodes_pad
