"""Voltrix-SpMM wrapper."""
import functools
import os
import re
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from ..base import SpMMMethod
from ..paths import VOLTRIX_ROOT
from ..utils import check_voltrix_prepare_limits
from ._cuda_build import cuda_build_env

_VOLTRIX_ROOT = str(VOLTRIX_ROOT)
_MIN_NVCC_VERSION = "12.0"
_VOLTRIX_TC_TILE: tuple[int, int] | None = None


def _voltrix_tc_tile() -> tuple[int, int]:
    global _VOLTRIX_TC_TILE
    if _VOLTRIX_TC_TILE is not None:
        return _VOLTRIX_TC_TILE

    _init_voltrix()
    from voltrix.spmm import BLK_H, BLK_W  # noqa: PLC0415

    _VOLTRIX_TC_TILE = (int(BLK_H), int(BLK_W))
    return _VOLTRIX_TC_TILE


def _ensure_voltrix_importable() -> None:
    if _VOLTRIX_ROOT not in sys.path:
        sys.path.insert(0, _VOLTRIX_ROOT)


def _patch_voltrix_nvcc() -> None:
    """Use the same conda nvcc 12.x as other methods (via cuda_build_env).

    spmm_kernel needs cccl headers for cuda::barrier experimental APIs; those
    come from get_cccl_include_dir() in voltrix/jit/compiler.py (system cccl
    path is headers-only — nvcc can stay at 12.0).
    """
    from voltrix.jit import compiler as jit_compiler

    if getattr(jit_compiler, "_reorder_bench_nvcc_patched", False):
        return

    @functools.lru_cache(maxsize=None)
    def get_nvcc_compiler():
        from voltrix.project import NVCC_COMPILER_FLAG

        env = cuda_build_env()
        paths: list[str] = []
        if os.getenv(NVCC_COMPILER_FLAG):
            paths.append(os.getenv(NVCC_COMPILER_FLAG))
        cuda_home = env.get("CUDA_HOME", "")
        if cuda_home:
            paths.append(str(Path(cuda_home) / "bin" / "nvcc"))
        paths.append("/usr/local/cuda/bin/nvcc")

        version_pattern = re.compile(r"release (\d+\.\d+)")
        seen: set[str] = set()
        for path in paths:
            if not path or path in seen or not os.path.exists(path):
                continue
            seen.add(path)
            match = version_pattern.search(os.popen(f"{path} --version").read())
            if not match:
                continue
            version = match.group(1)
            if version >= _MIN_NVCC_VERSION:
                return path, version
        raise RuntimeError(
            f"Cannot find nvcc >= {_MIN_NVCC_VERSION} for Voltrix JIT "
            f"(checked {list(seen)}). Ensure conda cuda-toolkit is installed "
            f"or set VOLTRIX_NVCC_COMPILER."
        )

    jit_compiler.get_nvcc_compiler = get_nvcc_compiler
    jit_compiler._reorder_bench_nvcc_patched = True


def _patch_voltrix_profiling() -> None:
    """Use CUDA events instead of kineto (broken on PyTorch 2.4 without use_cpu)."""
    from voltrix import utils as voltrix_utils

    if getattr(voltrix_utils, "_reorder_bench_patched", False):
        return

    orig_bench = voltrix_utils.GPU_bench

    def gpu_bench(func, iters=100, warmup=30, kernel_name=None):
        return orig_bench(func, iters=iters, warmup=warmup, kernel_name=None)

    voltrix_utils.GPU_bench = gpu_bench
    voltrix_utils._reorder_bench_patched = True


def _init_voltrix() -> None:
    _ensure_voltrix_importable()
    _patch_voltrix_nvcc()
    _patch_voltrix_profiling()


class VoltrixSpMM(SpMMMethod):
    """Voltrix-SpMM wrapper."""

    @property
    def name(self) -> str:
        return "voltrix"

    @property
    def tc_h(self) -> int:
        return _voltrix_tc_tile()[0]

    @property
    def tc_w(self) -> int:
        return _voltrix_tc_tile()[1]

    def build(self) -> None:
        _init_voltrix()

    def prepare(self, A_csr: sp.csr_matrix) -> tuple:
        _init_voltrix()
        import voltrix  # noqa: PLC0415

        tc_h, tc_w = _voltrix_tc_tile()
        check_voltrix_prepare_limits(A_csr, tc_h, tc_w)

        num_nodes = A_csr.shape[0]
        num_edges = A_csr.nnz

        indptr = torch.from_numpy(A_csr.indptr.astype(np.int32))
        indices = torch.from_numpy(A_csr.indices.astype(np.int32))

        blk_offsets, hspa_packed, hind = voltrix.csr_preprocess(
            indptr, indices, num_nodes
        )
        return blk_offsets, hspa_packed, hind, num_nodes, num_edges

    def run(self, prepared_A: tuple, B: torch.Tensor) -> tuple[torch.Tensor, float]:
        _init_voltrix()
        import voltrix  # noqa: PLC0415

        blk_offsets, hspa_packed, hind, num_nodes, num_edges = prepared_A
        num_feats = B.shape[1]
        self.check_dense_limits(num_nodes, num_feats)
        output = torch.empty((num_nodes, num_feats), dtype=torch.float32, device=B.device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        voltrix.jit_kernels.spmm_kernel(
            blk_offsets,
            hspa_packed,
            hind,
            num_nodes=num_nodes,
            num_edges=num_edges,
            embedding_dim=num_feats,
            input=B,
            output=output,
        )
        end.record()
        torch.cuda.synchronize()
        kernel_ms = start.elapsed_time(end)
        return output, kernel_ms
