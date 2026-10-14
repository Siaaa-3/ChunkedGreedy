"""DTC-SpMM wrapper."""
import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from ..base import SpMMMethod
from ..paths import DTC_ROOT
from ..utils import csr_to_torch
from ._cuda_build import (
    cuda_build_env,
    extension_importable,
    extension_source_newer_than_binary,
)

DTC_HOME = DTC_ROOT
DTC_SPmm_DIR = DTC_ROOT / "DTC-SpMM"
THIRD_PARTY_DIR = DTC_ROOT / "third_party"
GLOG_DIR = THIRD_PARTY_DIR / "glog"
SPUTNIK_DIR = THIRD_PARTY_DIR / "sputnik"

DTC_BLK_H = 16
DTC_BLK_W = 8
# cudaDevAttrMaxSharedMemoryPerBlockOptin on Hopper (sm_90): 227 KiB
_HOPPER_HARDWARE_OPTIN_BYTES = 232448
# Reserve headroom for static shared / driver overhead; do not use hardware max.
_SHARED_MEM_SAFETY_MARGIN = 32764
_CUDA_ATTR_MAX_SHARED_OPTIN = 97


def _estimate_max_tc_blocks(A_csr: sp.csr_matrix) -> int:
    """Mirror DTC-SpMM blockPartition: max TC blocks in any row window."""
    num_nodes = A_csr.shape[0]
    indptr, cols = A_csr.indptr, A_csr.indices
    max_blocks = 0
    for w in range((num_nodes + DTC_BLK_H - 1) // DTC_BLK_H):
        r0 = w * DTC_BLK_H
        r1 = min((w + 1) * DTC_BLK_H, num_nodes)
        start, end = indptr[r0], indptr[r1]
        if start >= end:
            continue
        unique_cols = np.unique(cols[start:end])
        n = (len(unique_cols) + DTC_BLK_W - 1) // DTC_BLK_W
        if n > max_blocks:
            max_blocks = n
    return max_blocks


def _cuda_hardware_max_dynamic_shared_bytes(device: int = 0) -> int:
    """Hardware opt-in ceiling (cudaDevAttrMaxSharedMemoryPerBlockOptin)."""
    if not torch.cuda.is_available():
        return _HOPPER_HARDWARE_OPTIN_BYTES
    try:
        libcudart = ctypes.CDLL("libcudart.so")
        val = ctypes.c_int()
        ret = libcudart.cudaDeviceGetAttribute(
            ctypes.byref(val), _CUDA_ATTR_MAX_SHARED_OPTIN, device
        )
        if ret == 0 and val.value > 0:
            return val.value
    except OSError:
        pass
    return _HOPPER_HARDWARE_OPTIN_BYTES


def _cuda_usable_dynamic_shared_bytes(device: int = 0) -> int:
    """Usable dynamic shared budget with safety margin below hardware max."""
    return max(0, _cuda_hardware_max_dynamic_shared_bytes(device) - _SHARED_MEM_SAFETY_MARGIN)


def _check_preprocess_shared_limit(A_csr: sp.csr_matrix) -> None:
    """Raise RuntimeError if preprocess shared memory exceeds usable limit."""
    max_blocks = _estimate_max_tc_blocks(A_csr)
    needed = (2 * max_blocks + 1) * 4
    usable = _cuda_usable_dynamic_shared_bytes()
    hardware_max = _cuda_hardware_max_dynamic_shared_bytes()
    if needed > usable:
        raise RuntimeError(
            f"DTC-SpMM preprocess requires {needed} bytes "
            f"({needed / 1024:.1f} KiB) dynamic shared memory but usable "
            f"limit is {usable} bytes ({usable / 1024:.1f} KiB, "
            f"hardware opt-in max {hardware_max} bytes minus "
            f"{_SHARED_MEM_SAFETY_MARGIN} byte margin); "
            f"max_tc_blocks={max_blocks}"
        )


def _preload_dtc_libs() -> None:
    """Preload third-party shared libs before importing DTCSpMM."""
    for lib in (
        GLOG_DIR / "build/lib/libglog.so",
        SPUTNIK_DIR / "build/sputnik/libsputnik.so",
    ):
        if lib.exists():
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)


def _build_third_party() -> None:
    env = os.environ.copy()
    env["DTC_HOME"] = str(DTC_HOME)
    env["GLOG_PATH"] = str(GLOG_DIR)
    env["SPUTNIK_PATH"] = str(SPUTNIK_DIR)
    env["LD_LIBRARY_PATH"] = (
        env.get("LD_LIBRARY_PATH", "")
        + f":{GLOG_DIR}/build/lib"
        + f":{SPUTNIK_DIR}/build/sputnik"
    )
    env["CPLUS_INCLUDE_PATH"] = (
        env.get("CPLUS_INCLUDE_PATH", "")
        + f":{GLOG_DIR}/build/include"
    )
    env["LIBRARY_PATH"] = (
        env.get("LIBRARY_PATH", "")
        + f":{GLOG_DIR}/build/lib"
        + f":{SPUTNIK_DIR}/build/sputnik"
    )

    if not (GLOG_DIR / "build/lib/libglog.so").exists():
        subprocess.check_call(
            f"mkdir -p {GLOG_DIR}/build && cd {GLOG_DIR}/build && "
            f"cmake -DCMAKE_INSTALL_PREFIX={GLOG_DIR}/build .. && make -j && make install",
            shell=True, executable="/bin/bash", env=env,
        )

    if not (SPUTNIK_DIR / "build/sputnik/libsputnik.so").exists():
        glog_def = "-DGLOG_USE_GLOG_EXPORT"
        subprocess.check_call(
            f"mkdir -p {SPUTNIK_DIR}/build && cd {SPUTNIK_DIR}/build && "
            f"cmake .. -DGLOG_INCLUDE_DIR={GLOG_DIR}/build/include "
            f"-DGLOG_LIBRARY={GLOG_DIR}/build/lib/libglog.so "
            f"-DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=OFF -DBUILD_BENCHMARK=OFF "
            f"-DCMAKE_CXX_STANDARD=17 -DCMAKE_CUDA_STANDARD=17 "
            f'-DCMAKE_CXX_FLAGS="{glog_def}" '
            f'-DCMAKE_CUDA_FLAGS="{glog_def}" '
            f"-DCUDA_ARCHS=\"90\" && make -j",
            shell=True, executable="/bin/bash", env=env,
        )


class _DTCSpMMBase(SpMMMethod):
    """Shared build/prepare logic for DTC-SpMM variants."""

    @property
    def tc_h(self) -> int:
        return DTC_BLK_H

    @property
    def tc_w(self) -> int:
        return DTC_BLK_W

    def _check_run_limits(self, B: torch.Tensor) -> None:
        self.check_dense_limits(B.size(0), B.size(1))

    def build(self) -> None:
        _build_third_party()
        env = cuda_build_env()
        env["DTC_HOME"] = str(DTC_HOME)
        env["GLOG_PATH"] = str(GLOG_DIR)
        env["SPUTNIK_PATH"] = str(SPUTNIK_DIR)
        env["LD_LIBRARY_PATH"] = (
            env.get("LD_LIBRARY_PATH", "")
            + f":{GLOG_DIR}/build/lib"
            + f":{SPUTNIK_DIR}/build/sputnik"
        )
        env["CPLUS_INCLUDE_PATH"] = (
            env.get("CPLUS_INCLUDE_PATH", "")
            + f":{GLOG_DIR}/build/include"
        )
        env["LIBRARY_PATH"] = (
            env.get("LIBRARY_PATH", "")
            + f":{GLOG_DIR}/build/lib"
            + f":{SPUTNIK_DIR}/build/sputnik"
        )

        dtc_sources = (
            DTC_SPmm_DIR / "DTCSpMM_kernel.cu",
            DTC_SPmm_DIR / "DTCSpMM.cpp",
        )
        if (
            not extension_importable("DTCSpMM", env)
            or extension_source_newer_than_binary("DTCSpMM", dtc_sources, env)
        ):
            subprocess.check_call(
                f"cd {DTC_SPmm_DIR} && TORCH_CUDA_ARCH_LIST=\"9.0\" "
                f"{sys.executable} setup.py install",
                shell=True, executable="/bin/bash", env=env,
            )

    def prepare(self, A_csr: sp.csr_matrix) -> tuple:
        _check_preprocess_shared_limit(A_csr)
        _preload_dtc_libs()
        import DTCSpMM  # noqa: PLC0415

        num_nodes = A_csr.shape[0]
        num_edges = A_csr.nnz
        row_ptr, col_idx, _ = csr_to_torch(A_csr, device="cuda")

        num_row_windows = (num_nodes + DTC_BLK_H - 1) // DTC_BLK_H

        block_partition = torch.zeros(num_row_windows, dtype=torch.int32, device="cuda")
        edge_to_column = torch.zeros(num_edges, dtype=torch.int32, device="cuda")
        edge_to_row = torch.zeros(num_edges, dtype=torch.int32, device="cuda")

        return DTCSpMM.preprocess_gpu(
            col_idx, row_ptr, num_nodes, DTC_BLK_H, DTC_BLK_W,
            block_partition, edge_to_column, edge_to_row,
        )


class DTCSpMM(_DTCSpMMBase):
    """DTC-SpMM non-balance variant."""

    @property
    def name(self) -> str:
        return "dtc-spmm"

    def run(self, prepared_A: tuple, B: torch.Tensor) -> tuple[torch.Tensor, float]:
        import DTCSpMM  # noqa: PLC0415

        self._check_run_limits(B)

        row_window_offset, tcblock_rowid, tcblocktile_id, tcblock_offset, \
            sparse_a_to_x_idx, block_counter = prepared_A

        num_nodes = B.size(0)
        num_nnz = sparse_a_to_x_idx.numel()

        outputs = DTCSpMM.run_DTCSpMM(
            B, row_window_offset, tcblocktile_id, tcblock_offset,
            sparse_a_to_x_idx, num_nodes, num_nnz, "float_nonsplit",
        )
        return outputs[0], float(outputs[1].item())


class DTCSpMMBalance(_DTCSpMMBase):
    """DTC-SpMM balance variant."""

    @property
    def name(self) -> str:
        return "dtc-spmm-balance"

    def run(self, prepared_A: tuple, B: torch.Tensor) -> tuple[torch.Tensor, float]:
        import DTCSpMM  # noqa: PLC0415

        self._check_run_limits(B)

        row_window_offset, tcblock_rowid, tcblocktile_id, tcblock_offset, \
            sparse_a_to_x_idx, block_counter = prepared_A

        num_nodes = B.size(0)

        outputs = DTCSpMM.run_DTCSpMM_balance(
            B, tcblock_rowid, tcblocktile_id, tcblock_offset,
            sparse_a_to_x_idx, num_nodes, "float4_split",
        )
        return outputs[0], float(outputs[1].item())
