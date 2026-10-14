"""FlashSparse wrapper."""
import subprocess
import sys

import numpy as np
import scipy.sparse as sp
import torch

from ..base import SpMMMethod
from ..paths import FLASH_ROOT
from ..utils import compute_degrees, pad_csr_for_flashsparse
from ._cuda_build import cuda_build_env, extension_ready

_FLASH_BLOCK_TF32_TILE: tuple[int, int] | None = None


def _flash_block_tf32_tile() -> tuple[int, int]:
    global _FLASH_BLOCK_TF32_TILE
    if _FLASH_BLOCK_TF32_TILE is not None:
        return _FLASH_BLOCK_TF32_TILE

    import FS_Block  # noqa: PLC0415

    _FLASH_BLOCK_TF32_TILE = (
        int(FS_Block.TF32_BLOCK_WINDOW),
        int(FS_Block.TF32_BLOCK_WIDE),
    )
    return _FLASH_BLOCK_TF32_TILE


def _flash_extensions_ready(env: dict[str, str]) -> bool:
    if not extension_ready("FS_SpMM", ("prepare_tf32_map", "run_tf32_map"), env):
        return False
    if not extension_ready(
        "FS_Block", ("TF32_BLOCK_WINDOW", "TF32_BLOCK_WIDE"), env
    ):
        return False
    # Patched FS_SpMM exposes _bench_api_version; old installs lack the 5-arg prepare API.
    check = (
        "import FS_SpMM, sys\n"
        "sys.exit(0 if getattr(FS_SpMM, '_bench_api_version', 0) >= 2 else 1)\n"
    )
    return (
        subprocess.run(
            [sys.executable, "-c", check],
            capture_output=True,
            env=env,
        ).returncode
        == 0
    )


class FlashSparse(SpMMMethod):
    """FlashSparse TF32 map variant (GNN degree-normalized)."""

    @property
    def name(self) -> str:
        return "flashsparse"

    @property
    def tc_h(self) -> int:
        return _flash_block_tf32_tile()[0]

    @property
    def tc_w(self) -> int:
        return _flash_block_tf32_tile()[1]

    def build(self) -> None:
        env = cuda_build_env()
        if _flash_extensions_ready(env):
            return
        subprocess.check_call(
            f"cd {FLASH_ROOT} && {sys.executable} setup.py install",
            shell=True,
            executable="/bin/bash",
            env=env,
        )

    def prepare(self, A_csr: sp.csr_matrix) -> object:
        import FS_Block  # noqa: PLC0415
        import FS_SpMM  # noqa: PLC0415

        padded_csr, num_nodes_ori, num_nodes_pad = pad_csr_for_flashsparse(A_csr)
        row_ptr = torch.from_numpy(padded_csr.indptr.astype(np.int32))
        col_idx = torch.from_numpy(padded_csr.indices.astype(np.int32))

        node_factors = compute_degrees(A_csr).float()
        if num_nodes_pad > num_nodes_ori:
            pad = num_nodes_pad - num_nodes_ori
            node_factors = torch.cat(
                [node_factors, torch.zeros(pad, dtype=torch.float32)]
            )

        tc_h, tc_w = _flash_block_tf32_tile()
        row_ptr, col_idx, edge_values = FS_Block.blockProcess_tf32(
            row_ptr, col_idx, node_factors, tc_h, tc_w
        )
        return FS_SpMM.prepare_tf32_map(
            row_ptr,
            col_idx,
            edge_values,
            num_nodes_pad,
            num_nodes_ori,
        )

    def run(self, prepared_A: object, B: torch.Tensor) -> tuple[torch.Tensor, float]:
        import FS_SpMM  # noqa: PLC0415

        num_nodes = B.size(0)
        dimN = B.size(1)
        self.check_dense_limits(num_nodes, dimN)
        output, kernel_ms_tensor = FS_SpMM.run_tf32_map(
            prepared_A,
            B.contiguous(),
            dimN,
            num_nodes,
            1,
        )
        return output, float(kernel_ms_tensor.item())
