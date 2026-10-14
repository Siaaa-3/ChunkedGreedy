"""AccSpMM wrapper with PyBind11 binding."""
import subprocess
import sys
import tempfile
from pathlib import Path

import scipy.sparse as sp
import torch

from ..base import SpMMMethod
from ..paths import ACC_ROOT
from ._cuda_build import cuda_build_env, extension_ready

HERE = Path(__file__).parent.resolve()


class AccSpMM(SpMMMethod):
    """AccSpMM TF32 wrapper."""

    @property
    def name(self) -> str:
        return "acc-spmm"

    @property
    def tc_h(self) -> int:
        import acc_spmm_bench  # noqa: PLC0415

        return int(acc_spmm_bench.ROW_WINDOW)

    @property
    def tc_w(self) -> int:
        import acc_spmm_bench  # noqa: PLC0415

        return int(acc_spmm_bench.COL_WINDOW)

    def build(self) -> None:
        env = cuda_build_env()
        if extension_ready(
            "acc_spmm_bench",
            ("prepare", "prepare_with_balance", "run", "ROW_WINDOW", "COL_WINDOW"),
            env,
        ):
            return

        setup_py = f"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='acc_spmm_bench',
    ext_modules=[
        CUDAExtension(
            'acc_spmm_bench',
            [
                '{HERE}/acc_spmm_binding.cpp',
                '{ACC_ROOT}/src/mma_tf32.cu',
            ],
            include_dirs=['{ACC_ROOT}/include', '{HERE}'],
            extra_compile_args={{
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3', '-arch=sm_90', '--use_fast_math'],
            }},
        )
    ],
    cmdclass={{'build_ext': BuildExtension}},
)
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_path = Path(tmpdir) / "setup.py"
            setup_path.write_text(setup_py)
            subprocess.check_call(
                [sys.executable, str(setup_path), "install"],
                env=env,
            )

    def prepare(self, A_csr: sp.csr_matrix) -> object:
        import acc_spmm_bench  # noqa: PLC0415

        return acc_spmm_bench.prepare(
            A_csr.indptr.astype("int32"),
            A_csr.indices.astype("int32"),
            A_csr.data.astype("float32"),
            A_csr.shape[0],
            A_csr.nnz,
        )

    def run(self, prepared_A: object, B: torch.Tensor) -> tuple[torch.Tensor, float]:
        import acc_spmm_bench  # noqa: PLC0415

        num_nodes = B.size(0)
        feature_dim = B.size(1)
        self.check_dense_limits(num_nodes, feature_dim)
        kernel_ms = float(acc_spmm_bench.run(prepared_A, B.contiguous()))
        output = torch.empty((num_nodes, feature_dim), dtype=torch.float32, device=B.device)
        return output, kernel_ms


class AccSpMMBalance(AccSpMM):
    """AccSpMM wrapper that always selects the load-balanced sparse format."""

    @property
    def name(self) -> str:
        return "acc-spmm-balance"

    def prepare(self, A_csr: sp.csr_matrix) -> object:
        import acc_spmm_bench  # noqa: PLC0415

        return acc_spmm_bench.prepare_with_balance(
            A_csr.indptr.astype("int32"),
            A_csr.indices.astype("int32"),
            A_csr.data.astype("float32"),
            A_csr.shape[0],
            A_csr.nnz,
            True,
        )
