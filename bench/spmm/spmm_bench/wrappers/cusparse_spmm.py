"""cuSPARSE SpMM wrapper (CUSPARSE_SPMM_ALG_DEFAULT)."""
import subprocess
import sys
import tempfile
from pathlib import Path

import scipy.sparse as sp
import torch

from ..base import SpMMMethod
from ._cuda_build import cuda_build_env, extension_ready

HERE = Path(__file__).parent.resolve()


def _cusparse_library_dirs(env: dict[str, str]) -> list[str]:
    prefix = Path(env.get("CONDA_PREFIX", ""))
    dirs: list[str] = []
    for candidate in (prefix / "lib", prefix / "targets/sbsa-linux/lib"):
        if candidate.is_dir():
            dirs.append(str(candidate))
    return dirs


class CuSparseSpMM(SpMMMethod):
    """cuSPARSE CSR SpMM baseline using CUSPARSE_SPMM_ALG_DEFAULT."""

    @property
    def name(self) -> str:
        return "cusparse-spmm"

    @property
    def tc_h(self) -> int:
        return 0

    @property
    def tc_w(self) -> int:
        return 0

    def build(self) -> None:
        env = cuda_build_env()
        if extension_ready("cusparse_spmm_bench", ("prepare", "run"), env):
            return

        lib_dirs = _cusparse_library_dirs(env)
        lib_dirs_literal = ", ".join(repr(path) for path in lib_dirs)
        setup_py = f"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cusparse_spmm_bench',
    ext_modules=[
        CUDAExtension(
            'cusparse_spmm_bench',
            [
                '{HERE}/cusparse_spmm_binding.cpp',
                '{HERE}/cusparse_spmm.cu',
            ],
            include_dirs=['{HERE}'],
            libraries=['cusparse'],
            library_dirs=[{lib_dirs_literal}],
            extra_compile_args={{
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': ['-O3'],
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
        import cusparse_spmm_bench  # noqa: PLC0415

        num_rows, num_cols = A_csr.shape
        if num_rows != num_cols:
            raise RuntimeError(
                f"cusparse-spmm requires a square CSR matrix; got {num_rows}x{num_cols}"
            )
        return cusparse_spmm_bench.prepare(
            A_csr.indptr.astype("int32"),
            A_csr.indices.astype("int32"),
            A_csr.data.astype("float32"),
            num_rows,
            num_cols,
            int(A_csr.nnz),
        )

    def run(self, prepared_A: object, B: torch.Tensor) -> tuple[torch.Tensor, float]:
        import cusparse_spmm_bench  # noqa: PLC0415

        num_nodes = B.size(0)
        feature_dim = B.size(1)
        self.check_dense_limits(num_nodes, feature_dim)
        output = torch.empty(
            (num_nodes, feature_dim), dtype=torch.float32, device=B.device
        )
        kernel_ms = float(
            cusparse_spmm_bench.run(prepared_A, B.contiguous(), output)
        )
        return output, kernel_ms
