"""cuSPARSE SpGEMM wrapper (A * A, CUSPARSE_SPGEMM_DEFAULT)."""
import subprocess
import sys
import tempfile
from pathlib import Path

import scipy.sparse as sp
import torch

from ..base import SpMMMethod
from ._cuda_build import cuda_build_env, extension_ready
from .cusparse_spmm import _cusparse_library_dirs

HERE = Path(__file__).parent.resolve()


class CuSparseSpGEMM(SpMMMethod):
    """cuSPARSE CSR SpGEMM baseline computing C = A * A."""

    @property
    def name(self) -> str:
        return "cusparse-spgemm"

    @property
    def uses_feature_dim(self) -> bool:
        return False

    @property
    def tc_h(self) -> int:
        return 0

    @property
    def tc_w(self) -> int:
        return 0

    def build(self) -> None:
        env = cuda_build_env()
        if extension_ready("cusparse_spgemm_bench", ("prepare", "run"), env):
            return

        lib_dirs = _cusparse_library_dirs(env)
        lib_dirs_literal = ", ".join(repr(path) for path in lib_dirs)
        setup_py = f"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cusparse_spgemm_bench',
    ext_modules=[
        CUDAExtension(
            'cusparse_spgemm_bench',
            [
                '{HERE}/cusparse_spgemm_binding.cpp',
                '{HERE}/cusparse_spgemm.cu',
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
        import cusparse_spgemm_bench  # noqa: PLC0415

        num_rows, num_cols = A_csr.shape
        if num_rows != num_cols:
            raise RuntimeError(
                f"cusparse-spgemm requires a square CSR matrix; got {num_rows}x{num_cols}"
            )

        indptr = A_csr.indptr.astype("int32")
        indices = A_csr.indices.astype("int32")
        values = A_csr.data.astype("float32")
        # B = A (same arrays uploaded as both operands).
        return cusparse_spgemm_bench.prepare(
            indptr,
            indices,
            values,
            indptr,
            indices,
            values,
            num_rows,
        )

    def run(
        self, prepared_A: object, B: torch.Tensor | None
    ) -> tuple[torch.Tensor, float]:
        import cusparse_spgemm_bench  # noqa: PLC0415

        _ = B
        kernel_ms = float(cusparse_spgemm_bench.run(prepared_A))
        return torch.empty(0, dtype=torch.float32, device="cuda"), kernel_ms
