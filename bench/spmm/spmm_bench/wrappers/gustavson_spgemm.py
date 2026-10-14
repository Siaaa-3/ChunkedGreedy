"""Gustavson (RowSpGEMM) CPU SpGEMM wrapper: C = A * A."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import scipy.sparse as sp
import torch

from ..base import SpMMMethod
from ..paths import CLUSTERWISE_ROOT
from ._cuda_build import extension_ready, extension_source_newer_than_binary

HERE = Path(__file__).parent.resolve()
_CORE_CPP = HERE / "gustavson_spgemm_core.cpp"


def _build_env() -> dict[str, str]:
    return os.environ.copy()


def _write_simde_shim(shim_dir: Path) -> None:
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "immintrin.h").write_text(
        "#pragma once\n"
        "#define SIMDE_ENABLE_NATIVE_ALIASES\n"
        "#include <simde/x86/avx2.h>\n"
        "#ifndef _popcnt32\n"
        "#define _popcnt32(x) "
        "__builtin_popcount(static_cast<unsigned int>(x))\n"
        "#endif\n",
        encoding="utf-8",
    )
    (shim_dir / "x86intrin.h").write_text(
        '#pragma once\n#include "immintrin.h"\n',
        encoding="utf-8",
    )


class GustavsonSpGEMM(SpMMMethod):
    """CPU Gustavson SpGEMM baseline (clusterwise RowSpGEMM), C = A * A."""

    @property
    def name(self) -> str:
        return "gustavson-spgemm"

    @property
    def uses_feature_dim(self) -> bool:
        return False

    @property
    def uses_cuda(self) -> bool:
        return False

    @property
    def tc_h(self) -> int:
        return 0

    @property
    def tc_w(self) -> int:
        return 0

    def build(self) -> None:
        env = _build_env()
        sources = [_CORE_CPP]
        if extension_ready(
            "gustavson_spgemm_bench", ("prepare", "run", "last_nnz"), env
        ) and not extension_source_newer_than_binary(
            "gustavson_spgemm_bench", sources, env
        ):
            return

        if not CLUSTERWISE_ROOT.is_dir():
            raise RuntimeError(
                f"clusterwise sources not found at {CLUSTERWISE_ROOT}"
            )

        conda_prefix = env.get("CONDA_PREFIX", "")
        include_dirs = [
            str(CLUSTERWISE_ROOT),
            str(CLUSTERWISE_ROOT / "GTgraph" / "sprng2.0-lite" / "include"),
        ]
        library_dirs: list[str] = []
        if conda_prefix:
            include_dirs.append(f"{conda_prefix}/include")
            library_dirs.append(f"{conda_prefix}/lib")

        include_dirs_literal = ", ".join(repr(path) for path in include_dirs)
        library_dirs_literal = ", ".join(repr(path) for path in library_dirs)

        with tempfile.TemporaryDirectory() as tmpdir:
            shim_dir = Path(tmpdir) / "simde_shim"
            _write_simde_shim(shim_dir)
            setup_py = f"""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='gustavson_spgemm_bench',
    ext_modules=[
        CppExtension(
            'gustavson_spgemm_bench',
            [{str(_CORE_CPP)!r}],
            include_dirs=[{include_dirs_literal}, {str(shim_dir)!r}],
            library_dirs=[{library_dirs_literal}],
            libraries=['tbb', 'tbbmalloc'],
            extra_compile_args=['-O3', '-std=c++17', '-fopenmp'],
            extra_link_args=['-fopenmp'],
        )
    ],
    cmdclass={{'build_ext': BuildExtension}},
)
"""
            setup_path = Path(tmpdir) / "setup.py"
            setup_path.write_text(setup_py, encoding="utf-8")
            subprocess.check_call(
                [sys.executable, str(setup_path), "install"],
                env=env,
            )

    def prepare(self, A_csr: sp.csr_matrix) -> object:
        import gustavson_spgemm_bench  # noqa: PLC0415

        num_rows, num_cols = A_csr.shape
        if num_rows != num_cols:
            raise RuntimeError(
                f"gustavson-spgemm requires a square CSR matrix; "
                f"got {num_rows}x{num_cols}"
            )

        # Respect existing OMP_NUM_THREADS; otherwise default to 64.
        os.environ.setdefault("OMP_NUM_THREADS", "64")

        return gustavson_spgemm_bench.prepare(
            A_csr.indptr.astype("int64"),
            A_csr.indices.astype("int64"),
            A_csr.data.astype("float64"),
            int(num_rows),
        )

    def run(
        self, prepared_A: object, B: torch.Tensor | None
    ) -> tuple[torch.Tensor, float]:
        import gustavson_spgemm_bench  # noqa: PLC0415

        _ = B
        kernel_ms = float(gustavson_spgemm_bench.run(prepared_A))
        return torch.empty(0, dtype=torch.float64, device="cpu"), kernel_ms
