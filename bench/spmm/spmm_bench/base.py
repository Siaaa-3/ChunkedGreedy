"""Base interface for SpMM methods."""
from abc import ABC, abstractmethod
from typing import Any

import torch


class SpMMMethod(ABC):
    """Abstract base class for SpMM kernel wrappers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the method name."""

    @property
    def uses_feature_dim(self) -> bool:
        """Whether this method benchmarks over dense feature dimensions."""
        return True

    @property
    def uses_cuda(self) -> bool:
        """Whether prepare/run require a CUDA device and cudaSynchronize."""
        return True

    @property
    @abstractmethod
    def tc_h(self) -> int:
        """Sparse tensor-core tile height used by this method."""

    @property
    @abstractmethod
    def tc_w(self) -> int:
        """Sparse tensor-core tile width used by this method."""

    @abstractmethod
    def build(self) -> None:
        """Compile/install the method. Raise RuntimeError on failure."""

    @abstractmethod
    def prepare(self, A_csr: Any) -> Any:
        """Convert scipy CSR to the method's internal format.

        This step is not timed and may be expensive.
        """

    @abstractmethod
    def run(
        self, prepared_A: Any, B: torch.Tensor | None
    ) -> tuple[torch.Tensor, float]:
        """Run a single kernel launch.

        Must perform exactly one timed kernel launch (no internal warm up or
        repeated execution). Returns (output, kernel_ms) where kernel_ms is
        measured inside the method with cudaEvent/GpuTimer around the kernel
        only. The caller handles warm up and averages kernel_ms over iterations.

        SpGEMM methods may ignore ``B`` (pass ``None`` from the batch runner).
        """

    def check_dense_limits(self, num_nodes: int, feature_dim: int) -> None:
        """Raise if dense shape exceeds this method's kernel indexing limits."""
        from .utils import check_method_dense_limits  # noqa: PLC0415

        check_method_dense_limits(self.name, num_nodes, feature_dim)

    @classmethod
    def create(cls, name: str) -> "SpMMMethod":
        """Factory: create a wrapper by registered name."""
        import importlib  # noqa: PLC0415

        registry = {
            "acc-spmm": ("acc_spmm", "AccSpMM"),
            "acc-spmm-balance": ("acc_spmm", "AccSpMMBalance"),
            "cusparse-spmm": ("cusparse_spmm", "CuSparseSpMM"),
            "cusparse-spgemm": ("cusparse_spgemm", "CuSparseSpGEMM"),
            "dtc-spmm": ("dtc_spmm", "DTCSpMM"),
            "dtc-spmm-balance": ("dtc_spmm", "DTCSpMMBalance"),
            "flashsparse": ("flash_sparse", "FlashSparse"),
            "gustavson-spgemm": ("gustavson_spgemm", "GustavsonSpGEMM"),
            "voltrix": ("voltrix_spmm", "VoltrixSpMM"),
        }
        if name not in registry:
            raise ValueError(f"Unknown method: {name}. Available: {list(registry)}")
        module, cls_name = registry[name]
        mod = importlib.import_module(f"{__package__}.wrappers.{module}")
        return getattr(mod, cls_name)()
