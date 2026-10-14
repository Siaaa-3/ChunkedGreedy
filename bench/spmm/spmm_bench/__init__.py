"""SpMM benchmark suite."""
from .api import run_spmm_method
from .batch_bench import run_batch_benchmark
from .utils import generate_dense

__all__ = [
    "run_spmm_method",
    "run_batch_benchmark",
    "generate_dense",
]
