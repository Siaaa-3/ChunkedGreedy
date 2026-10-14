"""High-level API for running a single SpMM method."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import scipy.sparse as sp
import torch

from .base import SpMMMethod
from .utils import estimate_tc_block_count, generate_dense


@dataclass(frozen=True)
class SpmmKernelResult:
    time_ms: float
    method: str
    feature_dim: int
    tc_h: int
    tc_w: int


def get_method_tc_tile(method_name: str) -> tuple[int, int]:
    """Return the sparse tensor-core tile shape used by a SpMM method."""
    wrapper = SpMMMethod.create(method_name)
    return wrapper.tc_h, wrapper.tc_w


def tc_num_for_csr(csr: sp.csr_matrix, tc_h: int, tc_w: int) -> int:
    """Count TC blocks for a CSR matrix under the given tile shape."""
    return int(estimate_tc_block_count(csr, tc_h, tc_w))


def load_origin_spmm_times(
    csv_path: str | Path,
    *,
    method: str,
    feature_dim: int,
    reorder_method: str = "ORIGIN",
) -> dict[str, float]:
    """Load per-matrix kernel ``time_ms`` from a batch-bench CSV.

    Keys are matrix names. By default only ORIGIN rows are returned.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {path}")

    times: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("success") != "1":
                continue
            if row["spmm_method"] != method:
                continue
            if reorder_method is not None and row.get("reorder_method") != reorder_method:
                continue
            if int(row["feature_dim"]) != int(feature_dim):
                continue
            times[row["matrix"]] = float(row["time_ms"])
    return times


def bench_spmm_kernel(
    method_name: str,
    A_csr: Any,
    *,
    feature_dim: int,
    warmup: int = 10,
    iterations: int = 30,
    seed: int = 42,
) -> SpmmKernelResult:
    """Benchmark kernel-only SpMM time on a CSR matrix (prepare is not timed)."""
    method = SpMMMethod.create(method_name)
    method.build()
    tc_h, tc_w = method.tc_h, method.tc_w

    num_nodes = A_csr.shape[0]
    method.check_dense_limits(num_nodes, feature_dim)
    B = generate_dense(num_nodes, feature_dim, seed=seed)
    prepared = method.prepare(A_csr)

    for _ in range(warmup):
        method.run(prepared, B)

    total_ms = 0.0
    for _ in range(iterations):
        _, kernel_ms = method.run(prepared, B)
        total_ms += kernel_ms

    return SpmmKernelResult(
        time_ms=total_ms / iterations,
        method=method_name,
        feature_dim=feature_dim,
        tc_h=tc_h,
        tc_w=tc_w,
    )


def run_spmm_method(
  method_name: str,
  A_csr: Any,
  B: torch.Tensor,
  warmup: int = 10,
  iterations: int = 10,
) -> dict:
  """Run one SpMM method with uniform warm up and timing.

  Args:
      method_name: Registered method name.
      A_csr: scipy.sparse.csr_matrix.
      B: Dense matrix, shape (num_nodes, feature_dim).
      warmup: Number of untimed warm-up launches.
      iterations: Number of timed launches.

  Returns:
      Dict with keys: output, time_ms, method, feature_dim.
  """
  method = SpMMMethod.create(method_name)
  method.build()

  prepared = method.prepare(A_csr)

  output = None
  for _ in range(warmup):
    method.run(prepared, B)

  total_ms = 0.0
  for _ in range(iterations):
    output, kernel_ms = method.run(prepared, B)
    total_ms += kernel_ms

  return {
    "output": output,
    "time_ms": total_ms / iterations,
    "method": method_name,
    "feature_dim": B.size(1),
  }
