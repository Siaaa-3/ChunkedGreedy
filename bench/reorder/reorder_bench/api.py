from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

from reorder_bench.config import (
  METHOD_NAMES,
  apply_runtime_env,
  env_for_hash,
  merge_params,
  params_for_hash,
  resolve_runtime_env,
)
from reorder_bench.metrics import compute_metrics
from reorder_bench.run_bench import METHOD_MODULES, _register_methods
from reorder_bench.types import ReorderResult


def apply_row_perm(csr: scipy.sparse.csr_matrix, row_perm: np.ndarray) -> scipy.sparse.csr_matrix:
  """Return vertex-renumbered ``csr[perm, :][:, perm]`` (P A P^T) as CSR.

  ``row_perm[new_i]`` is the original index at new index ``new_i``. Requires a
  square matrix (same convention as ``spmatkit`` reorder load).
  """
  if not scipy.sparse.isspmatrix_csr(csr):
    csr = csr.tocsr()
  nrows, ncols = csr.shape
  if nrows != ncols:
    raise ValueError(
      f"apply_row_perm requires a square matrix, got shape {csr.shape}"
    )
  perm = np.asarray(row_perm, dtype=np.int64)
  if perm.shape[0] != nrows:
    raise ValueError(
      f"row_perm length {perm.shape[0]} != matrix dimension {nrows}"
    )
  return csr[perm, :][:, perm].tocsr()


def reorder_once(
  method: str,
  csr: scipy.sparse.csr_matrix,
  params: dict[str, Any] | None = None,
) -> ReorderResult:
  """Run a single reorder method on an in-memory CSR matrix."""
  _register_methods()
  if method not in METHOD_NAMES:
    raise ValueError(f"Unknown method: {method}")
  merged = merge_params(method, params)
  return METHOD_MODULES[method](csr, merged)


def structure_metrics(
  csr: scipy.sparse.csr_matrix,
  row_perm: np.ndarray | None = None,
  *,
  tc_h: int,
  tc_w: int,
) -> dict[str, Any]:
  """Compute structure metrics (including ``tc_num``) for optional vertex reorder."""
  metrics, _eval_ms = compute_metrics(
    csr,
    row_perm,
    tc_h=tc_h,
    tc_w=tc_w,
    metrics=["tc_h", "tc_w", "tc_num"],
  )
  return metrics


def trial_variant_hash(
  method: str,
  params: dict[str, Any] | None,
  *,
  nrows: int,
  env_extra: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
  """Return (params_hash, hash_params, hash_env) for a tuning trial."""
  import spmatkit

  apply_runtime_env()
  runtime = resolve_runtime_env()
  merged = merge_params(method, params)
  hash_params = params_for_hash(method, merged)
  hash_env = env_for_hash(method, runtime, env_extra, nrows=nrows, params=merged)
  params_hash = spmatkit.compute_reorder_hash(method, hash_params, hash_env)
  return params_hash, hash_params, hash_env
