from __future__ import annotations

from typing import Any

import scipy.sparse

from reorder_bench.config import CLASSIC_FIXED
from reorder_bench.methods._classic_common import reorder_sparsebase
from reorder_bench.types import ReorderResult


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    del params
    fixed = CLASSIC_FIXED["gray"]
    resolution = int(fixed["resolution"])
    nnz_threshold = int(fixed["nnz_threshold"])
    nrows, _ = csr.shape
    nnz = csr.nnz
    avg_nnz = max(nnz // max(nrows, 1), 1)
    group_size = max(avg_nnz // max(resolution, 1), 1)
    effective = {
        "resolution": resolution,
        "nnz_threshold": nnz_threshold,
        "sparse_density_group_size": group_size,
    }

    def _run(ext, row_ptr, col, n, m):
        return ext.gray_reorder(
            row_ptr, col, n, m, resolution, nnz_threshold, group_size
        )

    return reorder_sparsebase(
        csr,
        method="gray",
        reorder_fn=_run,
        prep_ms=0.0,
        effective_params=effective,
    )
