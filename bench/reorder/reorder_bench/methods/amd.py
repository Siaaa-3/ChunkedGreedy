from __future__ import annotations

from typing import Any

import scipy.sparse

from reorder_bench.methods._classic_common import reorder_sparsebase
from reorder_bench.types import ReorderResult


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    del params

    def _run(ext, row_ptr, col, n, m):
        if not hasattr(ext, "amd_reorder"):
            raise ImportError(
                "amd requires sparsebase_reorder_core built with USE_AMD_ORDER. "
                "Install SuiteSparse (libamd) and rerun: ./scripts/build_native.sh sparsebase"
            )
        return ext.amd_reorder(row_ptr, col, n, m)

    return reorder_sparsebase(csr, method="amd", reorder_fn=_run, prep_ms=0.0)
