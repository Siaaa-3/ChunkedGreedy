from __future__ import annotations

from typing import Any

import scipy.sparse

from reorder_bench.methods._classic_common import reorder_sparsebase
from reorder_bench.types import ReorderResult


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    del params

    def _run(ext, row_ptr, col, n, m):
        return ext.rcm_reorder(row_ptr, col, n, m)

    return reorder_sparsebase(csr, method="rcm", reorder_fn=_run, prep_ms=0.0)
