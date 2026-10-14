from __future__ import annotations

from typing import Any

import scipy.sparse

from reorder_bench.config import CLASSIC_FIXED
from reorder_bench.methods._classic_common import reorder_sparsebase
from reorder_bench.types import ReorderResult


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    del params
    ascending = bool(CLASSIC_FIXED["degree"]["ascending"])

    def _run(ext, row_ptr, col, n, m):
        return ext.degree_reorder(row_ptr, col, n, m, ascending)

    return reorder_sparsebase(
        csr,
        method="degree",
        reorder_fn=_run,
        prep_ms=0.0,
        effective_params={"ascending": ascending},
    )
