from __future__ import annotations

from typing import Any

import scipy.sparse

from reorder_bench.config import CLASSIC_FIXED
from reorder_bench.methods._classic_common import reorder_sparsebase
from reorder_bench.types import ReorderResult


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    del params
    fixed = CLASSIC_FIXED["slashburn"]
    nrows = csr.shape[0]
    k_size = max(1, int(float(fixed["hubset_fraction"]) * nrows))
    greedy = bool(fixed["greedy"])
    hub_order = bool(fixed["hub_order"])
    effective = {
        "k_size": k_size,
        "greedy": greedy,
        "hub_order": hub_order,
    }

    def _run(ext, row_ptr, col, n, m):
        return ext.slashburn_reorder(
            row_ptr, col, n, m, k_size, greedy, hub_order
        )

    return reorder_sparsebase(
        csr,
        method="slashburn",
        reorder_fn=_run,
        prep_ms=0.0,
        effective_params=effective,
    )
