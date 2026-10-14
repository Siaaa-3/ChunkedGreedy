from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

from reorder_bench.config import CLASSIC_FIXED
from reorder_bench.timing import timed_ms
from reorder_bench.types import ReorderResult


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    del params
    seed = int(CLASSIC_FIXED["random"]["seed"])
    nrows = csr.shape[0]

    with timed_ms() as bucket:
        rng = np.random.default_rng(seed)
        row_perm = rng.permutation(nrows).astype(np.int64)
    reorder_ms = bucket[0]

    return ReorderResult(
        row_perm=row_perm,
        reorder_time_ms=reorder_ms,
        effective_params={"seed": seed},
    )
