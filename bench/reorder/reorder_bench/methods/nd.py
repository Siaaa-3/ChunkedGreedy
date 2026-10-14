from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

from reorder_bench.methods._classic_common import csr_to_symmetric_csr_arrays
from reorder_bench.timing import timed_ms
from reorder_bench.types import ReorderResult


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    del params
    try:
        import pymetis
    except ImportError as exc:
        raise ImportError(
            "nd requires pymetis: pip install pymetis"
        ) from exc

    xadj, adjncy, prep_ms = csr_to_symmetric_csr_arrays(csr)

    with timed_ms() as bucket:
        _perm, iperm = pymetis.nested_dissection(xadj=xadj, adjncy=adjncy)
        row_perm = np.asarray(iperm, dtype=np.int64)
    reorder_ms = bucket[0]

    return ReorderResult(
        row_perm=row_perm,
        reorder_time_ms=reorder_ms,
        prep_time_ms=prep_ms,
        perm_trans_ms=0.0,
    )
