from __future__ import annotations

from typing import Any

import scipy.sparse

from reorder_bench.config import CLASSIC_FIXED
from reorder_bench.methods._classic_common import (
    csr_to_symmetric_csr_arrays,
    partition_to_row_perm,
)
from reorder_bench.timing import timed_ms
from reorder_bench.types import ReorderResult


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    del params
    try:
        import pymetis
    except ImportError as exc:
        raise ImportError(
            "gp requires pymetis: pip install pymetis"
        ) from exc

    fixed = CLASSIC_FIXED["gp"]
    k = int(fixed["k"])
    objective = str(fixed["objective"])
    if objective != "edge-cut":
        raise ValueError(f"gp only supports objective=edge-cut, got {objective!r}")

    xadj, adjncy, prep_ms = csr_to_symmetric_csr_arrays(csr)

    with timed_ms() as bucket:
        part = pymetis.part_graph(k, xadj=xadj, adjncy=adjncy)
        row_perm = partition_to_row_perm(part.vertex_part)
    reorder_ms = bucket[0]

    return ReorderResult(
        row_perm=row_perm,
        reorder_time_ms=reorder_ms,
        prep_time_ms=prep_ms,
        perm_trans_ms=0.0,
        effective_params={"k": k, "objective": objective},
    )
