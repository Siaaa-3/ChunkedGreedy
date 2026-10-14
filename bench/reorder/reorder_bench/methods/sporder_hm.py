from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse

from reorder_bench.loader import prep_sporder_gpu, row_perm_from_order
from reorder_bench.timing import flush_l2, timed_ms
from reorder_bench.types import ReorderResult


def _import_sporder():
    try:
        import sporder
    except ImportError as exc:
        raise ImportError(
            "sporder-hm requires the sporder package. "
            "Install with: pip install -e Reordering_methods/sporder "
            "(needs CUDA + CuPy)"
        ) from exc
    return sporder


def reorder(csr: scipy.sparse.csr_matrix, params: dict[str, Any]) -> ReorderResult:
    sporder = _import_sporder()
    auto_detect = bool(params.get("auto_detect_symmetric", True))
    ptr, ind, is_symmetric, prep_ms = prep_sporder_gpu(
        csr, detect_symmetric=auto_detect
    )
    nrows = csr.shape[0]

    call_kwargs = {
        "is_symmetric": is_symmetric,
        "num_levels": int(params["num_levels"]),
        "k": int(params["k"]),
        "oversampling": float(params["oversampling"]),
        "with_stats": False,
    }

    sporder.hierarchical_matching(
        ptr, ind, with_timing=False, **call_kwargs
    )
    flush_l2()

    order, timing = sporder.hierarchical_matching(
        ptr, ind, with_timing=True, **call_kwargs
    )

    breakdown = {
        "csr2csc_ms": float(timing.get("csr2csc_ms", 0.0)),
        "similarity_ms": float(timing.get("similarity_ms", 0.0)),
        "matching_ms": float(timing.get("matching_ms", 0.0)),
        "advance_ms": float(timing.get("advance_ms", 0.0)),
    }
    reorder_time_ms = float(timing["wall_ms"])

    with timed_ms() as bucket:
        row_perm = row_perm_from_order(np.asarray(order.get(), dtype=np.int64), nrows)
    perm_trans_ms = bucket[0]

    return ReorderResult(
        row_perm=row_perm,
        reorder_time_ms=reorder_time_ms,
        prep_time_ms=prep_ms,
        perm_trans_ms=perm_trans_ms,
        timing_breakdown=breakdown,
        effective_params={"is_symmetric": is_symmetric},
    )
