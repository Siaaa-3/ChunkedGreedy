from __future__ import annotations

import time
from typing import Iterable

import scipy.sparse

import spmatkit

from reorder_bench.config import DEFAULT_TC_H, DEFAULT_TC_W

DEFAULT_METRICS = (
    "bandwidth",
    "profile",
    "tc_h",
    "tc_w",
    "tc_num",
    "tc_fill_rate",
)


def compute_metrics(
    csr: scipy.sparse.csr_matrix,
    row_perm,
    *,
    tc_h: int = DEFAULT_TC_H,
    tc_w: int = DEFAULT_TC_W,
    metrics: Iterable[str] | None = None,
) -> tuple[dict, float]:
    names = list(metrics) if metrics is not None else list(DEFAULT_METRICS)
    t0 = time.perf_counter()
    result = spmatkit.compute_structure_metrics(
        csr,
        row_perm=row_perm,
        metrics=names,
        tc_h=tc_h,
        tc_w=tc_w,
    )
    ms = (time.perf_counter() - t0) * 1000.0
    return result, ms
