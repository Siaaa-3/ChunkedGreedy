from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ReorderResult:
    row_perm: np.ndarray
    reorder_time_ms: float
    prep_time_ms: float = 0.0
    perm_trans_ms: float = 0.0
    timing_breakdown: dict[str, float] = field(default_factory=dict)
    effective_params: dict[str, Any] = field(default_factory=dict)
    env_extra: dict[str, Any] = field(default_factory=dict)
