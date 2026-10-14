from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timed_ms() -> Iterator[list[float]]:
    bucket: list[float] = [0.0]
    t0 = time.perf_counter()
    try:
        yield bucket
    finally:
        bucket[0] = (time.perf_counter() - t0) * 1000.0


def sync_gpu_if_needed(used_gpu: bool, device_id: int = 0) -> None:
    if not used_gpu:
        return
    try:
        import cupy

        cupy.cuda.Device(device_id).synchronize()
    except ImportError:
        pass


def flush_l2(device_id: int = 0) -> None:
    """Evict GPU L2 by writing ~2x L2-sized buffer (sporder bench protocol)."""
    import cupy

    with cupy.cuda.Device(device_id):
        try:
            l2 = cupy.cuda.runtime.getDeviceProperties(device_id).get(
                "l2CacheSize", 6 * 1024 * 1024
            )
        except Exception:
            l2 = 6 * 1024 * 1024
        buf = cupy.empty(int(l2) * 2, dtype=cupy.uint8)
        buf[...] = 0
        cupy.cuda.get_current_stream().synchronize()
