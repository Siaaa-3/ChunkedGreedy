from __future__ import annotations

import gc


def _needs_device_reset(msg: str, sticky_from_runtime: bool) -> bool:
    """Only poisoned CUDA contexts need cudaDeviceReset.

    Soft failures (OOM / cugraph business errors) should free pools only.
    Matching library names like "cugraph"/"cudf" is too broad and used to
    force a hard reset on almost every DTC failure.
    """
    if sticky_from_runtime:
        return True
    lowered = msg.lower()
    sticky_signals = (
        "illegal memory access",
        "illegaladdress",
        "cudaerrorillegaladdress",
        "device-side assert",
        "uncorrectable",
    )
    return any(signal in lowered for signal in sticky_signals)


def _free_cupy_pools(*, synchronize: bool) -> None:
    try:
        import cupy

        cupy.cuda.Device(0).use()
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        if synchronize:
            cupy.cuda.Device(0).synchronize()
    except Exception:
        pass


def _reinit_rmm() -> None:
    try:
        import rmm

        rmm.reinitialize()
    except Exception:
        pass


def _clear_torch_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    except Exception:
        pass


def recover_gpu_after_error(exc: BaseException | None = None) -> None:
    """Best-effort GPU recovery after cugraph/libMHCUDA/cupy failures.

    Soft path (OOM etc.): free CuPy/RMM pools without destroying the context.
    Hard path (illegal address / sticky CUDA error): cudaDeviceReset first,
    then rebuild RMM/CuPy/Torch on the fresh context. Never reinitialize RMM
    before a device reset — that leaves dangling allocator handles and causes
    cascading "No such CUDA device" failures on later matrices.
    """
    err_msg = str(exc).lower() if exc is not None else ""

    sticky_error = False
    libcudart = None
    try:
        import ctypes

        libcudart = ctypes.CDLL("libcudart.so")
        # Clears the sticky flag for subsequent CUDA calls.
        sticky_error = int(libcudart.cudaGetLastError()) != 0
    except OSError:
        pass

    if _needs_device_reset(err_msg, sticky_error) and libcudart is not None:
        # Drop Python-side GPU buffers before destroying the context.
        gc.collect()
        try:
            libcudart.cudaDeviceReset()
        except Exception:
            pass
        # Rebuild allocator stacks on the new context (order matters).
        _reinit_rmm()
        _free_cupy_pools(synchronize=False)
        _clear_torch_cache()
        gc.collect()
        return

    # Soft recovery: release pooled memory without destroying the context.
    _free_cupy_pools(synchronize=True)
    _reinit_rmm()
    gc.collect()
    _clear_torch_cache()
