"""Shared helpers for building CUDA extensions."""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def cuda_build_env() -> dict[str, str]:
    """Build env with nvcc matching the PyTorch CUDA runtime."""
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix and (Path(conda_prefix) / "bin" / "nvcc").exists():
        cuda_home = conda_prefix
    else:
        cuda_home = env.get("CUDA_HOME", "/usr/local/cuda")
    env["CUDA_HOME"] = cuda_home
    # Prefer conda nvcc over system /usr/local/cuda (may be a newer major).
    env["PATH"] = f"{cuda_home}/bin:" + env.get("PATH", "")
    env["TORCH_CUDA_ARCH_LIST"] = "9.0"

    if conda_prefix:
        sbsa_include = Path(conda_prefix) / "targets/sbsa-linux/include"
        sbsa_lib = Path(conda_prefix) / "targets/sbsa-linux/lib"
        if sbsa_include.is_dir():
            env["CPATH"] = _prepend_path(env.get("CPATH", ""), str(sbsa_include))
            env["CPLUS_INCLUDE_PATH"] = _prepend_path(
                env.get("CPLUS_INCLUDE_PATH", ""), str(sbsa_include)
            )
        if sbsa_lib.is_dir():
            env["LD_LIBRARY_PATH"] = _prepend_path(
                env.get("LD_LIBRARY_PATH", ""), str(sbsa_lib)
            )
            env["LIBRARY_PATH"] = _prepend_path(
                env.get("LIBRARY_PATH", ""), str(sbsa_lib)
            )
    return env


def _prepend_path(existing: str, entry: str) -> str:
    if not existing:
        return entry
    return f"{entry}:{existing}"


def extension_source_newer_than_binary(
    module: str,
    sources: Sequence[str | Path],
    env: dict[str, str] | None = None,
) -> bool:
    """Return True if any source file is newer than the installed extension."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {module} as m; print(m.__file__)"],
        capture_output=True,
        text=True,
        env=env or cuda_build_env(),
    )
    if result.returncode != 0:
        return True
    binary = Path(result.stdout.strip())
    if not binary.exists():
        return True
    binary_mtime = binary.stat().st_mtime
    for src in sources:
        path = Path(src)
        if path.exists() and path.stat().st_mtime > binary_mtime:
            return True
    return False


def extension_importable(module: str, env: dict[str, str] | None = None) -> bool:
    return (
        subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            env=env or cuda_build_env(),
        ).returncode
        == 0
    )


def extension_ready(
    module: str,
    attrs: Sequence[str],
    env: dict[str, str] | None = None,
) -> bool:
    """Return True if module imports and exposes all required attributes."""
    checks = "; ".join(f"getattr(m, {name!r})" for name in attrs)
    code = f"import {module} as m; {checks}"
    return (
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            env=env or cuda_build_env(),
        ).returncode
        == 0
    )
