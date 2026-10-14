from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_TC_H = 16
DEFAULT_TC_W = 8
DEFAULT_THREAD_COUNT = 32
DTC_GPU_ROW_THRESHOLD = 1_600_000

_METHODS_PATHS = (
    Path(__file__).resolve().parents[1] / "methods.json",
    Path.cwd() / "methods.json",
)


def _load_methods_config() -> dict[str, Any]:
    for path in _METHODS_PATHS:
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            break
    else:
        searched = ", ".join(str(path) for path in _METHODS_PATHS)
        raise FileNotFoundError(f"methods.json not found; searched: {searched}")
    if not isinstance(data, dict):
        raise TypeError("methods.json must contain a JSON object")
    return data


METHOD_CONFIG = _load_methods_config()
METHOD_NAMES = tuple(METHOD_CONFIG.keys())


def _params_schema(method: str) -> dict[str, dict[str, Any]]:
    spec = METHOD_CONFIG.get(method)
    if spec is None:
        raise ValueError(f"Unknown method: {method}")
    schema = spec.get("params_schema", {})
    if not isinstance(schema, dict):
        raise TypeError(f"{method}.params_schema must be a dict")
    return schema


def _env_schema(method: str) -> dict[str, dict[str, Any]]:
    spec = METHOD_CONFIG.get(method)
    if spec is None:
        raise ValueError(f"Unknown method: {method}")
    schema = spec.get("env_schema", {})
    if not isinstance(schema, dict):
        raise TypeError(f"{method}.env_schema must be a dict")
    return schema


def _coerce_param(method: str, name: str, value: Any, rule: dict[str, Any]) -> Any:
    kind = rule.get("type")
    try:
        if kind == "int":
            if isinstance(value, bool):
                raise ValueError
            return int(value)
        if kind == "float":
            if isinstance(value, bool):
                raise ValueError
            return float(value)
        if kind == "bool":
            if not isinstance(value, bool):
                raise ValueError
            return value
        if kind == "str":
            if not isinstance(value, str):
                raise ValueError
            return value
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{method}.{name} must be {kind}") from exc
    raise ValueError(f"Unsupported param type for {method}.{name}: {kind}")


def _default_params(method: str) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for name, rule in _params_schema(method).items():
        if not isinstance(rule, dict):
            raise TypeError(f"{method}.{name} schema must be a dict")
        if "default" in rule:
            defaults[name] = _coerce_param(method, name, rule["default"], rule)
    return defaults


METHOD_DEFAULT_PARAMS = {method: _default_params(method) for method in METHOD_NAMES}
GROOT_PY_FIXED = dict(METHOD_CONFIG.get("groot-py", {}).get("fixed_defaults", {}))
DTC_FIXED = dict(METHOD_CONFIG.get("dtc-tca", {}).get("fixed_defaults", {}))
CLASSIC_FIXED = {
    method: dict(METHOD_CONFIG.get(method, {}).get("fixed_defaults", {}))
    for method in (
        "degree",
        "gray",
        "slashburn",
        "gp",
        "random",
        "global-greedy",
    )
}

_GPU_SM_COUNT: int | None = None


def query_gpu_sm_count(device: int = 0) -> int:
    """Return CUDA multiProcessorCount (SM_NUM) for the given device."""
    global _GPU_SM_COUNT
    if _GPU_SM_COUNT is not None and device == 0:
        return _GPU_SM_COUNT
    try:
        import ctypes
        from ctypes import byref, c_int

        cudart = ctypes.CDLL("libcudart.so")
        cuda_dev_attr_multiprocessor_count = 16
        sm = c_int()
        err = cudart.cudaDeviceGetAttribute(
            byref(sm), cuda_dev_attr_multiprocessor_count, int(device)
        )
        if err != 0:
            raise RuntimeError(f"cudaDeviceGetAttribute failed with code {err}")
        value = int(sm.value)
    except OSError as exc:
        raise RuntimeError(
            "Failed to load libcudart.so while resolving dtc-tca c_thres=gpu_sm_count; "
            "pass c_thres explicitly via --params-json"
        ) from exc
    if device == 0:
        _GPU_SM_COUNT = value
    return value


def resolve_runtime_env() -> dict[str, Any]:
    """Effective OMP / N_JOBS; default 32 when env vars are unset."""
    omp_raw = os.environ.get("OMP_NUM_THREADS")
    n_jobs_raw = os.environ.get("N_JOBS")
    omp = int(omp_raw) if omp_raw is not None else DEFAULT_THREAD_COUNT
    n_jobs = int(n_jobs_raw) if n_jobs_raw is not None else omp
    return {"omp_threads": omp, "n_jobs": n_jobs}


def apply_runtime_env() -> dict[str, Any]:
    """Resolve thread settings and export OMP_NUM_THREADS for native OpenMP code."""
    runtime = resolve_runtime_env()
    os.environ.setdefault("OMP_NUM_THREADS", str(runtime["omp_threads"]))
    return runtime


def merge_params(method: str, user_params: dict[str, Any] | None) -> dict[str, Any]:
    if user_params is None:
        user_params = {}
    if not isinstance(user_params, dict):
        raise TypeError("params must be a dict")
    schema = _params_schema(method)
    unknown = sorted(set(user_params) - set(schema))
    if unknown:
        raise ValueError(f"Unknown params for {method}: {', '.join(unknown)}")
    defaults = METHOD_DEFAULT_PARAMS.get(method, {})
    merged = dict(defaults)
    for name, value in user_params.items():
        merged[name] = _coerce_param(method, name, value, schema[name])
    missing = [
        name
        for name, rule in schema.items()
        if name not in merged and not rule.get("optional", False)
    ]
    if missing:
        raise ValueError(f"Missing required params for {method}: {', '.join(missing)}")
    if method == "dtc-tca" and "c_thres" not in merged:
        merged["c_thres"] = query_gpu_sm_count()
    return merged


def resolve_groot_cpp_effective(params: dict[str, Any], nrows: int) -> dict[str, Any]:
    i_k = params.get("i_k")
    if i_k is None:
        i_k = min(nrows - 1, 20)
    i_l = params.get("i_l")
    if i_l is None:
        i_l = min(i_k + 50, 30)
    s_k = params.get("s_k")
    if s_k is None:
        s_k = min(nrows - 1, 25)
    iterations = params["iterations"]
    return {
        "i_k": int(i_k),
        "i_l": int(i_l),
        "s_k": int(s_k),
        "iterations": int(iterations),
        "nrows": int(nrows),
    }


def resolve_effective_params(method: str, params: dict[str, Any], nrows: int) -> dict[str, Any]:
    if method == "groot-cpp":
        return resolve_groot_cpp_effective(params, nrows)
    if method in METHOD_DEFAULT_PARAMS:
        out = dict(params)
        out["nrows"] = int(nrows)
        return out
    return {"nrows": int(nrows)}


def params_for_hash(method: str, params: dict[str, Any]) -> dict[str, Any]:
    """Only tunable params declared in methods.json enter the params hash."""
    schema = _params_schema(method)
    return {k: params[k] for k in schema if k in params}


def build_method_env(
    method: str,
    runtime: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env: dict[str, Any] = {}
    if method in METHOD_NAMES:
        env["omp_threads"] = runtime["omp_threads"]
    if method == "groot-py":
        env["n_jobs"] = runtime["n_jobs"]
    if extra:
        env.update(extra)
    return env


def env_for_hash(
    method: str,
    runtime: dict[str, Any],
    extra: dict[str, Any] | None = None,
    *,
    nrows: int | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Env keys from env_schema that participate in params_hash."""
    merged_extra = dict(extra or {})
    if method == "dtc-tca" and "dtc_minhash_backend" not in merged_extra:
        if nrows is None:
            raise ValueError("nrows is required to resolve dtc env_for_hash")
        from reorder_bench.methods.dtc_tca import _resolve_minhash_backend

        merged_extra["dtc_minhash_backend"] = _resolve_minhash_backend(
            nrows, params or {}
        )
    full_env = build_method_env(method, runtime, merged_extra)
    schema = _env_schema(method)
    return {k: full_env[k] for k in schema if k in full_env}


def method_env_json(method: str, runtime: dict[str, Any], extra: dict[str, Any] | None = None) -> str:
    env = build_method_env(method, runtime, extra)
    return json.dumps(env, sort_keys=True, separators=(",", ":"))


def sync_method_schemas_to_spmatkit(
    methods: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Mirror ``methods.json`` env-schema defaults into spmatkit registry.

    Param defaults are resolved from ``methods.json`` directly by spmatkit;
    this only mirrors env defaults (e.g. omp_threads=32) into
    ``reorder_methods.json`` for display / registry-only tooling.
    """
    import spmatkit

    names = list(methods) if methods is not None else list(METHOD_NAMES)
    for method in names:
        if method not in METHOD_CONFIG:
            continue

        env_defaults: dict[str, Any] = {}
        for key, rule in _env_schema(method).items():
            if isinstance(rule, dict) and "default" in rule:
                env_defaults[key] = _coerce_param(method, key, rule["default"], rule)
        # Methods whose env_schema lists omp_threads/n_jobs without an
        # explicit default still hash with DEFAULT_THREAD_COUNT at save time.
        for key in _env_schema(method):
            if key not in env_defaults and key in ("omp_threads", "n_jobs"):
                env_defaults[key] = DEFAULT_THREAD_COUNT
        if env_defaults:
            spmatkit.set_reorder_method_env_defaults(method, env_defaults)
