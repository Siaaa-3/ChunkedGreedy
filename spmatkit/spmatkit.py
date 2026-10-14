"""
spmatkit

Sparse matrix storage, reordering, and structure metrics for SpMM benchmarks.

Every matrix (original or reordered) is uniquely identified by the key
``(matrix, reorder_method, reorder_hash)``:

* Original matrices use ``reorder_method="ORIGIN"`` and a fixed ``reorder_hash``.
* Reordered matrices use the reorder method name and a hash derived from
  ``(reorder_method, params, env)``.

Converted CSR matrices are stored under ``matrices/npz/`` as two files per
matrix:

* ``<matrix>.pattern.npz`` – shape, indices, indptr
* ``<matrix>.data.npz``     – nonzero values (``data`` array)

All statistics are kept in four files under ``matrices/meta_data/``:

* ``catalog.csv``                    – every stored matrix + basic info
* ``structure_metrics.csv``          – structural metrics per (key, tc_h, tc_w)
* ``merged_structure_spmm_data.csv`` – structure metrics joined with SpMM timings
* ``reorder_methods.json``           – stored param_sets (hash inventory) +
  fallback schemas for methods not listed in reordering-bench

Default param/env schemas (what ``all_param_sets=False`` resolves to) come from
the sibling ``reordering-bench/methods.json`` when present; ``param_sets`` and
catalog rows still live under ``meta_data/``.

Usage:
    import spmatkit

    mat = spmatkit.load_matrix("example")                       # original
    mat = spmatkit.load_matrix("example", reorder_method="acc") # reordered

    for matrix, rmethod, rhash, csr in spmatkit.iter_dataset("testDataset"):
        process(matrix, csr)
"""

import csv
import hashlib
import json
import os
import threading
import queue
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy.sparse

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None

PATTERN_SUFFIX = ".pattern.npz"
DATA_SUFFIX = ".data.npz"
PERM_SUFFIX = ".perm.npz"
LOAD_MODES = ("pattern", "full")

ORIGIN_METHOD = "ORIGIN"

CATALOG_FILENAME = "catalog.csv"
STRUCTURE_METRICS_FILENAME = "structure_metrics.csv"
MERGED_DATA_FILENAME = "merged_structure_spmm_data.csv"
REORDER_METHODS_FILENAME = "reorder_methods.json"

# Default SpMM batch-bench CSV (sibling repo under projects/).
DEFAULT_SPMM_BENCH_CSV = (
    Path(__file__).resolve().parent.parent
    / "bench" / "spmm"
    / "results"
    / "spmm_bench.csv"
)

# Authoritative tunable defaults for reorder variants (sibling repo).
DEFAULT_REORDER_BENCH_METHODS_JSON = (
    Path(__file__).resolve().parent.parent
    / "bench" / "reorder"
    / "methods.json"
)

# Env keys that reordering-bench hashes with a fixed default when schema
# omits ``default`` (see reorder_bench.config.sync_method_schemas_to_spmatkit).
_DEFAULT_THREAD_COUNT = 32
_DEFAULT_DTC_MINHASH_BACKEND = "cpu"

CATALOG_COLUMNS = (
    "matrix",
    "reorder_method",
    "reorder_hash",
    "params_json",
    "env_json",
    "nnz",
    "rows",
    "reorder_time_ms",
    "created_at",
    "source",
)

# ---------------------------------------------------------------------------
# Global data paths
# ---------------------------------------------------------------------------
_data_path = None
_npz_path = None
_datasets_path = None
_meta_data_path = None
_catalog_path = None
_reorder_dir = None
_reorder_methods_path = None
_reorder_perms_dir = None
_load_mode = None
_reorder_bench_methods_path = None
_bench_methods_cache = None  # (path_str, mtime_ns, data) | None


def _init_default_data_path():
    global _data_path, _npz_path, _datasets_path, _load_mode
    global _reorder_bench_methods_path
    if _data_path is None:
        try:
            module_dir = Path(__file__).resolve().parent
        except NameError:
            module_dir = Path.cwd()
        env_path = os.environ.get("SPMATKIT_DATA", "").strip()
        _data_path = Path(env_path).expanduser().resolve() if env_path else module_dir / "matrices"
        _npz_path = _data_path / "npz"
        _datasets_path = _data_path / "datasets"
        _load_mode = "full"
        _set_derived_paths(_data_path)
    if _reorder_bench_methods_path is None:
        _reorder_bench_methods_path = DEFAULT_REORDER_BENCH_METHODS_JSON


def _set_derived_paths(data_path):
    global _meta_data_path, _catalog_path
    global _reorder_dir, _reorder_methods_path, _reorder_perms_dir
    data_path = Path(data_path)
    _meta_data_path = data_path / "meta_data"
    _catalog_path = _meta_data_path / CATALOG_FILENAME
    _reorder_methods_path = _meta_data_path / REORDER_METHODS_FILENAME
    _reorder_dir = data_path / "reorder"
    _reorder_perms_dir = _reorder_dir / "perms"


def set_data_path(new_path):
    """Change the root data directory (expects ``npz/`` and ``datasets/``)."""
    global _data_path, _npz_path, _datasets_path
    _data_path = Path(new_path)
    _npz_path = _data_path / "npz"
    _datasets_path = _data_path / "datasets"
    _set_derived_paths(_data_path)


def set_load_mode(mode):
    """
    Set the default matrix loading mode.

    ``pattern`` loads the sparsity structure only (values filled with 1.0);
    ``full`` loads the pattern and stored nonzero values.
    """
    global _load_mode
    if mode not in LOAD_MODES:
        raise ValueError(f"Unsupported load mode '{mode}'. Choose from {LOAD_MODES}.")
    _load_mode = mode


def get_load_mode():
    """Return the active default load mode."""
    return _load_mode


def get_npz_dir():
    """Return the directory that holds converted ``.npz`` files."""
    return _npz_path


def get_metadata_dir():
    """Return ``matrices/meta_data``."""
    return _meta_data_path


def get_catalog_path():
    """Return ``matrices/meta_data/catalog.csv``."""
    return _catalog_path


def get_structure_metrics_path():
    """Return ``matrices/meta_data/structure_metrics.csv``."""
    return _meta_data_path / STRUCTURE_METRICS_FILENAME


def get_merged_data_path():
    """Return ``matrices/meta_data/merged_structure_spmm_data.csv``."""
    return _meta_data_path / MERGED_DATA_FILENAME


def get_default_spmm_bench_csv():
    """Return the default SpMM bench CSV (``spmm-bench/results/spmm_bench.csv``)."""
    return DEFAULT_SPMM_BENCH_CSV


def get_reorder_methods_path():
    """Return ``matrices/meta_data/reorder_methods.json``."""
    return _reorder_methods_path


def get_reorder_bench_methods_path():
    """Return path to ``reordering-bench/methods.json`` (schema defaults source)."""
    _init_default_data_path()
    return _reorder_bench_methods_path


def set_reorder_bench_methods_path(path):
    """Override the ``methods.json`` used for default param/env schemas."""
    global _reorder_bench_methods_path, _bench_methods_cache
    _reorder_bench_methods_path = Path(path) if path is not None else None
    _bench_methods_cache = None


def get_reorder_dir():
    """Return ``matrices/reorder``."""
    return _reorder_dir


def get_reorder_perms_dir():
    """Return ``matrices/reorder/perms``."""
    return _reorder_perms_dir


def get_pattern_path(name, npz_dir=None):
    """Path to the pattern file for a matrix name."""
    return Path(npz_dir or _npz_path) / f"{name}{PATTERN_SUFFIX}"


def get_data_path(name, npz_dir=None):
    """Path to the data file for a matrix name."""
    return Path(npz_dir or _npz_path) / f"{name}{DATA_SUFFIX}"


_init_default_data_path()


# ---------------------------------------------------------------------------
# NPZ helpers – force eager reads (np.load is lazy / mmap-backed)
# ---------------------------------------------------------------------------
def _materialize_array(arr):
    """Copy array contents into memory and touch the underlying buffer."""
    out = np.asarray(arr)
    _ = out.data
    return out


def _read_npz_arrays(path, keys):
    with np.load(path) as archive:
        return {key: _materialize_array(archive[key]) for key in keys}


def _utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Params and hashing
# ---------------------------------------------------------------------------
def _normalize_param_value(value, spec=None):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if value == int(value):
            return int(value)
        return float(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"Unsupported param value type: {type(value)!r}")


def canonicalize_params(params):
    """Return canonical JSON for a params dict."""
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")
    normalized = {}
    for key in sorted(params):
        normalized[key] = _normalize_param_value(params[key])
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def compute_reorder_hash(reorder_method, params, env=None):
    """
    Hash from (reorder_method, params[, env]); used in filenames and catalog.

    ``env`` keys (e.g. omp_threads) participate in the hash when non-empty.
    """
    if env is None:
        env = {}
    if not isinstance(env, dict):
        raise TypeError("env must be a dict")
    params_obj = json.loads(canonicalize_params(params))
    env_obj = json.loads(canonicalize_params(env)) if env else {}
    payload_dict = {"method": reorder_method, "params": params_obj}
    if env_obj:
        payload_dict["env"] = env_obj
    payload = json.dumps(payload_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


ORIGIN_HASH = compute_reorder_hash(ORIGIN_METHOD, {}, {})


def lookup_reorder_method_for_hash(reorder_hash):
    """Return the reorder method name that owns ``reorder_hash``.

    Looks up ``reorder_methods.json`` ``param_sets`` first, then ``catalog.csv``.
    Raises ``ValueError`` if the hash is unknown or maps to multiple methods.
    """
    if not isinstance(reorder_hash, str) or not reorder_hash:
        raise ValueError(f"reorder_hash must be a non-empty str, got {reorder_hash!r}")
    if reorder_hash == ORIGIN_HASH:
        return ORIGIN_METHOD

    methods = set()
    registry = _read_methods_registry()
    for method, entry in registry.items():
        param_sets = entry.get("param_sets") or {}
        if reorder_hash in param_sets:
            methods.add(method)

    if not methods:
        for entry in list_catalog_entries(all_param_sets=True):
            if entry.get("reorder_hash") == reorder_hash:
                methods.add(entry["reorder_method"])

    if not methods:
        raise ValueError(
            f"Unknown reorder_hash {reorder_hash!r}: not found in "
            f"reorder_methods.json param_sets or catalog.csv"
        )
    if len(methods) > 1:
        raise ValueError(
            f"reorder_hash {reorder_hash!r} maps to multiple methods: "
            f"{sorted(methods)}"
        )
    return next(iter(methods))


def resolve_reorder_method_and_hash(reorder_method=None, reorder_hash=None):
    """Validate / complete ``(reorder_method, reorder_hash)``.

    * Hash only (``reorder_method`` is ``None``): look up method for the hash.
    * Method + hash: require the hash to belong to that method, else raise.
    * Method only / neither: return ``(reorder_method or ORIGIN, reorder_hash)``.

    Returns ``(reorder_method, reorder_hash)``.
    """
    if reorder_hash is None:
        return (ORIGIN_METHOD if reorder_method is None else reorder_method, None)

    looked_up = lookup_reorder_method_for_hash(reorder_hash)
    if reorder_method is None:
        return (looked_up, reorder_hash)
    if reorder_method != looked_up:
        raise ValueError(
            f"reorder_method {reorder_method!r} does not match reorder_hash "
            f"{reorder_hash!r} (hash belongs to method {looked_up!r})"
        )
    return (reorder_method, reorder_hash)


def _params_equal(left, right):
    return canonicalize_params(left) == canonicalize_params(right)


def _params_subset_match(full_params, partial_params):
    if not partial_params:
        return True
    for key, value in partial_params.items():
        if key not in full_params:
            return False
        if _normalize_param_value(full_params[key]) != _normalize_param_value(value):
            return False
    return True


def _env_equal(left, right):
    return canonicalize_params(left or {}) == canonicalize_params(right or {})


def _env_subset_match(full_env, partial_env):
    return _params_subset_match(full_env or {}, partial_env or {})


# ---------------------------------------------------------------------------
# Reorder methods registry (reorder_methods.json)
# ---------------------------------------------------------------------------
def _read_methods_registry():
    path = _reorder_methods_path
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid methods registry: {path}")
    return data


def _write_methods_registry(registry):
    path = _reorder_methods_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(registry, fh, indent=2, sort_keys=True)
        fh.write("\n")
    tmp_path.replace(path)


def _get_method_entry(registry, method, create=False):
    if method in registry:
        return registry[method]
    blank = {
        "description": "",
        "params_schema": {},
        "env_schema": {},
        "param_sets": {},
    }
    if not create:
        return blank
    registry[method] = blank
    return registry[method]


def _read_bench_methods_config():
    """Load ``reordering-bench/methods.json`` (cached by path + mtime)."""
    global _bench_methods_cache
    _init_default_data_path()
    path = _reorder_bench_methods_path
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return {}
    cache = _bench_methods_cache
    if cache is not None and cache[0] == str(path) and cache[1] == mtime_ns:
        return cache[2]
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid methods.json (expected object): {path}")
    _bench_methods_cache = (str(path), mtime_ns, data)
    return data


def _bench_method_spec(method):
    """Return methods.json entry for ``method``, or None."""
    if method == ORIGIN_METHOD:
        return None
    return _read_bench_methods_config().get(method)


def _schema_from_bench_spec(spec, *, kind):
    """Build a resolve-ready schema from a methods.json method entry."""
    key = "params_schema" if kind == "params" else "env_schema"
    raw = spec.get(key) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"methods.json {key} must be a dict")
    schema = {}
    for name, rule in raw.items():
        if not isinstance(rule, dict):
            raise TypeError(f"methods.json {key}.{name} must be a dict")
        entry = dict(rule)
        if kind == "env" and "default" not in entry:
            if name in ("omp_threads", "n_jobs"):
                entry["default"] = _DEFAULT_THREAD_COUNT
            elif name == "dtc_minhash_backend":
                entry["default"] = _DEFAULT_DTC_MINHASH_BACKEND
        schema[name] = entry
    return schema


def _get_method_schema(method):
    """Params schema: ``methods.json`` wins; else ``reorder_methods.json``."""
    bench = _bench_method_spec(method)
    if bench is not None:
        return _schema_from_bench_spec(bench, kind="params")
    registry = _read_methods_registry()
    entry = _get_method_entry(registry, method, create=False)
    return entry.get("params_schema", {})


def _get_method_env_schema(method):
    """Env schema: ``methods.json`` wins; else ``reorder_methods.json``."""
    bench = _bench_method_spec(method)
    if bench is not None:
        return _schema_from_bench_spec(bench, kind="env")
    registry = _read_methods_registry()
    entry = _get_method_entry(registry, method, create=False)
    return entry.get("env_schema", {})


def ensure_origin_registered():
    """Ensure the ORIGIN method is present in the reorder methods registry."""
    registry = _read_methods_registry()
    if ORIGIN_METHOD in registry:
        return
    registry[ORIGIN_METHOD] = {
        "description": "Original matrix without reordering",
        "params_schema": {},
        "env_schema": {},
        "param_sets": {ORIGIN_HASH: {"params": {}, "env": {}}},
    }
    _write_methods_registry(registry)


def _resolve_schema_values(method, schema, user_values, *, kind):
    """Merge user values with schema defaults for params or env."""
    if user_values is None:
        user_values = {}
    if not isinstance(user_values, dict):
        raise TypeError(f"{kind} must be a dict")

    resolved = {}
    for key, spec in schema.items():
        if key in user_values:
            resolved[key] = _normalize_param_value(user_values[key], spec)
        elif "default" in spec:
            resolved[key] = spec["default"]
        elif spec.get("optional"):
            # methods.json optional keys (e.g. groot-cpp i_k) stay unset.
            continue
        else:
            if kind == "params":
                _raise_missing_params_hint(method, user_values, missing_keys=[key])
            raise ValueError(
                f"Missing required {kind} key '{key}' for reorder method '{method}'. "
                f"Provided: {user_values!r}"
            )

    for key in user_values:
        if key not in schema:
            raise ValueError(
                f"Unknown {kind} key '{key}' for reorder method '{method}'. "
                f"Known {kind} keys: {sorted(schema.keys())}"
            )
    return resolved


def resolve_params(method, user_params):
    """
    Merge user params with schema defaults.

    Defaults come from ``reordering-bench/methods.json`` when the method is
    listed there; otherwise from ``reorder_methods.json``.
    ``user_params`` may be ``None`` or omitted (treated as ``{}``).
    """
    if method == ORIGIN_METHOD:
        return {}
    return _resolve_schema_values(
        method, _get_method_schema(method), user_params, kind="params"
    )


def resolve_env(method, user_env=None):
    """
    Merge user env with ``env_schema`` defaults.

    Schema source matches ``resolve_params`` (methods.json first).
    ``user_env`` may be ``None`` or omitted (treated as ``{}``). Empty schema
    yields ``{}``. Default-mode catalog / load matching uses this result so
    each method's defaults map to a single ``reorder_hash``.
    """
    if method == ORIGIN_METHOD:
        return {}
    return _resolve_schema_values(
        method, _get_method_env_schema(method), user_env, kind="env"
    )


def _raise_missing_params_hint(method, user_params, missing_keys=None):
    schema = _get_method_schema(method)
    missing_keys = missing_keys or [
        key for key in schema if key not in (user_params or {}) and "default" not in schema[key]
    ]
    lines = [
        f"Missing required params for reorder method '{method}'.",
        f"  Provided: {user_params!r}",
    ]
    if schema:
        lines.append("  Schema defaults:")
        for key, spec in sorted(schema.items()):
            default = spec.get("default", "<none>")
            lines.append(f"    {key}: default={default!r}, type={spec.get('type', 'any')}")
    catalog_rows = list_catalog_entries(reorder_method=method, all_param_sets=True)
    if catalog_rows:
        lines.append("  Stored param instances (sample):")
        for row in catalog_rows[:5]:
            lines.append(f"    matrix={row['matrix']!r}, params={row['params']!r}")
    lines.append(
        f"  Usage: load_matrix(matrix, reorder_method={method!r}, params=...)"
    )
    raise ValueError("\n".join(lines))


def set_reorder_method_env_defaults(method, defaults):
    """Update ``env_schema`` defaults in ``reorder_methods.json``.

    For methods listed in ``reordering-bench/methods.json``, ``resolve_env`` /
    default catalog matching read that file instead; this API only updates
    the registry fallback / inventory display.
    """
    if not isinstance(defaults, dict):
        raise TypeError("defaults must be a dict")
    registry = _read_methods_registry()
    entry = _get_method_entry(registry, method, create=True)
    schema = entry.setdefault("env_schema", {})
    for key, value in defaults.items():
        if key not in schema:
            schema[key] = {"type": type(value).__name__}
        schema[key]["default"] = _normalize_param_value(value)
    _write_methods_registry(registry)


def _register_method_params(method, params, env=None):
    registry = _read_methods_registry()
    entry = _get_method_entry(registry, method, create=True)
    env = json.loads(canonicalize_params(env or {}))
    reorder_hash = compute_reorder_hash(method, params, env)
    param_sets = entry.setdefault("param_sets", {})
    if reorder_hash not in param_sets:
        record = {"params": json.loads(canonicalize_params(params))}
        record["env"] = env
        param_sets[reorder_hash] = record
    _write_methods_registry(registry)


def list_reorder_methods(method=None):
    """
    List reorder method metadata.

    With ``method=None``, return a list of all method entries.
    With a method name, return that method's entry dict.

    ``params_schema`` / ``env_schema`` reflect the effective resolve schemas
    (``methods.json`` when present, else registry).
    """
    registry = _read_methods_registry()
    bench = _read_bench_methods_config()

    def _entry_for(name, reg_entry):
        out = {
            "method": name,
            "description": reg_entry.get("description", ""),
            "param_sets": reg_entry.get("param_sets", {}),
            "params_schema": _get_method_schema(name),
            "env_schema": _get_method_env_schema(name),
        }
        if name in bench:
            out["schema_source"] = "reordering-bench/methods.json"
        else:
            out["schema_source"] = "reorder_methods.json"
        return out

    if method is None:
        names = sorted(set(registry) | set(bench))
        return [
            _entry_for(name, registry.get(name) or {})
            for name in names
        ]
    reg_entry = registry.get(method) or {}
    if method not in registry and method not in bench:
        return {
            "method": method,
            "description": "",
            "params_schema": {},
            "env_schema": {},
            "param_sets": {},
            "schema_source": "reorder_methods.json",
        }
    return _entry_for(method, reg_entry)


# ---------------------------------------------------------------------------
# Catalog (catalog.csv)
# ---------------------------------------------------------------------------
def _normalize_catalog_row(row):
    normalized = {col: "" for col in CATALOG_COLUMNS}
    normalized.update({k: row.get(k, "") for k in CATALOG_COLUMNS})
    return normalized


def _read_catalog():
    path = _catalog_path
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return []
        for row in reader:
            if not row.get("matrix"):
                continue
            rows.append(_normalize_catalog_row(row))
    return rows


def _write_catalog(rows):
    path = _catalog_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".csv.tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CATALOG_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CATALOG_COLUMNS})
    tmp_path.replace(path)


def _catalog_key(row):
    return (row["matrix"], row["reorder_method"], row["reorder_hash"])


def _upsert_catalog_row(new_row):
    """Insert or replace a catalog row by its (matrix, method, hash) key."""
    row = _normalize_catalog_row(new_row)
    key = _catalog_key(row)
    catalog = [r for r in _read_catalog() if _catalog_key(r) != key]
    catalog.append(row)
    _write_catalog(catalog)


def _parse_reorder_time_ms(value):
    if value in (None, ""):
        return None
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"reorder_time_ms must be non-negative, got {parsed}")
    return parsed


def _catalog_entry_from_row(row):
    return {
        "matrix": row["matrix"],
        "reorder_method": row["reorder_method"],
        "reorder_hash": row["reorder_hash"],
        "params": json.loads(row.get("params_json") or "{}"),
        "env": json.loads(row.get("env_json") or "{}"),
        "nnz": int(row["nnz"]) if row.get("nnz") not in (None, "") else None,
        "rows": int(row["rows"]) if row.get("rows") not in (None, "") else None,
        "reorder_time_ms": _parse_reorder_time_ms(row.get("reorder_time_ms")),
        "created_at": row.get("created_at", ""),
        "source": row.get("source", ""),
    }


def list_catalog_entries(
    matrix=None,
    reorder_method=None,
    params=None,
    env=None,
    *,
    all_param_sets=False,
):
    """
    List matrices from the catalog (originals and reorder variants).

    By default (``all_param_sets=False``), for a given reorder method only the
    schema-default ``params`` + ``env`` variant is returned (one
    ``reorder_hash``). With ``all_param_sets=True``, return all matching
    catalog instances; ``params`` / ``env`` partially constrain them.
    """
    results = []
    for row in _read_catalog():
        if matrix is not None and row["matrix"] != matrix:
            continue
        if reorder_method is not None and row["reorder_method"] != reorder_method:
            continue

        row_method = row["reorder_method"]
        row_params = json.loads(row.get("params_json") or "{}")
        row_env = json.loads(row.get("env_json") or "{}")

        if row_method == ORIGIN_METHOD:
            include = True
        elif all_param_sets:
            include = True
            if params and not _params_subset_match(row_params, params):
                include = False
            if env is not None and not _env_subset_match(row_env, env):
                include = False
        else:
            resolved_params = resolve_params(row_method, params or {})
            resolved_env = resolve_env(row_method, env or {})
            include = (
                _params_equal(row_params, resolved_params)
                and _env_equal(row_env, resolved_env)
            )

        if include:
            results.append(_catalog_entry_from_row(row))

    results.sort(
        key=lambda item: (
            item["matrix"],
            item["reorder_method"],
            item["reorder_hash"],
        )
    )
    return results


def get_reorder_time_ms(matrix, reorder_method, params=None, env=None, reorder_hash=None):
    """Return ``reorder_time_ms`` from the catalog for a stored variant, or None."""
    if reorder_hash is None:
        reorder_hash = compute_reorder_hash(reorder_method, params or {}, env or {})
    key = (matrix, reorder_method, reorder_hash)
    for row in _read_catalog():
        if _catalog_key(row) == key:
            return _parse_reorder_time_ms(row.get("reorder_time_ms"))
    return None


def has_reorder_perm(matrix, reorder_method, params, env=None):
    """Return True when a reorder variant is fully stored (catalog row + perm file)."""
    if params is None:
        raise ValueError("params must be provided explicitly (use {} if empty)")
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")
    if env is None:
        env = {}
    if not isinstance(env, dict):
        raise TypeError("env must be a dict")

    params = json.loads(canonicalize_params(params))
    env = json.loads(canonicalize_params(env)) if env else {}
    reorder_hash = compute_reorder_hash(reorder_method, params, env)
    key = (matrix, reorder_method, reorder_hash)
    catalog_has = any(_catalog_key(row) == key for row in _read_catalog())
    if not catalog_has:
        return False
    return get_perm_path(matrix, reorder_method, reorder_hash).exists()


# ---------------------------------------------------------------------------
# Matrix file I/O
# ---------------------------------------------------------------------------
def save_matrix(name, csr, npz_dir=None, *, source="", update_catalog=True):
    """
    Save a CSR matrix as ``<name>.pattern.npz`` and ``<name>.data.npz``.

    By default this also upserts an ORIGIN row into ``catalog.csv`` and ensures
    the ORIGIN method is registered.
    """
    npz_dir = Path(npz_dir or _npz_path)
    npz_dir.mkdir(parents=True, exist_ok=True)

    if not scipy.sparse.isspmatrix_csr(csr):
        csr = csr.tocsr()

    pattern_path = get_pattern_path(name, npz_dir)
    data_path = get_data_path(name, npz_dir)

    # Temp names must end with .npz, otherwise np.savez appends another ".npz".
    pattern_tmp = npz_dir / f"{name}.pattern.tmp.npz"
    data_tmp = npz_dir / f"{name}.data.tmp.npz"

    try:
        np.savez(
            pattern_tmp,
            format=np.array(b"csr"),
            shape=np.array(csr.shape),
            indices=csr.indices,
            indptr=csr.indptr,
        )
        np.savez(data_tmp, data=csr.data)
        pattern_tmp.replace(pattern_path)
        data_tmp.replace(data_path)
    except Exception:
        for path in (pattern_tmp, data_tmp):
            if path.exists():
                path.unlink()
        raise

    if update_catalog:
        ensure_origin_registered()
        _upsert_catalog_row(
            {
                "matrix": name,
                "reorder_method": ORIGIN_METHOD,
                "reorder_hash": ORIGIN_HASH,
                "params_json": "{}",
                "env_json": "{}",
                "nnz": str(int(csr.nnz)),
                "rows": str(int(csr.shape[0])),
                "reorder_time_ms": "0",
                "created_at": _utc_now_iso(),
                "source": source or "",
            }
        )


def matrices_equal_structure(left, right):
    """Return True if two CSR matrices share shape and sparsity pattern."""
    return (
        left.shape == right.shape
        and left.nnz == right.nnz
        and np.array_equal(left.indices, right.indices)
        and np.array_equal(left.indptr, right.indptr)
    )


def matrix_files_complete(name, npz_dir=None):
    """Return True if both pattern and data files exist for ``name``."""
    npz_dir = npz_dir or _npz_path
    return (
        get_pattern_path(name, npz_dir).exists()
        and get_data_path(name, npz_dir).exists()
    )


def matrix_memory_size(mat):
    """Approximate in-memory size (bytes) of a CSR matrix."""
    return mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes


def list_converted_matrices(npz_dir=None):
    """Return matrix names that have complete ``.npz`` files."""
    npz_dir = Path(npz_dir or _npz_path)
    if not npz_dir.exists():
        return []
    names = []
    for pattern_path in sorted(npz_dir.glob(f"*{PATTERN_SUFFIX}")):
        name = pattern_path.name[: -len(PATTERN_SUFFIX)]
        if matrix_files_complete(name, npz_dir):
            names.append(name)
    return names


# ---------------------------------------------------------------------------
# Reorder permutation storage
# ---------------------------------------------------------------------------
def get_perm_path(matrix, reorder_method, reorder_hash):
    """Path to the perm file for a reorder variant."""
    return _reorder_perms_dir / reorder_method / f"{matrix}_{reorder_hash}{PERM_SUFFIX}"


def _validate_row_perm(row_perm, nrows):
    perm = np.asarray(row_perm, dtype=np.int64)
    if perm.ndim != 1:
        raise ValueError("row_perm must be a 1-D array")
    if perm.shape[0] != nrows:
        raise ValueError(f"row_perm length {perm.shape[0]} != matrix rows {nrows}")
    if not np.array_equal(np.sort(perm), np.arange(nrows, dtype=np.int64)):
        raise ValueError("row_perm is not a valid permutation of [0, nrows)")
    return perm


def _apply_row_perm(csr, row_perm):
    """Apply vertex renumbering ``A' = A[perm, :][:, perm]`` (P A P^T).

    ``row_perm[new_i]`` is the original index at new index ``new_i``. The same
    permutation is applied to rows and columns. Requires a square matrix.
    """
    if not scipy.sparse.isspmatrix_csr(csr):
        csr = csr.tocsr()
    nrows, ncols = csr.shape
    if nrows != ncols:
        raise ValueError(
            f"reorder perm requires a square matrix (vertex renumbering), "
            f"got shape {csr.shape}"
        )
    perm = np.asarray(row_perm, dtype=np.int64)
    if perm.shape[0] != nrows:
        raise ValueError(
            f"row_perm length {perm.shape[0]} != matrix dimension {nrows}"
        )
    return csr[perm, :][:, perm].tocsr()


def _load_row_perm(matrix, reorder_method, reorder_hash):
    path = get_perm_path(matrix, reorder_method, reorder_hash)
    if not path.exists():
        raise FileNotFoundError(
            f"Reorder perm not found: {path} "
            f"(matrix={matrix!r}, reorder_method={reorder_method!r}, "
            f"reorder_hash={reorder_hash!r})"
        )
    with np.load(path) as archive:
        row_perm = _materialize_array(archive["row_perm"])
    return row_perm


def save_reorder_perm(
    matrix,
    reorder_method,
    params,
    row_perm,
    *,
    env=None,
    source=None,
    reorder_time_ms=None,
    force=False,
):
    """
    Store a vertex permutation for a reorder variant and upsert its catalog row.

    ``params`` must be passed explicitly (use ``{}`` for parameterless methods).
    ``env`` is optional; non-empty env keys participate in ``reorder_hash``.
    ``row_perm[new_i]`` is the original index at new index ``new_i``. On load,
    the same perm is applied to rows and columns: ``A' = A[perm,:][:,perm]``.
    The base matrix must be square.
    """
    if reorder_method == ORIGIN_METHOD:
        raise ValueError("save_reorder_perm cannot be used for ORIGIN matrices")
    if not matrix_files_complete(matrix):
        raise FileNotFoundError(f"Base matrix '{matrix}' does not have complete npz files")
    if params is None:
        raise ValueError("params must be provided explicitly (use {} if empty)")
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    base = _load_base_matrix(matrix, load_mode="pattern")
    nrows, ncols = base.shape
    if nrows != ncols:
        raise ValueError(
            f"save_reorder_perm requires a square matrix, got shape {base.shape}"
        )
    perm = _validate_row_perm(row_perm, nrows)

    params = json.loads(canonicalize_params(params))
    if env is None:
        env = {}
    if not isinstance(env, dict):
        raise TypeError("env must be a dict")
    env = json.loads(canonicalize_params(env)) if env else {}
    env_json = canonicalize_params(env)
    params_json = canonicalize_params(params)
    reorder_hash = compute_reorder_hash(reorder_method, params, env)
    perm_path = get_perm_path(matrix, reorder_method, reorder_hash)

    catalog = _read_catalog()
    key = (matrix, reorder_method, reorder_hash)
    existing_row = next((r for r in catalog if _catalog_key(r) == key), None)

    if perm_path.exists() and not force and existing_row is not None:
        return False

    if reorder_time_ms is not None:
        reorder_time_ms = _parse_reorder_time_ms(reorder_time_ms)
        reorder_time_ms_text = f"{float(reorder_time_ms):.10g}"
    elif existing_row is not None:
        reorder_time_ms_text = existing_row.get("reorder_time_ms", "")
    else:
        reorder_time_ms_text = ""

    perm_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = perm_path.with_suffix(".perm.tmp.npz")
    created_at = _utc_now_iso()

    try:
        save_kwargs = {
            "row_perm": perm,
            "matrix": np.array(matrix),
            "reorder_method": np.array(reorder_method),
            "params_json": np.array(params_json),
            "reorder_hash": np.array(reorder_hash),
            "env_json": np.array(env_json),
            "created_at": np.array(created_at),
        }
        if source is not None:
            save_kwargs["source"] = np.array(source)
        np.savez(tmp_path, **save_kwargs)
        tmp_path.replace(perm_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    _register_method_params(reorder_method, params, env)

    _upsert_catalog_row(
        {
            "matrix": matrix,
            "reorder_method": reorder_method,
            "reorder_hash": reorder_hash,
            "params_json": params_json,
            "env_json": env_json,
            "nnz": str(int(base.nnz)),
            "rows": str(int(nrows)),
            "reorder_time_ms": reorder_time_ms_text,
            "created_at": created_at,
            "source": source or "",
        }
    )
    return True


def _resolve_delete_targets(
    reorder_method,
    *,
    matrix=None,
    params=None,
    env=None,
    reorder_hash=None,
    all_param_sets=False,
):
    """Return (matrix, reorder_method, reorder_hash) tuples to delete."""
    if reorder_method == ORIGIN_METHOD:
        raise ValueError("delete_reorder_data cannot be used for ORIGIN matrices")

    if reorder_hash is not None:
        entries = list_catalog_entries(
            matrix=matrix,
            reorder_method=reorder_method,
            all_param_sets=True,
        )
        targets = [
            (entry["matrix"], entry["reorder_method"], entry["reorder_hash"])
            for entry in entries
            if entry["reorder_hash"] == reorder_hash
        ]
        if not targets and matrix is not None:
            targets = [(matrix, reorder_method, reorder_hash)]
        return targets

    entries = list_catalog_entries(
        matrix=matrix,
        reorder_method=reorder_method,
        params=params,
        env=env,
        all_param_sets=all_param_sets,
    )
    return [
        (entry["matrix"], entry["reorder_method"], entry["reorder_hash"])
        for entry in entries
    ]


def _read_structure_metrics_csv_rows():
    path = get_structure_metrics_path()
    if not path.exists():
        return [], path
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return [], path
        for row in reader:
            if row.get("matrix"):
                rows.append(row)
    return rows, path


def _read_merged_data_csv_rows():
    path = get_merged_data_path()
    if not path.exists():
        return [], path
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return [], path
        for row in reader:
            if row.get("matrix"):
                rows.append(row)
    return rows, path


def _prune_method_param_sets(reorder_method, hashes):
    """Remove param-set registry entries no longer referenced by the catalog."""
    if not hashes:
        return []
    registry = _read_methods_registry()
    entry = registry.get(reorder_method)
    if not entry:
        return []

    catalog_hashes = {
        row["reorder_hash"]
        for row in _read_catalog()
        if row["reorder_method"] == reorder_method
    }
    param_sets = entry.get("param_sets", {})
    removed = []
    for rhash in hashes:
        if rhash in catalog_hashes:
            continue
        if rhash in param_sets:
            del param_sets[rhash]
            removed.append(rhash)
    if removed:
        _write_methods_registry(registry)
    return removed


def delete_reorder_data(
    reorder_method,
    *,
    matrix=None,
    params=None,
    env=None,
    reorder_hash=None,
    all_param_sets=False,
    clean_registry=True,
    dry_run=False,
):
    """
    Delete stored reorder variants: perm files and CSV rows.

    Removes matching entries from ``catalog.csv``, ``structure_metrics.csv``,
    and ``merged_structure_spmm_data.csv``. Perm files under
    ``matrices/reorder/perms/<method>/`` are unlinked.

    Parameter selection matches ``list_catalog_entries`` / ``load_matrix``:
    when ``params`` / ``env`` are omitted, schema defaults are used unless
    ``all_param_sets=True`` (then all stored param/env instances are removed;
    ``params`` / ``env`` may partially constrain them). ``reorder_hash``
    selects a variant directly.

    Returns a summary dict with counts and the resolved target keys.
    """
    targets = _resolve_delete_targets(
        reorder_method,
        matrix=matrix,
        params=params,
        env=env,
        reorder_hash=reorder_hash,
        all_param_sets=all_param_sets,
    )
    if not targets:
        return {
            "targets": [],
            "deleted_perms": 0,
            "missing_perms": 0,
            "removed_catalog": 0,
            "removed_structure_metrics": 0,
            "removed_merged": 0,
            "removed_registry_hashes": [],
            "dry_run": dry_run,
        }

    target_set = set(targets)
    deleted_hashes = {rhash for _matrix, _method, rhash in targets}

    deleted_perms = 0
    missing_perms = 0
    for matrix_name, method, rhash in targets:
        perm_path = get_perm_path(matrix_name, method, rhash)
        if perm_path.exists():
            deleted_perms += 1
            if not dry_run:
                perm_path.unlink()
        else:
            missing_perms += 1

    catalog = _read_catalog()
    kept_catalog = []
    removed_catalog = 0
    for row in catalog:
        key = _catalog_key(row)
        if key in target_set:
            removed_catalog += 1
        else:
            kept_catalog.append(row)
    if not dry_run and removed_catalog:
        _write_catalog(kept_catalog)

    structure_rows, structure_path = _read_structure_metrics_csv_rows()
    kept_structure = []
    removed_structure = 0
    for row in structure_rows:
        key = (row["matrix"], row["reorder_method"], row["reorder_hash"])
        if key in target_set:
            removed_structure += 1
        else:
            kept_structure.append(row)
    if not dry_run and removed_structure:
        _write_csv(structure_path, STRUCTURE_METRICS_COLUMNS, kept_structure)

    merged_rows, merged_path = _read_merged_data_csv_rows()
    kept_merged = []
    removed_merged = 0
    for row in merged_rows:
        key = (row["matrix"], row["reorder_method"], row["reorder_hash"])
        if key in target_set:
            removed_merged += 1
        else:
            kept_merged.append(row)
    if not dry_run and removed_merged:
        _write_csv(merged_path, MERGED_DATA_COLUMNS, kept_merged)

    removed_registry_hashes = []
    if clean_registry and not dry_run:
        removed_registry_hashes = _prune_method_param_sets(
            reorder_method, deleted_hashes
        )

    return {
        "targets": [
            {"matrix": m, "reorder_method": method, "reorder_hash": rhash}
            for m, method, rhash in sorted(targets)
        ],
        "deleted_perms": deleted_perms,
        "missing_perms": missing_perms,
        "removed_catalog": removed_catalog,
        "removed_structure_metrics": removed_structure,
        "removed_merged": removed_merged,
        "removed_registry_hashes": removed_registry_hashes,
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# Matrix loading
# ---------------------------------------------------------------------------
def _load_base_matrix(name, load_mode=None, fill_value=1.0, npz_dir=None):
    """Load a base (original) CSR matrix by name."""
    if load_mode is None:
        load_mode = _load_mode
    if load_mode not in LOAD_MODES:
        raise ValueError(f"Unsupported load mode '{load_mode}'. Choose from {LOAD_MODES}.")

    npz_dir = Path(npz_dir or _npz_path)
    pattern_path = get_pattern_path(name, npz_dir)
    data_path = get_data_path(name, npz_dir)

    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern file not found: {pattern_path}")

    pattern = _read_npz_arrays(pattern_path, ("shape", "indices", "indptr"))
    shape = tuple(pattern["shape"])
    indices = pattern["indices"]
    indptr = pattern["indptr"]

    if load_mode == "full":
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        data = _read_npz_arrays(data_path, ("data",))["data"]
        if len(data) != len(indices):
            raise ValueError(
                f"data length {len(data)} != indices length {len(indices)} "
                f"for matrix '{name}'"
            )
    else:
        data = np.full(len(indices), fill_value, dtype=np.float64)

    return scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)


def _resolve_variant_hashes(matrix, reorder_method, reorder_hash, params, env, all_param_sets):
    """Return a list of (reorder_hash, params, env) tuples for a reorder method.

    Default mode (``all_param_sets=False``) resolves schema-default params and
    env so each method maps to a single ``reorder_hash``. With
    ``all_param_sets=True``, ``params`` / ``env`` act as optional subset filters.
    ``reorder_hash`` selects a variant directly.
    """
    entries = list_catalog_entries(
        matrix=matrix, reorder_method=reorder_method, all_param_sets=True
    )

    if reorder_hash is not None:
        for entry in entries:
            if entry["reorder_hash"] == reorder_hash:
                return [(reorder_hash, entry["params"], entry["env"])]
        return [(reorder_hash, params or {}, env or {})]

    if all_param_sets:
        result = []
        for entry in entries:
            if params and not _params_subset_match(entry["params"], params):
                continue
            if env is not None and not _env_subset_match(entry["env"], env):
                continue
            result.append((entry["reorder_hash"], entry["params"], entry["env"]))
        return result

    resolved_params = resolve_params(reorder_method, params or {})
    resolved_env = resolve_env(reorder_method, env or {})
    result = []
    for entry in entries:
        if not _params_equal(entry["params"], resolved_params):
            continue
        if not _env_equal(entry["env"], resolved_env):
            continue
        result.append((entry["reorder_hash"], entry["params"], entry["env"]))
    return result


def load_matrix(
    matrix,
    *,
    reorder_method=None,
    reorder_hash=None,
    params=None,
    env=None,
    all_param_sets=False,
    load_mode=None,
    fill_value=1.0,
    npz_dir=None,
):
    """
    Load a CSR matrix identified by ``(matrix, reorder_method, reorder_hash)``.

    * ``reorder_method=None`` (default) with no hash loads the original matrix
      (``ORIGIN``); with ``reorder_hash`` only, the method is looked up from
      the registry/catalog.
    * ``reorder_method`` + ``reorder_hash`` must match; mismatches raise.
    * ``params`` (with ``reorder_method``) selects a variant by parameters;
      when omitted, schema defaults are used.
    * ``env`` selects by environment keys; when omitted, ``env_schema``
      defaults are used (together with default params → one hash).
    * ``all_param_sets=True`` returns every matching variant.

    Returns a single ``csr_matrix`` when ``all_param_sets=False``, otherwise a
    list of dicts ``{matrix, reorder_method, reorder_hash, params, env, csr}``.
    """
    reorder_method, reorder_hash = resolve_reorder_method_and_hash(
        reorder_method, reorder_hash
    )
    base = _load_base_matrix(matrix, load_mode=load_mode, fill_value=fill_value, npz_dir=npz_dir)

    if reorder_method == ORIGIN_METHOD:
        if all_param_sets:
            return [
                {
                    "matrix": matrix,
                    "reorder_method": ORIGIN_METHOD,
                    "reorder_hash": ORIGIN_HASH,
                    "params": {},
                    "env": {},
                    "csr": base,
                }
            ]
        return base

    variants = _resolve_variant_hashes(
        matrix, reorder_method, reorder_hash, params, env, all_param_sets
    )
    if not variants:
        raise FileNotFoundError(
            f"No reorder variant found for matrix '{matrix}', method '{reorder_method}'"
            + (f", params {params!r}" if params else "")
        )

    results = []
    for rhash, rparams, renv in variants:
        perm = _load_row_perm(matrix, reorder_method, rhash)
        results.append(
            {
                "matrix": matrix,
                "reorder_method": reorder_method,
                "reorder_hash": rhash,
                "params": rparams,
                "env": renv,
                "csr": _apply_row_perm(base, perm),
            }
        )

    if all_param_sets:
        return results
    return results[0]["csr"]


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------
def iter_matrices(
    names,
    max_memory=128,
    load_mode=None,
    fill_value=1.0,
    *,
    reorder_method=ORIGIN_METHOD,
    params=None,
    reorder_hash=None,
    all_param_sets=False,
):
    """
    Iterate over matrices with optional background preloading (originals only).

    Yields ``(matrix, reorder_method, reorder_hash, csr)``. For reorder methods
    (or ``all_param_sets``), preloading is disabled and variants are produced
    per matrix.
    """
    if not names:
        return

    if reorder_method != ORIGIN_METHOD or all_param_sets or reorder_hash is not None:
        for name in names:
            if all_param_sets:
                variants = load_matrix(
                    name,
                    reorder_method=reorder_method,
                    reorder_hash=reorder_hash,
                    params=params,
                    all_param_sets=True,
                    load_mode=load_mode,
                    fill_value=fill_value,
                )
                for item in variants:
                    yield item["matrix"], item["reorder_method"], item["reorder_hash"], item["csr"]
            else:
                # Schema-default (or explicit hash/params) — do not take catalog[:1].
                rhash = reorder_hash
                if rhash is None:
                    resolved_params = resolve_params(reorder_method, params or {})
                    resolved_env = resolve_env(reorder_method, {})
                    rhash = compute_reorder_hash(reorder_method, resolved_params, resolved_env)
                csr = load_matrix(
                    name,
                    reorder_method=reorder_method,
                    reorder_hash=rhash,
                    params=None if reorder_hash is not None else params,
                    all_param_sets=False,
                    load_mode=load_mode,
                    fill_value=fill_value,
                )
                yield name, reorder_method, rhash, csr
        return

    max_memory_bytes = 0 if max_memory <= 0 else int(max_memory * 1024**3)

    if max_memory_bytes <= 0:
        for name in names:
            yield name, ORIGIN_METHOD, ORIGIN_HASH, _load_base_matrix(
                name, load_mode=load_mode, fill_value=fill_value
            )
        return

    result_q = queue.Queue()
    error_q = queue.Queue()
    stop_event = threading.Event()
    memory_cond = threading.Condition()
    current_mem = 0

    def producer():
        nonlocal current_mem
        try:
            for name in names:
                if stop_event.is_set():
                    break
                mat = _load_base_matrix(name, load_mode=load_mode, fill_value=fill_value)
                size = matrix_memory_size(mat)

                with memory_cond:
                    while current_mem > 0 and current_mem + size > max_memory_bytes:
                        if stop_event.is_set():
                            break
                        memory_cond.wait(timeout=0.1)
                    if stop_event.is_set():
                        break
                    current_mem += size
                    result_q.put((name, mat, size))

            result_q.put(None)
        except Exception as exc:
            error_q.put((name, exc))
            stop_event.set()

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    try:
        while True:
            if not error_q.empty():
                name_err, exc = error_q.get()
                stop_event.set()
                with memory_cond:
                    memory_cond.notify_all()
                raise RuntimeError(f"Error loading matrix '{name_err}'") from exc

            try:
                item = result_q.get(timeout=0.1)
            except queue.Empty:
                if stop_event.is_set() and result_q.empty():
                    break
                continue

            if item is None:
                break

            name, mat, size = item
            with memory_cond:
                current_mem -= size
                memory_cond.notify_all()

            yield name, ORIGIN_METHOD, ORIGIN_HASH, mat

    finally:
        stop_event.set()
        with memory_cond:
            memory_cond.notify_all()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Dataset handling
# ---------------------------------------------------------------------------
def list_datasets():
    """Return dataset names from ``datasets/*.txt``."""
    if not _datasets_path.exists():
        return []
    return sorted(p.stem for p in _datasets_path.glob("*.txt"))


def get_dataset_names(dataset_name):
    """Read a dataset file and return matrix names."""
    filepath = _datasets_path / f"{dataset_name}.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    with open(filepath, "r") as fh:
        return [line.strip() for line in fh if line.strip()]


def iter_dataset(
    dataset_name,
    max_memory=128,
    load_mode=None,
    fill_value=1.0,
    *,
    reorder_method=ORIGIN_METHOD,
    params=None,
    reorder_hash=None,
    all_param_sets=False,
):
    """
    Iterate over matrices in a named dataset.

    Yields ``(matrix, reorder_method, reorder_hash, csr)``.
    """
    names = get_dataset_names(dataset_name)
    yield from iter_matrices(
        names,
        max_memory=max_memory,
        load_mode=load_mode,
        fill_value=fill_value,
        reorder_method=reorder_method,
        params=params,
        reorder_hash=reorder_hash,
        all_param_sets=all_param_sets,
    )


# ---------------------------------------------------------------------------
# Structure metrics
# ---------------------------------------------------------------------------
DEFAULT_TC_H = 16
DEFAULT_TC_W = 8

STRUCTURE_KEY_COLUMNS = ("matrix", "reorder_method", "reorder_hash")

# Metric field order for structure_metrics.csv (after the key columns).
STRUCTURE_METRIC_COLUMNS = (
    "nnz",
    "rows",
    "avg_row_len",
    "max_row_len",
    "max_avg_row_len_ratio",
    "row_len_cv",
    "row_len_gini",
    "bandwidth",
    "profile",
    "tc_h",
    "tc_w",
    "tc_num",
    "tc_fill_rate",
    "window_n_col",
    "col_spread",
    "col_spread_norm",
    "row_spread",
    "row_spread_norm",
    "reuse_distance",
    "window_col_spread",
    "window_col_spread_norm",
    "window_row_spread",
    "window_row_spread_norm",
)

STRUCTURE_METRICS_COLUMNS = STRUCTURE_KEY_COLUMNS + STRUCTURE_METRIC_COLUMNS

# Integer-valued structure metric columns (all others are floats).
_INT_METRIC_COLUMNS = frozenset(
    {
        "nnz",
        "rows",
        "max_row_len",
        "bandwidth",
        "profile",
        "tc_h",
        "tc_w",
        "tc_num",
        "window_n_col",
        "reuse_distance",
    }
)


def _bandwidth_and_profile(csr):
    bandwidth = 0
    profile = 0
    indptr = csr.indptr
    indices = csr.indices
    for row in range(csr.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        if start >= end:
            continue
        dist = np.abs(row - indices[start:end])
        row_max = int(dist.max())
        bandwidth = max(bandwidth, row_max)
        profile += row_max
    return bandwidth, profile


def _row_length_stats(csr):
    """
    Return row-length imbalance stats from CSR ``indptr``.

    Returns
    -------
    avg_row_len, max_row_len, max_avg_row_len_ratio, row_len_cv, row_len_gini
    """
    num_rows = int(csr.shape[0])
    nnz = int(csr.nnz)
    nan = float("nan")
    if num_rows <= 0:
        return 0.0, 0, nan, nan, nan
    row_lens = np.diff(np.asarray(csr.indptr, dtype=np.int64)).astype(np.float64, copy=False)
    avg_row_len = float(nnz) / float(num_rows)
    max_row_len = int(row_lens.max()) if row_lens.size else 0
    if avg_row_len > 0:
        ratio = float(max_row_len) / avg_row_len
        # Population CV: std(r) / mean(r).
        row_len_cv = float(row_lens.std(ddof=0) / avg_row_len)
    else:
        ratio = nan
        row_len_cv = nan
    # Gini of non-negative row lengths (0=equal, ~1=extreme inequality).
    total = float(row_lens.sum())
    if total <= 0.0 or num_rows == 0:
        row_len_gini = 0.0 if num_rows > 0 and max_row_len == 0 else nan
    else:
        sorted_lens = np.sort(row_lens)
        index = np.arange(1, num_rows + 1, dtype=np.float64)
        row_len_gini = float(
            (2.0 * np.dot(index, sorted_lens)) / (num_rows * total) - (num_rows + 1.0) / num_rows
        )
        # Tiny negative from float noise when nearly equal.
        if row_len_gini < 0.0:
            row_len_gini = 0.0
    return avg_row_len, max_row_len, ratio, row_len_cv, row_len_gini


def _id_span(first_id, last_id):
    """Inclusive span: last_id - first_id + 1."""
    return int(last_id - first_id + 1)


def _col_spread_and_norm(csc):
    """Column row-id span and normalized span on a CSC matrix."""
    col_spread = 0
    col_spread_norm = 0.0
    indptr = csc.indptr
    indices = csc.indices
    for col in range(csc.shape[1]):
        start, end = indptr[col], indptr[col + 1]
        if start >= end:
            continue
        rows = indices[start:end]
        span = _id_span(rows.min(), rows.max())
        nnz = end - start
        col_spread += span
        col_spread_norm += span / nnz
    return col_spread, col_spread_norm


def _row_spread_and_norm(csr):
    """Row column-id span and normalized span on a CSR matrix."""
    row_spread = 0
    row_spread_norm = 0.0
    indptr = csr.indptr
    indices = csr.indices
    for row in range(csr.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        if start >= end:
            continue
        cols = indices[start:end]
        span = _id_span(cols.min(), cols.max())
        nnz = end - start
        row_spread += span
        row_spread_norm += span / nnz
    return row_spread, row_spread_norm


if njit is not None:

    @njit(cache=True)
    def _reuse_distance_numba(indices):
        n = len(indices)
        if n == 0:
            return 0

        max_col = 0
        for i in range(n):
            c = indices[i]
            if c > max_col:
                max_col = c
        ncol = max_col + 1
        if ncol == 0:
            return 0

        freq = np.zeros(ncol, dtype=np.int64)
        for i in range(n):
            freq[indices[i]] += 1

        offsets = np.zeros(ncol + 1, dtype=np.int64)
        for c in range(ncol):
            offsets[c + 1] = offsets[c] + freq[c]

        col_pos = np.empty(n, dtype=np.int64)
        next_slot = offsets.copy()
        for i in range(n):
            c = indices[i]
            col_pos[next_slot[c]] = i
            next_slot[c] += 1

        marker = np.zeros(ncol, dtype=np.int32)
        total = 0
        stamp = 1

        for c in range(ncol):
            start = offsets[c]
            end = offsets[c + 1]
            count = end - start
            if count < 2:
                continue
            for j in range(1, count):
                left = col_pos[start + j - 1]
                right = col_pos[start + j]
                if right - left <= 1:
                    continue
                stamp += 1
                if stamp == 2147483647:
                    marker[:] = 0
                    stamp = 1
                distinct = 0
                for k in range(left + 1, right):
                    u = indices[k]
                    if marker[u] != stamp:
                        marker[u] = stamp
                        distinct += 1
                total += distinct
        return total

else:  # pragma: no cover

    def _reuse_distance_numba(indices):
        raise RuntimeError(
            "reuse_distance requires numba; install it with: pip install numba"
        )


def _compute_reuse_distance(csr):
    """
    Sum reuse distances over CSR ``indices`` in global storage order.

    When a column index repeats, add the number of *distinct* column values
    strictly between the previous and current occurrence (endpoints excluded).
    """
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    return int(_reuse_distance_numba(indices))


_STRUCTURE_METRIC_GROUPS = {
    "basic": (
        "rows",
        "nnz",
        "avg_row_len",
        "max_row_len",
        "max_avg_row_len_ratio",
        "row_len_cv",
        "row_len_gini",
    ),
    "csr_diag": ("bandwidth", "profile"),
    "csr_spread": ("row_spread", "row_spread_norm"),
    "csr_reuse": ("reuse_distance",),
    "csc_spread": ("col_spread", "col_spread_norm"),
    "tc_window": ("tc_h", "tc_w", "tc_num", "window_n_col", "tc_fill_rate"),
    "window_spread": (
        "window_col_spread",
        "window_col_spread_norm",
        "window_row_spread",
        "window_row_spread_norm",
    ),
}

_METRIC_TO_GROUP = {
    metric: group
    for group, names in _STRUCTURE_METRIC_GROUPS.items()
    for metric in names
}

ALL_STRUCTURE_METRICS = frozenset(_METRIC_TO_GROUP)

_METRIC_ALIASES = {
    "tc": "tc_num",
}


def _normalize_metrics_arg(metrics):
    """Return a frozenset of metric names, or None for all metrics."""
    if metrics is None:
        return None
    if isinstance(metrics, str):
        names = {item.strip() for item in metrics.split(",") if item.strip()}
    else:
        names = {str(item).strip() for item in metrics if str(item).strip()}
    if not names:
        raise ValueError("metrics must name at least one structure metric")
    names = {_METRIC_ALIASES.get(name, name) for name in names}
    unknown = names - ALL_STRUCTURE_METRICS
    if unknown:
        raise ValueError(f"Unknown structure metric(s): {', '.join(sorted(unknown))}")
    return frozenset(names)


def parse_structure_metrics_arg(metrics):
    """Parse a metric name, comma-separated string, or iterable of names."""
    return _normalize_metrics_arg(metrics)


def _groups_for_metrics(metrics):
    if metrics is None:
        return tuple(_STRUCTURE_METRIC_GROUPS)
    groups = []
    seen = set()
    for metric in metrics:
        group = _METRIC_TO_GROUP[metric]
        if group not in seen:
            seen.add(group)
            groups.append(group)
    return tuple(groups)


def _build_window_compressed_csr(csr, tc_h):
    """Compress each row window into a single row (union of nonzero columns)."""
    num_rows, num_cols = csr.shape
    num_windows = (num_rows + tc_h - 1) // tc_h
    indptr = np.zeros(num_windows + 1, dtype=np.int64)
    chunks = []

    for window in range(num_windows):
        r0 = window * tc_h
        r1 = min((window + 1) * tc_h, num_rows)
        start, end = csr.indptr[r0], csr.indptr[r1]
        if start < end:
            chunks.append(np.unique(csr.indices[start:end]))
        indptr[window + 1] = indptr[window] + (0 if start >= end else len(chunks[-1]))

    if chunks:
        indices = np.concatenate(chunks)
    else:
        indices = np.array([], dtype=csr.indices.dtype)

    data = np.ones(len(indices), dtype=np.float64)
    return scipy.sparse.csr_matrix((data, indices, indptr), shape=(num_windows, num_cols))


def _row_window_tc_stats(csr, tc_h, tc_w):
    """Return window_n_col and tc_num."""
    num_rows = csr.shape[0]
    indptr = csr.indptr
    indices = csr.indices
    num_windows = (num_rows + tc_h - 1) // tc_h

    window_n_col = 0
    tc_num = 0

    for window in range(num_windows):
        r0 = window * tc_h
        r1 = min((window + 1) * tc_h, num_rows)
        start, end = indptr[r0], indptr[r1]
        if start >= end:
            continue
        n = len(np.unique(indices[start:end]))
        window_n_col += n
        tc_num += (n + tc_w - 1) // tc_w

    return window_n_col, tc_num


def compute_structure_metrics(
    csr,
    tc_h=DEFAULT_TC_H,
    tc_w=DEFAULT_TC_W,
    metrics=None,
    row_perm=None,
):
    """
    Compute structural metrics for a CSR matrix.

    ``metrics`` is an optional iterable of metric names (or comma-separated
    string). When omitted, all metrics are computed. The alias ``tc`` is
    accepted for ``tc_num``. ``row_perm`` optionally applies vertex
    renumbering first: ``A' = A[perm,:][:,perm]`` (requires square ``csr``).

    Returns a dict with keys from ``STRUCTURE_METRIC_COLUMNS``.
    """
    if not scipy.sparse.isspmatrix_csr(csr):
        csr = csr.tocsr()
    if row_perm is not None:
        csr = _apply_row_perm(csr, row_perm)

    metric_names = _normalize_metrics_arg(metrics)
    groups = _groups_for_metrics(metric_names)
    result = {}

    if "basic" in groups:
        rows, _cols = csr.shape
        result["rows"] = rows
        result["nnz"] = csr.nnz
        avg_row_len, max_row_len, max_avg_row_len_ratio, row_len_cv, row_len_gini = (
            _row_length_stats(csr)
        )
        result["avg_row_len"] = avg_row_len
        result["max_row_len"] = max_row_len
        result["max_avg_row_len_ratio"] = max_avg_row_len_ratio
        result["row_len_cv"] = row_len_cv
        result["row_len_gini"] = row_len_gini

    if "csr_diag" in groups:
        bandwidth, profile = _bandwidth_and_profile(csr)
        result["bandwidth"] = bandwidth
        result["profile"] = profile

    if "csr_spread" in groups:
        row_spread, row_spread_norm = _row_spread_and_norm(csr)
        result["row_spread"] = row_spread
        result["row_spread_norm"] = row_spread_norm

    if "csr_reuse" in groups:
        result["reuse_distance"] = _compute_reuse_distance(csr)

    if "csc_spread" in groups:
        csc = csr.tocsc()
        col_spread, col_spread_norm = _col_spread_and_norm(csc)
        result["col_spread"] = col_spread
        result["col_spread_norm"] = col_spread_norm

    if "tc_window" in groups:
        window_n_col, tc_num = _row_window_tc_stats(csr, tc_h, tc_w)
        nnz = result.get("nnz", csr.nnz)
        tc_capacity = tc_num * tc_h * tc_w
        tc_fill_rate = nnz / tc_capacity if tc_capacity > 0 else float("nan")
        result["tc_h"] = tc_h
        result["tc_w"] = tc_w
        result["tc_num"] = tc_num
        result["window_n_col"] = window_n_col
        result["tc_fill_rate"] = tc_fill_rate

    if "window_spread" in groups:
        compressed = _build_window_compressed_csr(csr, tc_h)
        window_row_spread, window_row_spread_norm = _row_spread_and_norm(compressed)
        compressed_csc = compressed.tocsc()
        window_col_spread, window_col_spread_norm = _col_spread_and_norm(compressed_csc)
        result["window_col_spread"] = window_col_spread
        result["window_col_spread_norm"] = window_col_spread_norm
        result["window_row_spread"] = window_row_spread
        result["window_row_spread_norm"] = window_row_spread_norm

    if metric_names is not None:
        return {name: result[name] for name in metric_names}
    return result


def format_metric_value(value):
    """Format a metric value for CSV output."""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


# ---------------------------------------------------------------------------
# Merge structure metrics with SpMM bench timings
# ---------------------------------------------------------------------------
DEFAULT_SPMM_METHODS = (
    "dtc-spmm",
    "dtc-spmm-balance",
    "voltrix",
    "acc-spmm",
    "flashsparse",
)

DEFAULT_FEATURE_DIMS = (32, 64, 128, 256, 512)

MERGED_KEY_COLUMNS = (
    "matrix",
    "reorder_method",
    "reorder_hash",
    "spmm_method",
    "spmm_tc_h",
    "spmm_tc_w",
    "feature_dim",
    "tc_h",
    "tc_w",
)

MERGED_DATA_COLUMNS = (
    "matrix",
    "reorder_method",
    "reorder_hash",
    "spmm_method",
    "spmm_tc_h",
    "spmm_tc_w",
    "feature_dim",
    "rows",
    "nnz",
    "avg_row_len",
    "max_row_len",
    "max_avg_row_len_ratio",
    "row_len_cv",
    "row_len_gini",
    "bandwidth",
    "profile",
    "tc_h",
    "tc_w",
    "tc_num",
    "tc_fill_rate",
    "window_n_col",
    "col_spread",
    "col_spread_norm",
    "row_spread",
    "row_spread_norm",
    "reuse_distance",
    "window_col_spread",
    "window_col_spread_norm",
    "window_row_spread",
    "window_row_spread_norm",
    "time_ms",
    "gflops",
    "success",
)


def compute_gflops(nnz, feature_dim, time_ms):
    """GFLOPS = 2 * nnz * feature_dim / (time_ms * 1e6)."""
    if time_ms <= 0:
        raise ValueError(f"time_ms must be positive, got {time_ms}")
    return 2.0 * float(nnz) * float(feature_dim) / (float(time_ms) * 1_000_000.0)


def _matrix_key_of(row):
    return (row["matrix"], row["reorder_method"], row["reorder_hash"])


def read_structure_metrics(path, tc_h=None, tc_w=None):
    """
    Read ``structure_metrics.csv`` into a list of dicts.

    Optionally filter to a specific ``(tc_h, tc_w)`` structure tile.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structure metrics CSV not found: {path}")

    # Newly added columns may be absent until compute_metrics.py is re-run.
    _optional_metric_columns = frozenset(
        {
            "avg_row_len",
            "max_row_len",
            "max_avg_row_len_ratio",
            "row_len_cv",
            "row_len_gini",
        }
    )

    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Structure metrics CSV has no header: {path}")
        missing = set(STRUCTURE_METRICS_COLUMNS) - set(reader.fieldnames)
        hard_missing = missing - _optional_metric_columns
        if hard_missing:
            raise ValueError(f"Structure metrics CSV missing columns: {sorted(hard_missing)}")
        required_nonempty = tuple(
            col for col in STRUCTURE_METRIC_COLUMNS if col not in _optional_metric_columns
        )
        for row in reader:
            if not row.get("matrix"):
                continue
            if any(row.get(k, "").strip() == "" for k in required_nonempty):
                continue
            parsed = {k: row[k] for k in STRUCTURE_KEY_COLUMNS}
            for key in STRUCTURE_METRIC_COLUMNS:
                raw = row.get(key, "").strip()
                if raw == "":
                    if key in _optional_metric_columns:
                        continue
                    break
                if key in _INT_METRIC_COLUMNS:
                    parsed[key] = int(float(raw))
                else:
                    parsed[key] = float(raw)
            else:
                if tc_h is not None and parsed["tc_h"] != tc_h:
                    continue
                if tc_w is not None and parsed["tc_w"] != tc_w:
                    continue
                rows.append(parsed)
    return rows


def read_spmm_bench_rows(path, spmm_methods=None):
    """Read a SpMM bench CSV (new schema) into a list of dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bench CSV not found: {path}")

    method_set = set(spmm_methods) if spmm_methods else None
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        required = {
            "matrix",
            "reorder_method",
            "reorder_hash",
            "spmm_method",
            "spmm_tc_h",
            "spmm_tc_w",
            "feature_dim",
            "nnz",
            "time_ms",
            "success",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Bench CSV missing columns: {sorted(missing)}")
        for row in reader:
            if method_set is not None and row["spmm_method"] not in method_set:
                continue
            success = str(row.get("success", "")).strip().lower() in ("1", "true", "yes")
            if not success:
                continue
            time_ms = float(row["time_ms"]) if row.get("time_ms") not in (None, "") else 0.0
            if time_ms <= 0:
                continue
            rows.append(
                {
                    "matrix": row["matrix"].strip(),
                    "reorder_method": row["reorder_method"].strip(),
                    "reorder_hash": row["reorder_hash"].strip(),
                    "spmm_method": row["spmm_method"].strip(),
                    "spmm_tc_h": int(row["spmm_tc_h"]),
                    "spmm_tc_w": int(row["spmm_tc_w"]),
                    "feature_dim": int(row["feature_dim"]),
                    "nnz": int(row["nnz"]),
                    "time_ms": time_ms,
                    "success": 1,
                }
            )
    return rows


def _merged_row_key(row):
    return tuple(int(row[k]) if k in ("spmm_tc_h", "spmm_tc_w", "feature_dim", "tc_h", "tc_w")
                 else row[k] for k in MERGED_KEY_COLUMNS)


def _read_merged_data(path):
    path = Path(path)
    if not path.exists():
        return {}
    rows_by_key = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = tuple(
                int(row[k]) if k in ("spmm_tc_h", "spmm_tc_w", "feature_dim", "tc_h", "tc_w")
                else row[k]
                for k in MERGED_KEY_COLUMNS
            )
            rows_by_key[key] = {col: row.get(col, "") for col in MERGED_DATA_COLUMNS}
    return rows_by_key


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(path)


def _build_merged_rows(structure_rows, bench_rows, feature_dims):
    """Join structure and bench rows by matrix key (cartesian over structure tc)."""
    structure_by_key = {}
    for srow in structure_rows:
        structure_by_key.setdefault(_matrix_key_of(srow), []).append(srow)

    merged = []
    for bench in bench_rows:
        if feature_dims is not None and bench["feature_dim"] not in feature_dims:
            continue
        key = _matrix_key_of(bench)
        srows = structure_by_key.get(key)
        if not srows:
            continue
        gflops = compute_gflops(bench["nnz"], bench["feature_dim"], bench["time_ms"])
        for srow in srows:
            if srow.get("nnz") != bench["nnz"]:
                print(
                    f"Warning: nnz mismatch for {key}: "
                    f"structure={srow['nnz']} bench={bench['nnz']} "
                    f"(using bench nnz for GFLOPS, structure nnz in merged row)"
                )
            row = {
                "matrix": bench["matrix"],
                "reorder_method": bench["reorder_method"],
                "reorder_hash": bench["reorder_hash"],
                "spmm_method": bench["spmm_method"],
                "spmm_tc_h": bench["spmm_tc_h"],
                "spmm_tc_w": bench["spmm_tc_w"],
                "feature_dim": bench["feature_dim"],
                "time_ms": bench["time_ms"],
                "gflops": gflops,
                "success": bench["success"],
            }
            for col in STRUCTURE_METRIC_COLUMNS:
                row[col] = srow[col]
            merged.append(row)
    return merged


def merge_structure_spmm(
    structure_csv=None,
    bench_csv=None,
    output_csv=None,
    *,
    spmm_methods=None,
    feature_dims=None,
    structure_tc_h=None,
    structure_tc_w=None,
    force=False,
):
    """
    Join ``structure_metrics.csv`` with a SpMM bench CSV and incrementally write
    ``merged_structure_spmm_data.csv``.

    Matrices are matched by ``(matrix, reorder_method, reorder_hash)``. When a
    structure tile filter is not given, all structure ``(tc_h, tc_w)`` rows are
    joined (cartesian product with each bench row).

    Without ``force``, existing bench/perf fields are kept, but structure metric
    columns are refreshed from the latest ``structure_metrics.csv`` (so newly
    added metrics propagate without a full rebuild).

    Returns a dict with ``output_csv``, ``added``, ``skipped``, and ``refreshed``.
    """
    structure_csv = Path(structure_csv or get_structure_metrics_path())
    output_csv = Path(output_csv or get_merged_data_path())
    bench_csv = Path(bench_csv or get_default_spmm_bench_csv())
    if not bench_csv.exists():
        raise FileNotFoundError(f"SpMM bench CSV not found: {bench_csv}")

    feature_dims = tuple(feature_dims) if feature_dims else None
    structure_rows = read_structure_metrics(structure_csv, structure_tc_h, structure_tc_w)
    bench_rows = read_spmm_bench_rows(bench_csv, spmm_methods)

    existing = {} if force else _read_merged_data(output_csv)
    new_rows = _build_merged_rows(structure_rows, bench_rows, feature_dims)

    added = 0
    skipped = 0
    refreshed = 0
    for row in new_rows:
        formatted = {
            col: format_metric_value(row[col]) if col in row else "" for col in MERGED_DATA_COLUMNS
        }
        key = _merged_row_key(row)
        if key in existing and not force:
            # Keep existing bench/perf fields, but refresh structure metric columns
            # so newly added metrics (and recomputed ones) propagate without --force.
            prev = existing[key]
            changed = False
            for col in STRUCTURE_METRIC_COLUMNS:
                new_val = formatted.get(col, "")
                if prev.get(col, "") != new_val:
                    prev[col] = new_val
                    changed = True
            # Ensure schema stays aligned if MERGED_DATA_COLUMNS grows.
            for col in MERGED_DATA_COLUMNS:
                if col not in prev:
                    prev[col] = formatted.get(col, "")
                    changed = True
            if changed:
                refreshed += 1
            skipped += 1
            continue
        existing[key] = formatted
        added += 1

    merged_rows = [existing[key] for key in sorted(existing, key=lambda k: tuple(str(x) for x in k))]
    _write_csv(output_csv, MERGED_DATA_COLUMNS, merged_rows)

    return {
        "output_csv": output_csv,
        "added": added,
        "skipped": skipped,
        "refreshed": refreshed,
    }


# ---------------------------------------------------------------------------
# Catalog validation
# ---------------------------------------------------------------------------
def validate_catalog():
    """
    Check catalog consistency against npz/perm files and the methods registry.

    Returns a list of human-readable problem descriptions (empty when valid).
    """
    problems = []
    catalog = _read_catalog()
    registry = _read_methods_registry()
    catalog_origin = set()

    for row in catalog:
        matrix = row["matrix"]
        method = row["reorder_method"]
        rhash = row["reorder_hash"]
        key = (matrix, method, rhash)

        if method == ORIGIN_METHOD:
            catalog_origin.add(matrix)
            if not matrix_files_complete(matrix):
                problems.append(f"ORIGIN catalog row {key} has no complete npz files")
                continue
            try:
                base = _load_base_matrix(matrix, load_mode="pattern")
            except Exception as exc:  # pragma: no cover
                problems.append(f"Failed to load ORIGIN matrix {matrix!r}: {exc}")
                continue
            if row.get("nnz") not in (None, "") and int(row["nnz"]) != int(base.nnz):
                problems.append(
                    f"nnz mismatch for {key}: catalog={row['nnz']} npz={base.nnz}"
                )
            if row.get("rows") not in (None, "") and int(row["rows"]) != int(base.shape[0]):
                problems.append(
                    f"rows mismatch for {key}: catalog={row['rows']} npz={base.shape[0]}"
                )
            if rhash != ORIGIN_HASH:
                problems.append(f"ORIGIN row {key} has unexpected hash (want {ORIGIN_HASH})")
        else:
            perm_path = get_perm_path(matrix, method, rhash)
            if not perm_path.exists():
                problems.append(f"Reorder catalog row {key} has no perm file: {perm_path}")
            params = json.loads(row.get("params_json") or "{}")
            env = json.loads(row.get("env_json") or "{}")
            expected = compute_reorder_hash(method, params, env)
            if expected != rhash:
                problems.append(
                    f"hash mismatch for {key}: stored={rhash} computed={expected}"
                )
            if method not in registry:
                problems.append(f"method {method!r} in catalog is not registered")

    for name in list_converted_matrices():
        if name not in catalog_origin:
            problems.append(f"npz matrix {name!r} has no ORIGIN catalog row")

    return problems
