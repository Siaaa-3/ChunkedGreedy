from __future__ import annotations

import csv
import fcntl
import json
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

RESULT_COLUMNS = [
    "matrix",
    "reorder_method",
    "params_json",
    "reorder_hash",
    "effective_params_json",
    "env_json",
    "success",
    "error",
    "load_matrix_ms",
    "prep_ms",
    "reorder_time_ms",
    "perm_trans_ms",
    "eval_time_ms",
    "timing_breakdown_json",
    "bandwidth",
    "profile",
    "tc_h",
    "tc_w",
    "tc_num",
    "tc_fill_rate",
    "source_commit",
    "timestamp",
]


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def source_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()[:12]
    except (OSError, subprocess.CalledProcessError):
        return ""


def ensure_results_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _lock_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


@contextmanager
def locked_path(path: Path) -> Iterator[None]:
    ensure_results_dir(path.parent)
    lock_path = _lock_path(path)
    with lock_path.open("w", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _append_text_line_locked(path: Path, line: str) -> None:
    with locked_path(path):
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())


def append_text_line(path: Path, line: str) -> None:
    _append_text_line_locked(path, line)


def _load_csv_rows_unlocked(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != RESULT_COLUMNS:
            raise ValueError(
                f"{path} has an outdated CSV header; write to a new file or remove it"
            )
        return list(reader)


def load_completed_keys(path: Path) -> set[tuple[str, str, str]]:
    """Return (matrix, reorder_method, reorder_hash) for successful rows in CSV."""
    with locked_path(path):
        if not path.exists() or path.stat().st_size == 0:
            return set()
        rows = _load_csv_rows_unlocked(path)
        return {
            (row["matrix"], row["reorder_method"], row["reorder_hash"])
            for row in rows
            if row.get("success") == "1"
        }


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("matrix", "")),
        str(row.get("reorder_method", "")),
        str(row.get("reorder_hash", "")),
    )


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid4().hex}")
    try:
        with tmp_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({col: row.get(col, "") for col in RESULT_COLUMNS})
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def write_result_row(
    path: Path, row: dict[str, Any], *, overwrite: bool = False
) -> None:
    """Append a result row, or replace matching (matrix, method, hash) when overwrite."""
    ensure_results_dir(path.parent)
    new_row = {col: row.get(col, "") for col in RESULT_COLUMNS}
    with locked_path(path):
        if not path.exists() or path.stat().st_size == 0:
            _write_csv_rows(path, [new_row])
            return

        existing = _load_csv_rows_unlocked(path)
        key = _row_key(new_row)

        if overwrite:
            kept = [r for r in existing if _row_key(r) != key]
            kept.append(new_row)
            _write_csv_rows(path, kept)
            return

        if any(_row_key(r) == key and r.get("success") == "1" for r in existing):
            return

        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=RESULT_COLUMNS,
                extrasaction="ignore",
            )
            writer.writerow(new_row)
            fh.flush()
            os.fsync(fh.fileno())


def make_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
