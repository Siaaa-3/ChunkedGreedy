#!/usr/bin/env python3
"""Download SuiteSparse matrices and convert them to spmatkit's CSR npz format.

    python datasets/download.py --list datasets/final_300.txt --data data

Each matrix is fetched from https://sparse.tamu.edu/MM/<group>/<name>.tar.gz,
read with scipy.io.mmread and stored as <data>/npz/<name>.{pattern,data}.npz.
The Matrix Market files are removed after conversion unless --keep-mtx is given.
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import scipy.io
import scipy.sparse

ROOT = Path(__file__).resolve().parents[1]
INDEX_URL = "https://sparse.tamu.edu/files/ssstats.csv"
MM_URL = "https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"


def load_index(cache: Path) -> dict[str, str]:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(INDEX_URL) as r:
            cache.write_bytes(r.read())
    rows = list(csv.reader(io.StringIO(cache.read_text())))
    # first two lines: number of matrices, date of the index
    return {row[1]: row[0] for row in rows[2:] if len(row) > 1}


def fetch(name: str, group: str, mtx_dir: Path) -> Path:
    mtx = mtx_dir / f"{name}.mtx"
    if mtx.exists():
        return mtx
    url = MM_URL.format(group=group, name=name)
    tmp = mtx_dir / f"{name}.tar.gz"
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    with tarfile.open(tmp) as tar:
        member = tar.getmember(f"{name}/{name}.mtx")
        with tar.extractfile(member) as src, mtx.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    tmp.unlink()
    return mtx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", type=Path, required=True, help="file with one matrix name per line")
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("SPMATKIT_DATA", ROOT / "data")))
    ap.add_argument("--keep-mtx", action="store_true")
    args = ap.parse_args()

    os.environ["SPMATKIT_DATA"] = str(args.data.resolve())
    sys.path.insert(0, str(ROOT / "spmatkit"))
    import spmatkit

    names = [n for n in args.list.read_text().split() if n]
    index = load_index(args.data / "ssstats.csv")
    mtx_dir = args.data / "mtx"
    mtx_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, name in enumerate(names, 1):
        if spmatkit.matrix_files_complete(name):
            print(f"[{i}/{len(names)}] {name}: present")
            continue
        if name not in index:
            print(f"[{i}/{len(names)}] {name}: not in the SuiteSparse index")
            failed.append(name)
            continue
        try:
            mtx = fetch(name, index[name], mtx_dir)
            csr = scipy.io.mmread(str(mtx)).tocsr()
            spmatkit.save_matrix(name, scipy.sparse.csr_matrix(csr), source="suitesparse")
            if not args.keep_mtx:
                mtx.unlink()
            print(f"[{i}/{len(names)}] {name}: {csr.shape[0]} rows, {csr.nnz} nnz")
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(names)}] {name}: FAILED ({exc})")
            failed.append(name)
    if failed:
        print(f"{len(failed)} failed: {' '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
