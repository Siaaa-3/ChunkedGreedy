import os
import sys
import time
import argparse
from pathlib import Path
import scipy.io
import scipy.sparse

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MATRICES_DIR = PROJECT_DIR / "matrices"
DIR_MTX = MATRICES_DIR / "mtx"
DIR_NPZ = MATRICES_DIR / "npz"

sys.path.insert(0, str(PROJECT_DIR))
import spmatkit  # noqa: E402


def format_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PB"


def matrix_name_from_pattern_path(path: Path) -> str:
    return path.name[: -len(spmatkit.PATTERN_SUFFIX)]


def resolve_matrix_names(matrix_name=None, dataset_name=None):
    """Return explicit matrix names to process, or None for all .mtx files."""
    if matrix_name and dataset_name:
        names = spmatkit.get_dataset_names(dataset_name)
        if matrix_name not in names:
            raise ValueError(
                f"Matrix '{matrix_name}' is not listed in dataset '{dataset_name}'"
            )
        return [matrix_name]
    if matrix_name:
        return [matrix_name]
    if dataset_name:
        names = spmatkit.get_dataset_names(dataset_name)
        if not names:
            raise FileNotFoundError(f"Dataset '{dataset_name}' is empty or not found")
        return names
    return None


def resolve_mtx_files(matrix_name=None, dataset_name=None):
    """Resolve .mtx paths from a matrix name, dataset, or the whole mtx directory."""
    names = resolve_matrix_names(matrix_name, dataset_name)
    if names is None:
        mtx_files = sorted(DIR_MTX.glob("*.mtx"))
        return mtx_files, []

    mtx_files = []
    missing = []
    for name in names:
        path = DIR_MTX / f"{name}.mtx"
        if path.exists():
            mtx_files.append(path)
        else:
            missing.append(name)
    return mtx_files, missing


def verify_npz_files(matrix_name=None, dataset_name=None):
    """Verify split npz files against source .mtx files."""
    print("=== Verification mode ===")
    output_dir = DIR_NPZ

    names = resolve_matrix_names(matrix_name, dataset_name)
    if names is not None:
        pattern_files = []
        for name in names:
            pattern_path = spmatkit.get_pattern_path(name, output_dir)
            if pattern_path.exists():
                pattern_files.append(pattern_path)
        if not pattern_files:
            print("No matching pattern files found for the requested matrix(es).")
            return
    else:
        pattern_files = sorted(output_dir.glob(f"*{spmatkit.PATTERN_SUFFIX}"))
        if not pattern_files:
            print(f"No pattern files found in {output_dir}.")
            return

    total = len(pattern_files)
    ok = 0
    missing = 0
    mismatch = 0
    problematic = []

    for i, pattern_path in enumerate(pattern_files, start=1):
        name = matrix_name_from_pattern_path(pattern_path)
        data_path = spmatkit.get_data_path(name, output_dir)

        t0 = time.perf_counter()
        try:
            converted = spmatkit.load_matrix(name, load_mode="full", npz_dir=output_dir)
            npz_time = time.perf_counter() - t0
            npz_size = pattern_path.stat().st_size + (
                data_path.stat().st_size if data_path.exists() else 0
            )
            npz_info = f"npz: {npz_time:.3f}s ({format_size(npz_size)})"
        except Exception as e:
            print(f"[{i}/{total}] {name} | npz: FAILED ({e})")
            problematic.append(name)
            mismatch += 1
            continue

        mtx_path = DIR_MTX / f"{name}.mtx"
        if not mtx_path.exists():
            print(f"[{i}/{total}] {name} | {npz_info} | mtx: MISSING")
            problematic.append(name)
            missing += 1
            continue

        mtx_size = mtx_path.stat().st_size

        t1 = time.perf_counter()
        try:
            csr_mtx = scipy.io.mmread(str(mtx_path)).tocsr()
            mtx_time = time.perf_counter() - t1
            mtx_info = f"mtx: {mtx_time:.3f}s ({format_size(mtx_size)})"
        except Exception as e:
            print(f"[{i}/{total}] {name} | {npz_info} | mtx: FAILED ({e})")
            problematic.append(name)
            mismatch += 1
            continue

        if converted.shape != csr_mtx.shape:
            print(
                f"[{i}/{total}] {name} | {npz_info} | {mtx_info} | "
                f"SHAPE MISMATCH (npz {converted.shape}, mtx {csr_mtx.shape})"
            )
            problematic.append(name)
            mismatch += 1
        elif (converted != csr_mtx).nnz == 0:
            print(f"[{i}/{total}] {name} | {npz_info} | {mtx_info} | OK")
            ok += 1
        else:
            diff_count = (converted != csr_mtx).nnz
            print(
                f"[{i}/{total}] {name} | {npz_info} | {mtx_info} | "
                f"MISMATCH ({diff_count} diffs)"
            )
            problematic.append(name)
            mismatch += 1

    print(f"\nResult: {ok} OK, {missing} missing, {mismatch} mismatches (total {total} files)")

    error_list_path = output_dir / "wrong_matrices.txt"
    if problematic:
        with open(error_list_path, "w") as f:
            for name in problematic:
                f.write(name + "\n")
        print(f"Problematic matrix list written to {error_list_path}")
    else:
        if error_list_path.exists():
            error_list_path.unlink()
        print("No problematic matrices.")


def cleanup_incomplete_matrix(name, npz_dir=DIR_NPZ):
    """Remove partial output files for a failed conversion."""
    candidates = [
        spmatkit.get_pattern_path(name, npz_dir),
        spmatkit.get_data_path(name, npz_dir),
        npz_dir / f"{name}.pattern.tmp.npz",
        npz_dir / f"{name}.data.tmp.npz",
        npz_dir / f"{name}.pattern.npz.tmp.npz",
        npz_dir / f"{name}.data.npz.tmp.npz",
    ]
    for path in candidates:
        if path.exists():
            path.unlink()


def write_failed_list(failed):
    """Write failed matrix names to failed_matrices.txt."""
    error_list_path = PROJECT_DIR / "failed_matrices.txt"
    if failed:
        with open(error_list_path, "w") as f:
            for name in failed:
                f.write(name + "\n")
        print(f"Failed matrix list written to {error_list_path}")
    elif error_list_path.exists():
        error_list_path.unlink()


def run_conversion(matrix_name=None, dataset_name=None):
    os.makedirs(DIR_NPZ, exist_ok=True)

    mtx_files, missing_mtx = resolve_mtx_files(matrix_name, dataset_name)
    if missing_mtx:
        print(
            f"Warning: {len(missing_mtx)} matrix(es) missing .mtx in {DIR_MTX}: "
            + ", ".join(missing_mtx)
        )
    if not mtx_files:
        if dataset_name:
            print(f"No .mtx files found for dataset '{dataset_name}' in {DIR_MTX}")
        elif matrix_name:
            print(f"Matrix '{matrix_name}' not found as .mtx in {DIR_MTX}")
        else:
            print(f"No .mtx files found in {DIR_MTX}")
        return

    total = len(mtx_files)
    if dataset_name:
        scope = f"dataset '{dataset_name}'"
    elif matrix_name:
        scope = f"matrix '{matrix_name}'"
    else:
        scope = f"directory {DIR_MTX}"
    print(f"Found {total} .mtx file(s) from {scope} to process into {DIR_NPZ}.")
    skipped = 0
    converted = 0
    failed = list(missing_mtx)

    for i, mtx_path in enumerate(mtx_files, start=1):
        name = mtx_path.stem
        if spmatkit.matrix_files_complete(name, DIR_NPZ):
            print(f"[{i}/{total}] Skipping {mtx_path.name} - already exists.")
            skipped += 1
            continue

        start_time = time.perf_counter()
        try:
            print(f"[{i}/{total}] Converting {mtx_path.name} ...", flush=True)
            matrix = scipy.io.mmread(str(mtx_path))
            csr = matrix.tocsr() if not scipy.sparse.isspmatrix_csr(matrix) else matrix

            spmatkit.save_matrix(name, csr, npz_dir=DIR_NPZ)

            elapsed = time.perf_counter() - start_time
            pattern_path = spmatkit.get_pattern_path(name, DIR_NPZ)
            data_path = spmatkit.get_data_path(name, DIR_NPZ)
            print(
                f"[{i}/{total}] Done {mtx_path.name} -> "
                f"{pattern_path.name}, {data_path.name} ({elapsed:.2f}s)",
                flush=True,
            )
            converted += 1

        except Exception as e:
            cleanup_incomplete_matrix(name, DIR_NPZ)
            failed.append(name)
            print(f"[{i}/{total}] ERROR {mtx_path.name}: {e}", flush=True)

    print(
        f"\nConversion done. Converted: {converted}, Skipped: {skipped}, "
        f"Failed: {len(failed)}"
    )
    write_failed_list(failed)


def main():
    parser = argparse.ArgumentParser(
        description="Convert .mtx files to split CSR .npz files, or verify them."
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify .npz files against .mtx sources (skips conversion)."
    )
    parser.add_argument(
        "-m", "--matrix",
        type=str,
        default=None,
        help="Process only this matrix name (stem, no extension).",
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default=None,
        help="Process all matrices listed in datasets/<name>.txt.",
    )
    args = parser.parse_args()

    if args.verify:
        verify_npz_files(matrix_name=args.matrix, dataset_name=args.dataset)
    else:
        run_conversion(matrix_name=args.matrix, dataset_name=args.dataset)


if __name__ == "__main__":
    main()
