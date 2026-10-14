"""Paths, configuration and plotting style shared by the analysis scripts."""
from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("CG_RESULTS", ROOT / "results"))
OUT = Path(os.environ.get("CG_OUT", ROOT / "out"))
FIGURES = OUT / "figures"
TABLES = OUT / "tables"

CONFIG = yaml.safe_load((ROOT / "configs" / "paper.yaml").read_text())
HASHES = {m: c["hash"] for m, c in CONFIG["methods"].items()}
MAIN_METHODS = CONFIG["main_methods"]
CG = CONFIG["cg"]
KERNELS = CONFIG["spmm"]["kernels"]
WIDTHS = CONFIG["spmm"]["widths"]

METHOD_LABELS = {m: c["label"] for m, c in CONFIG["methods"].items()}
METHOD_COLORS = {
    "acc": "#4C72B0",
    "dtc-tca": "#DD8452",
    "groot-py": "#55A868",
    "clusterwise-hier": "#C44E52",
    "sporder-csg": "#8172B3",
    "sporder-cg": "#64B5CD",
}
SPMM_LABELS = {
    "voltrix": "Voltrix",
    "flashsparse": "FlashSparse",
    "acc-spmm": "Acc-SpMM",
    "dtc-spmm": "DTC-SpMM",
    "acc-spmm-balance": "Acc-SpMM (load-balanced)",
}


def matrix_set(name: str) -> list[str]:
    path = ROOT / "datasets" / CONFIG["sets"][name]
    return [m for m in path.read_text().split() if m]


def load_reorder_times(methods=None, matrices=None) -> pd.DataFrame:
    """One row per (matrix, method) for the configurations in configs/paper.yaml."""
    df = pd.read_csv(RESULTS / "reorder_bench.csv")
    df = df[(df["success"] == 1) & (df["reorder_hash"] == df["reorder_method"].map(HASHES))]
    if methods is not None:
        df = df[df["reorder_method"].isin(methods)]
    if matrices is not None:
        df = df[df["matrix"].isin(matrices)]
    df = df.sort_values("timestamp").groupby(["matrix", "reorder_method"], as_index=False).tail(1)
    return df.reset_index(drop=True)


def parse_amortization(value) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"", "nan"}:
        return math.nan
    if text in {"inf", "+inf", "infinity"}:
        return math.inf
    if text in {"-inf", "-infinity"}:
        return -math.inf
    return float(text)


def load_amort(methods=None, matrices=None) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "reorder_spmm_amortization.csv")
    df = df[df["reorder_hash"] == df["reorder_method"].map(HASHES)]
    df["feature_dim"] = df["feature_dim"].astype(int)
    df["amortization"] = df["amortization"].map(parse_amortization)
    if methods is not None:
        df = df[df["reorder_method"].isin(methods)]
    if matrices is not None:
        df = df[df["matrix"].isin(matrices)]
    return df.reset_index(drop=True)


def load_quality(orders=None, matrices=None) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "quality_metrics.csv")
    if orders is not None:
        df = df[df["order"].isin(orders)]
    if matrices is not None:
        df = df[df["matrix"].isin(matrices)]
    return df.reset_index(drop=True)


def geo_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) == 0:
        return float("nan")
    return float(np.exp(np.log(arr).mean()))


def apply_paper_style(font_size: int = 12) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "legend.fontsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
        }
    )


def save_fig(name: str, *, close: bool = True) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        plt.savefig(FIGURES / f"{name}{ext}", bbox_inches="tight")
    if close:
        plt.close()


def save_table(df: pd.DataFrame, name: str, **kw) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.csv"
    df.to_csv(path, **{"index": False, **kw})
    return path
