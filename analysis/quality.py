"""Structural quality metrics (Table R3): improvement over the original order and the
Spearman correlation of each metric with SpMM speedup at w=128."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import (KERNELS, MAIN_METHODS, METHOD_COLORS, METHOD_LABELS, SPMM_LABELS, apply_paper_style, load_amort,
                    load_quality, matrix_set, save_fig, save_table)

MET = ["pillars", "tc_blocks", "bandwidth", "profile", "log_gap"]
MLABEL = ["Pillars", "TC blocks", "Bandwidth", "Profile", "Log-gap"]
ORDER = ["sporder-csg", "clusterwise-hier", "acc", "dtc-tca", "groot-py"]


def gm(x):
    return float(np.exp(np.log(np.asarray(x, float)).mean()))


def main():
    mats = matrix_set("final")
    q = load_quality(orders=["ORIGIN", *MAIN_METHODS], matrices=mats)
    o = q[q.order == "ORIGIN"].set_index("matrix")[MET]
    rows, ratios = [], []
    for m in ORDER:
        x = q[q.order == m].set_index("matrix")[MET]
        common = x.index.intersection(o.index)
        r = (o.loc[common] / x.loc[common].replace(0, np.nan)).dropna()
        ratios.append(r.assign(order=m).reset_index())
        rows.append(dict(method=m, n=len(r), **{f"{k}_gm": gm(r[k].clip(lower=1e-12)) for k in MET},
                         **{f"{k}_median": r[k].median() for k in MET}, **{f"{k}_mean": r[k].mean() for k in MET},
                         **{f"{k}_impr": (r[k] > 1).mean() for k in MET}))
    T = pd.DataFrame(rows)
    R = pd.concat(ratios)
    save_table(T, "tableR3_quality")

    a = load_amort(methods=MAIN_METHODS)
    a["sp"] = a.spmm_orig_ms / a.spmm_reord_ms
    out = []
    for k in KERNELS:
        b = a[(a.spmm_method == k) & (a.feature_dim == 128)][["matrix", "reorder_method", "sp"]]
        d = b.rename(columns={"reorder_method": "order"}).merge(R, on=["matrix", "order"])
        out.append(dict(kernel=k, n=len(d), **{m: spearmanr(np.log(d[m]), np.log(d.sp)).correlation for m in MET}))
    C = pd.DataFrame(out)
    save_table(C, "tableR3_quality_corr")
    print(T.round(3).to_string(index=False))
    print(C.round(3).to_string(index=False))

    apply_paper_style()
    plt.rcParams["axes.grid"] = False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(MET))
    for j, m in enumerate(MAIN_METHODS):
        r = R[R.order == m]
        ax1.bar(x + (j - 2) * 0.16, [gm(r[c].clip(lower=1e-12)) for c in MET], 0.16, color=METHOD_COLORS[m],
                label=METHOD_LABELS[m])
    ax1.axhline(1, color="k", lw=0.8)
    ax1.set_yscale("log")
    ax1.set_xticks(x, MLABEL)
    ax1.set_ylabel("original / reordered (geomean)\n>1 = better")
    ax1.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    im = ax2.imshow(C[MET].values, cmap="viridis", vmin=0, vmax=1)
    ax2.set_xticks(range(len(MET)), MLABEL, rotation=30)
    ax2.set_yticks(range(len(C)), [SPMM_LABELS[k] for k in C.kernel])
    for i in range(len(C)):
        for j, c in enumerate(MET):
            v = C[c].iloc[i]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v < 0.6 else "k", fontsize=8)
    ax2.set_title("Spearman corr. with SpMM speedup (w=128)")
    fig.colorbar(im, ax=ax2, fraction=0.04)
    fig.tight_layout()
    save_fig("figR3_quality_metrics")


if __name__ == "__main__":
    main()
