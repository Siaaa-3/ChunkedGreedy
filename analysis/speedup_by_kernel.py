"""SpMM speedup per kernel and dense width on the common SpMM set (geometric mean)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (KERNELS, MAIN_METHODS, METHOD_COLORS, METHOD_LABELS, SPMM_LABELS, TABLES, WIDTHS,
                    apply_paper_style, load_amort, save_fig, save_table)


def gm(x):
    return float(np.exp(np.log(np.asarray(x, float)).mean()))


def main():
    common = pd.read_csv(TABLES / "spmm_common_set.csv").matrix  # written by paper_numbers.py
    a = load_amort(methods=MAIN_METHODS, matrices=set(common))
    a = a[a.spmm_method.isin(KERNELS)]
    a["sp"] = a.spmm_orig_ms / a.spmm_reord_ms
    rows = [dict(kernel=k, w=w, method=m, am=g.sp.mean(), gm=gm(g.sp), mx=g.sp.max(), frac=(g.sp > 1).mean())
            for (k, w, m), g in a.groupby(["spmm_method", "feature_dim", "reorder_method"])]
    T = pd.DataFrame(rows)
    save_table(T, "speedup_by_kernel_width")

    apply_paper_style()
    fig, axes = plt.subplots(1, len(KERNELS), figsize=(16, 3.4), sharey=True)
    x = np.arange(len(WIDTHS))
    for ax, k in zip(axes, KERNELS):
        for j, m in enumerate(MAIN_METHODS):
            t = T[(T.kernel == k) & (T.method == m)].sort_values("w")
            ax.bar(x + (j - 2) * 0.16, t.gm, 0.16, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xticks(x, [str(w) for w in WIDTHS])
        ax.set_xlabel("dense width w")
        ax.set_title(SPMM_LABELS[k])
    axes[0].set_ylabel("SpMM speedup (geomean)")
    axes[0].set_ylim(0.8, None)
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save_fig("fig_spmm_speedup_by_kernel")


if __name__ == "__main__":
    main()
