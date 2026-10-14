"""Fig. 2: pillar count vs SpMM time in the original order (Acc-SpMM load-balanced, w=32).

The linear fit excludes matrices whose time is more than 3x off the log-log trend.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import CG, apply_paper_style, load_amort, load_quality, matrix_set, save_fig, save_table

KERNEL, WIDTH, RATIO = "acc-spmm-balance", 32, 3.0


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    b = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    a = y.mean() - b * x.mean()
    r2 = 1 - np.sum((y - (a + b * x)) ** 2) / np.sum((y - y.mean()) ** 2)
    return a, b, r2


def main():
    mats = matrix_set("final")
    # spmm_orig_ms is the original-order time; it is identical across reorder methods
    a = load_amort(methods=[CG], matrices=mats)
    t = a[(a.spmm_method == KERNEL) & (a.feature_dim == WIDTH)].set_index("matrix").spmm_orig_ms
    p = load_quality(orders=["ORIGIN"], matrices=mats).set_index("matrix").pillars
    d = pd.concat([p.rename("pillars"), t.rename("time_ms")], axis=1).dropna()
    d = d[(d.pillars > 0) & (d.time_ms > 0)]

    la, lb, _ = ols(np.log(d.pillars), np.log(d.time_ms))
    d["ratio_vs_trend"] = d.time_ms / np.exp(la + lb * np.log(d.pillars))
    d["outlier"] = (d.ratio_vs_trend < 1 / RATIO) | (d.ratio_vs_trend > RATIO)
    kept = d[~d.outlier]
    a0, b0, r2 = ols(kept.pillars, kept.time_ms)
    print(f"n={len(d)}, outliers={int(d.outlier.sum())}, fit kept={len(kept)}, R2={r2:.4f}, y={a0:.4g}+{b0:.3e}*x")
    save_table(d.reset_index(names="matrix"), "fig2_points")
    save_table(pd.DataFrame([dict(n=len(d), n_fit=len(kept), intercept=a0, slope=b0, r2=r2)]), "fig2_fit")

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(d.pillars, d.time_ms, s=18, alpha=0.55, color="#4C72B0", edgecolors="none", label="Matrices", zorder=2)
    xs = np.linspace(d.pillars.min(), d.pillars.max(), 200)
    ax.plot(xs, a0 + b0 * xs, color="#222222", linewidth=1.8, label="Linear Fit", zorder=3)
    ax.set_xlabel("Pillar Count", fontsize=13)
    ax.set_ylabel("SpMM Time (ms)", fontsize=13)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", alpha=0.35)
    save_fig("fig2_pillars_vs_spmm_time")


if __name__ == "__main__":
    main()
