"""Fig. 7: CG vs CG without Stage 1 (global similarity instead of chunks): mean reorder
time and mean pillar count."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from common import METHOD_COLORS, apply_paper_style, load_quality, load_reorder_times, matrix_set, save_fig, save_table

METHODS = ["sporder-csg", "sporder-cg"]
LABELS = {"sporder-csg": "CG", "sporder-cg": "CG w/o Stage 1"}
RATIO_COLOR = "#C44E52"
FS = 16


def bracket(ax, x0, x1, y_cg, y_wo, *, logy):
    ratio = y_wo / y_cg
    y_lo, y_hi = min(y_cg, y_wo), max(y_cg, y_wo)
    x_line, half = 0.5 * (x0 + x1), 0.12
    for y in (y_lo, y_hi):
        ax.plot([x_line - half, x_line + half], [y, y], color=RATIO_COLOR, lw=1.2, clip_on=False, zorder=5)
    ax.annotate("", xy=(x_line, y_lo), xytext=(x_line, y_hi),
                arrowprops=dict(arrowstyle="<->", color=RATIO_COLOR, lw=1.2), zorder=5)
    if logy:
        y_mid = float(np.sqrt(y_lo * y_hi)) * (10 ** 0.02)
        ax.set_ylim(top=y_hi * (10 ** 0.1))
    else:
        y_mid = 0.5 * (y_lo + y_hi) + (y_hi - y_lo) * 0.7
    ax.text(x_line - half, y_mid, f"{ratio:.2f}$\\times$", ha="right", va="center", fontsize=FS,
            color=RATIO_COLOR, clip_on=False, zorder=6)


def main():
    mats = matrix_set("final")
    t = load_reorder_times(methods=METHODS, matrices=mats).pivot(index="matrix", columns="reorder_method",
                                                                 values="reorder_time_ms")
    p = load_quality(orders=METHODS, matrices=mats).pivot(index="matrix", columns="order", values="pillars")
    common = t[METHODS].dropna().index.intersection(p[METHODS].dropna().index)
    mean_t = [float(t.loc[common, m].mean()) for m in METHODS]
    mean_p = [float(p.loc[common, m].mean()) for m in METHODS]
    per_matrix_t = t.loc[common, METHODS[1]] / t.loc[common, METHODS[0]]
    summary = pd.DataFrame(dict(method=METHODS, label=[LABELS[m] for m in METHODS], n=len(common),
                                mean_reorder_ms=mean_t, mean_pillars=mean_p,
                                median_reorder_ms=[float(t.loc[common, m].median()) for m in METHODS]))
    save_table(summary, "fig7_ablation_summary")
    print(summary.to_string(index=False))
    print(f"time ratio of means {mean_t[1] / mean_t[0]:.2f}x, per-matrix median {per_matrix_t.median():.2f}x; "
          f"pillar ratio {mean_p[1] / mean_p[0]:.3f}x")

    apply_paper_style(FS)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.4))
    x = np.arange(len(METHODS))
    colors = [METHOD_COLORS[m] for m in METHODS]
    labels = [LABELS[m] for m in METHODS]

    ax = axes[0]
    ax.bar(x, mean_t, color=colors, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylabel("Mean Reorder Time (ms)")
    for xi, yi in zip(x, mean_t):
        ax.text(xi, yi * (10 ** 0.008), f"{yi:.1f}", ha="center", va="bottom", fontsize=FS, color="#222222", zorder=6)
    bracket(ax, x[0], x[1], mean_t[0], mean_t[1], logy=True)

    ax = axes[1]
    ax.bar(x, mean_p, color=colors, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Mean Pillar Count ($\times 10^{6}$)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / 1e6:.2f}"))
    ax.set_yticks([1.55e6, 1.60e6, 1.65e6, 1.70e6])
    for xi, yi in zip(x, mean_p):
        ax.text(xi, yi + 0.002e6, f"{yi / 1e6:.2f}", ha="center", va="bottom", fontsize=FS, color="#222222", zorder=6)
    bracket(ax, x[0], x[1], mean_p[0], mean_p[1], logy=False)
    ax.set_ylim(1.55e6, 1.70e6)

    fig.tight_layout()
    save_fig("fig7_ablation")


if __name__ == "__main__":
    main()
