"""Fig. 6: reorder quality (original pillars / reordered pillars) vs reorder time.

One point per (matrix, method); stars mark the per-method medians.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from common import (MAIN_METHODS, METHOD_COLORS, METHOD_LABELS, apply_paper_style, geo_mean, load_quality,
                    load_reorder_times, matrix_set, save_fig, save_table)


def main():
    mats = matrix_set("final")
    t = load_reorder_times(methods=MAIN_METHODS, matrices=mats).pivot_table(
        index="matrix", columns="reorder_method", values="reorder_time_ms", aggfunc="first")
    q = load_quality(orders=["ORIGIN", *MAIN_METHODS], matrices=mats).pivot(index="matrix", columns="order",
                                                                            values="pillars")
    cloud, summary = [], []
    for m in MAIN_METHODS:
        j = pd.concat([q[m].rename("p"), q["ORIGIN"].rename("p0"), t[m].rename("t")], axis=1).dropna()
        ratio = j.p0 / j.p
        cloud.append(pd.DataFrame(dict(matrix=j.index, method=m, reorder_time_ms=j.t, quality=ratio)))
        summary.append(dict(method=m, label=METHOD_LABELS[m], n=len(j), median_reorder_ms=j.t.median(),
                            median_quality=ratio.median(), geo_mean_quality=geo_mean(ratio)))
    cloud = pd.concat(cloud, ignore_index=True)
    summary = pd.DataFrame(summary)
    save_table(cloud, "fig6_points")
    save_table(summary, "fig6_summary")
    print(summary.to_string(index=False))

    apply_paper_style(16)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for m in MAIN_METHODS:
        d = cloud[cloud.method == m]
        ax.scatter(d.reorder_time_ms, d.quality, s=12, alpha=0.45, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
    for _, r in summary.iterrows():
        ax.scatter(r.median_reorder_ms, r.median_quality, s=280, marker="*", edgecolors="black", linewidths=1.0,
                   color=METHOD_COLORS[r.method], zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reorder Time (ms)")
    ax.set_ylabel("Reorder Quality")
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_yticks([0.3, 0.5, 1, 2, 3, 5, 8])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(MAIN_METHODS), fontsize=13, markerscale=1.4,
              frameon=False, columnspacing=1.0, handletextpad=0.35)
    save_fig("fig6_quality_vs_time")


if __name__ == "__main__":
    main()
