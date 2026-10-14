"""Fig. 5: reorder time vs (a) rows and (b) average degree on the final set."""
import matplotlib.pyplot as plt
import pandas as pd

from common import (CG, MAIN_METHODS, METHOD_COLORS, METHOD_LABELS, apply_paper_style, geo_mean, load_quality,
                    load_reorder_times, matrix_set, save_fig, save_table)


def scatter(common, x, xlabel, name):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m in MAIN_METHODS:
        ax.scatter(x.reindex(common.index), common[m], s=14, alpha=0.65, label=METHOD_LABELS[m], color=METHOD_COLORS[m])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Reorder Time (ms)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=13, frameon=False,
              columnspacing=0, handletextpad=0, borderaxespad=0.0, borderpad=0.0)
    save_fig(name)


def main():
    mats = matrix_set("final")
    bench = load_reorder_times(methods=MAIN_METHODS, matrices=mats)
    common = bench.pivot_table(index="matrix", columns="reorder_method", values="reorder_time_ms",
                               aggfunc="first")[MAIN_METHODS].dropna()
    stats = load_quality(orders=["ORIGIN"], matrices=mats).set_index("matrix")
    print(f"matrices with all reorder times: {len(common)}")
    points = common.rename_axis(columns=None).join(stats[["n", "nnz"]].rename(columns={"n": "rows"}))
    points["avg_degree"] = points.nnz / points.rows
    save_table(points.reset_index(), "fig5_points")

    apply_paper_style(14)
    scatter(common, stats.n, "Rows", "fig5a_reorder_time_vs_rows")
    scatter(common, stats.nnz / stats.n, "Average degree", "fig5b_reorder_time_vs_avg_degree")

    summary = []
    for m in MAIN_METHODS:
        rel = common[m] / common[CG]
        summary.append(dict(method=m, label=METHOD_LABELS[m], n=len(common), median_ms=common[m].median(),
                            mean_ms=common[m].mean(), geo_mean_rel_to_cg=geo_mean(rel),
                            median_rel_to_cg=rel.median()))
    out = pd.DataFrame(summary)
    save_table(out, "fig5_reorder_time_summary")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
