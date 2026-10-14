"""Fig. 4: amortization at w=256.

(a) mean amortization per SpMM kernel, on matrices with finite amortization for every
    kernel and method;
(b) per-matrix amortization with FlashSparse. Matrices are ordered by CG's amortization;
    matrices where CG does not amortize are spread over the middle half of the axis.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CG, KERNELS, MAIN_METHODS, METHOD_COLORS, METHOD_LABELS, SPMM_LABELS, apply_paper_style,
                    load_amort, matrix_set, save_fig, save_table)

WIDTH = 256
MARKERS = {"acc": "o", "dtc-tca": "s", "groot-py": "^", "clusterwise-hier": "D", "sporder-csg": "v"}


def fmt_amort(v):
    if v >= 1e6:
        return f"{v / 1e6:.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:.1f}k"
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"


def joint_finite(a):
    common = None
    for k in KERNELS:
        piv = a[a.spmm_method == k].pivot_table(index="matrix", columns="reorder_method",
                                                values="amortization", aggfunc="first")
        mats = set(piv[MAIN_METHODS].replace([np.inf, -np.inf], np.nan).dropna().index)
        common = mats if common is None else common & mats
    return sorted(common)


def fig4a(a):
    joint = joint_finite(a)
    means = {m: [] for m in MAIN_METHODS}
    rows = []
    for k in KERNELS:
        piv = a[(a.spmm_method == k) & a.matrix.isin(joint)].pivot_table(
            index="matrix", columns="reorder_method", values="amortization", aggfunc="first")
        for m in MAIN_METHODS:
            means[m].append(float(piv[m].mean()))
            rows.append(dict(kernel=k, method=m, n=len(joint), mean=means[m][-1]))
    save_table(pd.DataFrame(rows), "fig4a_mean_amortization")
    print(f"Fig. 4a: joint finite matrices n={len(joint)}")

    apply_paper_style(18)
    fig, ax = plt.subplots(figsize=(11.0, 6.8))
    x = np.arange(len(KERNELS))
    width = 0.14
    offsets = (np.arange(len(MAIN_METHODS)) - (len(MAIN_METHODS) - 1) / 2.0) * width
    for off, m in zip(offsets, MAIN_METHODS):
        bars = ax.bar(x + off, means[m], width=width * 0.95, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        for bar, h in zip(bars, means[m]):
            if np.isfinite(h) and h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.0, h * (10 ** 0.04), fmt_amort(h), ha="center",
                        va="bottom", fontsize=18, rotation=90, color="#222222", clip_on=False)
    ax.set_yscale("log")
    vals = [v for vs in means.values() for v in vs if np.isfinite(v) and v > 0]
    ax.set_ylim(bottom=min(vals) / 1.2, top=max(vals) * 100.0)
    ax.set_xticks(x)
    ax.set_xticklabels([SPMM_LABELS[k] for k in KERNELS], rotation=20, ha="right")
    ax.set_xlabel("SpMM Method")
    ax.set_ylabel("Mean Amortization")
    ax.legend(fontsize=18, ncol=3, loc="upper left")
    save_fig("fig4a_mean_amortization")


def matrix_order(sub):
    piv = sub.pivot_table(index="matrix", columns="reorder_method", values="amortization", aggfunc="first")
    for m in MAIN_METHODS:
        if m not in piv.columns:
            piv[m] = np.nan
    cg = piv[CG]
    cg_finite = list(cg[np.isfinite(cg)].sort_values().index)
    mins = []
    for mat in piv.index.difference(cg_finite):
        vals = piv.loc[mat, MAIN_METHODS].to_numpy(dtype=float)
        fin = vals[np.isfinite(vals)]
        if len(fin):
            mins.append((float(fin.min()), mat))
    inf_mats = [m for _, m in sorted(mins)]
    if not inf_mats or not cg_finite:
        return cg_finite or inf_mats
    n = len(cg_finite)
    lo, hi = n // 4, max(n // 4 + 1, (3 * n) // 4)
    span, used, insert_at = hi - lo, set(), []
    for i in range(len(inf_mats)):
        pos = lo + int(round((i + 1) * span / (len(inf_mats) + 1)))
        while pos in used and pos < n:
            pos += 1
        if pos >= n:
            pos = n - 1
            while pos in used and pos > lo:
                pos -= 1
        used.add(pos)
        insert_at.append(pos)
    order = list(cg_finite)
    for pos, mat in sorted(zip(insert_at, inf_mats), key=lambda t: t[0], reverse=True):
        order.insert(pos, mat)
    return order


def fig4b(a, kernel="flashsparse"):
    sub = a[a.spmm_method == kernel]
    order = matrix_order(sub)
    points = sub.pivot_table(index="matrix", columns="reorder_method", values="amortization", aggfunc="first")
    points = points.reindex(order)[MAIN_METHODS].rename_axis(index="matrix", columns=None)
    save_table(points.reset_index(), f"fig4b_points_{kernel}")
    apply_paper_style(17)
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    x_index = {mat: i for i, mat in enumerate(order)}
    counts = {}
    for m in MAIN_METHODS:
        s = sub[sub.reorder_method == m].set_index("matrix").amortization
        pts = [(x_index[mat], float(v)) for mat, v in s.items() if mat in x_index and np.isfinite(v)]
        counts[m] = len(pts)
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=18, alpha=0.85, marker=MARKERS[m], color=METHOD_COLORS[m], label=METHOD_LABELS[m])
    ax.set_yscale("log")
    ax.set_xlabel("Matrices")
    ax.set_ylabel("Amortization")
    ax.set_xticks([])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=14, frameon=False,
              columnspacing=0.6, handletextpad=0.25, borderaxespad=0.0, borderpad=0.0)
    save_fig(f"fig4b_amortization_{kernel}")
    print(f"Fig. 4b ({kernel}): finite points per method {counts}")


def main():
    a = load_amort(methods=MAIN_METHODS, matrices=matrix_set("final"))
    a = a[a.feature_dim == WIDTH]
    fig4a(a)
    fig4b(a)


if __name__ == "__main__":
    main()
