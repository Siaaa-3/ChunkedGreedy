"""End-to-end GCN training on the 60 undirected graphs (Table R2 and its figure).

Total time = reorder + GPU permutation + kernel preprocessing + 200 epochs. The
permutation time is measured separately with sporder's GPU permutation
(perm_time_sporder.csv).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CONFIG, MAIN_METHODS, METHOD_COLORS, METHOD_LABELS, RESULTS, SPMM_LABELS, apply_paper_style,
                    save_fig, save_table)

E = CONFIG["e2e"]["epochs"]
ORDER = ["sporder-csg", "clusterwise-hier", "acc", "dtc-tca", "groot-py"]


def gm(x):
    return float(np.exp(np.log(np.asarray(x, float)).mean()))


def derive():
    d = pd.read_csv(RESULTS / "e2e_gcn.csv")
    pt = pd.read_csv(RESULTS / "perm_time_sporder.csv").set_index("matrix").perm_ms_sporder
    d["perm_ms"] = np.where(d.order == "ORIGIN", 0.0, d.matrix.map(pt))
    d["total_ms"] = d.reorder_ms + d.perm_ms + d.prep_ms + E * d.epoch_ms
    o = d[d.order == "ORIGIN"].set_index(["matrix", "kernel"])
    d = d.join(o[["total_ms", "epoch_ms", "spmm_ms_per_epoch"]].add_suffix("_orig"), on=["matrix", "kernel"])
    d["e2e_speedup"] = d.total_ms_orig / d.total_ms
    d["epoch_speedup"] = d.epoch_ms_orig / d.epoch_ms
    d["spmm_speedup"] = d.spmm_ms_per_epoch_orig / d.spmm_ms_per_epoch
    gain = d.epoch_ms_orig - d.epoch_ms
    d["breakeven_epochs"] = np.where(gain > 0, (d.reorder_ms + d.perm_ms) / gain, np.inf)
    return d


def main():
    d = derive()
    save_table(d, "e2e_gcn_derived")
    rows = []
    for (k, m), x in d[d.order != "ORIGIN"].groupby(["kernel", "order"]):
        be = x.breakeven_epochs[np.isfinite(x.breakeven_epochs)]
        rows.append(dict(kernel=k, method=m, n=len(x), epoch_speedup_gm=gm(x.epoch_speedup),
                         e2e_speedup_gm=gm(x.e2e_speedup), e2e_faster_frac=(x.e2e_speedup > 1).mean(),
                         breakeven_within_200=(x.breakeven_epochs <= E).mean(),
                         breakeven_median=float(np.median(be)) if len(be) else np.inf,
                         reorder_ms_median=x.reorder_ms.median(), perm_ms_median=x.perm_ms.median()))
    s = pd.DataFrame(rows)
    s["method"] = pd.Categorical(s.method, ORDER)
    s = s.sort_values(["kernel", "method"])
    save_table(s, "tableR2_e2e_gcn")
    print(s.round(3).to_string(index=False))

    apply_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, k in zip(axes, CONFIG["e2e"]["kernels"]):
        x = d[d.kernel == k]
        bp = ax.boxplot([x[x.order == m].e2e_speedup.values for m in MAIN_METHODS], patch_artist=True)
        for patch, m in zip(bp["boxes"], MAIN_METHODS):
            patch.set_facecolor(METHOD_COLORS[m])
        ax.set_yscale("log")
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xticks(range(1, len(MAIN_METHODS) + 1), [METHOD_LABELS[m] for m in MAIN_METHODS], rotation=15)
        ax.set_ylabel(f"end-to-end speedup ({E} epochs)")
        ax.set_title(f"GCN training, {SPMM_LABELS[k]}")
    fig.tight_layout()
    save_fig("figR2_e2e_gcn")


if __name__ == "__main__":
    main()
