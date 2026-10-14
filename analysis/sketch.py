"""Space-Saving sketch: (i) counters per row vs pillars and reorder time on the final set;
(ii) recall of the exact top-k Dice neighbors on 62 sampled matrices."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import METHOD_COLORS, RESULTS, apply_paper_style, matrix_set, save_fig, save_table

# M = round_up(gamma * k, 32) with k = 16
VARIANTS = [("gamma=1", "32"), ("gamma=4", "64"), ("gamma=8", "128"), ("gamma=32", "512"),
            ("default", "default (k=nnz/n, M=4k)")]


def gm(x):
    return float(np.exp(np.log(np.asarray(x, float)).mean()))


def main():
    s = pd.read_csv(RESULTS / "sketch_accuracy.csv")
    s = s[s.matrix.isin(matrix_set("final"))]
    sw = s[s.part == "sweep"]
    ref = sw[sw.variant == "gamma=32"].dropna(subset=["pillars"]).set_index("matrix")
    o = sw[sw.variant == "ORIGIN"].set_index("matrix")
    rows = []
    for v, M in VARIANTS:
        x = sw[sw.variant == v].dropna(subset=["pillars"]).set_index("matrix")
        c = x.index.intersection(ref.index)
        r = x.loc[c].pillars / ref.loc[c].pillars
        rows.append(dict(variant=v, M=M, n=len(c), pillar_vs_M512_gm=gm(r), pillar_vs_M512_max=r.max(),
                         within_1pct=(abs(r - 1) <= 0.01).mean(), pillar_reduction_gm=gm(o.loc[c].pillars / x.loc[c].pillars),
                         time_ms_gm=gm(x.loc[c].wall_ms), time_ms_median=x.loc[c].wall_ms.median()))
    T = pd.DataFrame(rows)
    save_table(T, "sketch_sweep")

    ac = s[s.part == "accuracy"]
    ex = ac.drop_duplicates("matrix")
    rows = []
    for v in ["gamma=1", "gamma=4", "gamma=8"]:
        x = ac[ac.variant == v]
        ok = x[x.exact_ms.notna()]
        rows.append(dict(variant=v, n=len(x), exact_failed=int(ex.exact_ms.isna().sum()),
                         set_recall_mean=x.set_recall.mean(), weight_recall_mean=x.weight_recall.mean(),
                         sketch_ms_gm=gm(x.sketch_ms), exact_ms_gm_on_ok=gm(ok.exact_ms),
                         sketch_ms_gm_on_ok=gm(ok.sketch_ms), speedup_gm=gm(ok.exact_ms / ok.sketch_ms)))
    A = pd.DataFrame(rows)
    save_table(A, "sketch_accuracy")
    print(T.round(4).to_string(index=False))
    print(A.round(3).to_string(index=False))

    apply_paper_style()
    labels, rel, tm = [], [], []
    for v, M in VARIANTS[:4]:
        x = sw[sw.variant == v].set_index("matrix")
        c = x.index.intersection(ref.index)
        labels.append(f"M={M}")
        rel.append(gm(x.loc[c].pillars / ref.loc[c].pillars))
        tm.append(gm(x.loc[c].wall_ms))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    ax1.plot(labels, rel, "o-", color=METHOD_COLORS["sporder-csg"])
    ax1.set_ylabel("pillars / pillars(M=512)\n(geomean)")
    ax1.axhline(1, color="k", lw=0.6, ls="--")
    ax2.plot(labels, tm, "o-", color=METHOD_COLORS["sporder-csg"])
    ax2.set_ylabel("reorder time (ms, geomean)")
    for ax in (ax1, ax2):
        ax.set_xlabel("Space-Saving counters per row (k=16)")
    fig.tight_layout()
    save_fig("fig_sketch_sweep")


if __name__ == "__main__":
    main()
