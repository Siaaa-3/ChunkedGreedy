"""Numbers quoted in the text, Table 3 and the Fig. 4 statistics.

Outputs (out/tables/):
  reorder_time.csv            reorder time per method (final set; and each method on its
                              completed candidates)
  speedup_overall.csv         SpMM speedup over 4 kernels x 4 widths, per method
  speedup_kernel_w.csv        SpMM speedup per method x kernel x width
  fail_rate.csv               share of matrices with no speedup, per method x kernel
  table3_not_sped_up.csv      Table 3 (submitted layout and the common-set layout)
  amort_w{w}.csv              Fig. 4a: amortization on the joint-positive set
  amort_own_positive_w256.csv Fig. 4b / Sec. 5.2: each method over its own positive matrices
  cg_misc.csv                 single CG numbers quoted in the text
"""
import numpy as np
import pandas as pd

from common import CG, KERNELS, MAIN_METHODS, TABLES, WIDTHS, load_amort, load_reorder_times, matrix_set, save_table

M = MAIN_METHODS
K = KERNELS
KB = ["acc-spmm-balance" if k == "acc-spmm" else k for k in K]
W = WIDTHS


def gm(x):
    x = np.asarray(x, float)
    return float(np.exp(np.log(x).mean()))


def desc(x, p=""):
    x = np.asarray(x, float)
    return {p + "n": len(x), p + "median": np.median(x), p + "geomean": gm(x), p + "mean": x.mean(),
            p + "p25": np.percentile(x, 25), p + "p75": np.percentile(x, 75), p + "max": x.max()}


def reorder_time(S, S475):
    t = load_reorder_times(methods=M).pivot(index="matrix", columns="reorder_method", values="reorder_time_ms")
    out = []
    f = t.loc[S, M]
    assert f.notna().all().all() and (f < 3.6e6).all().all()
    for m in M:
        d = dict(version="final (300)", method=m, **desc(f[m]))
        if m != CG:
            pr = f[m] / f[CG]
            d.update(per_matrix_ratio_median=pr.median(), per_matrix_ratio_geomean=gm(pr),
                     per_matrix_ratio_mean=pr.mean(), ratio_of_medians=f[m].median() / f[CG].median(),
                     ratio_of_geomeans=gm(f[m]) / gm(f[CG]), ratio_of_means=f[m].mean() / f[CG].mean())
        out.append(d)
    e = t[t.index.isin(S475)]
    for m in M:
        x = e[m].dropna()
        x = x[x < 3.6e6]
        d = dict(version="candidates (475, each method on its own completed matrices)", method=m, **desc(x))
        if m != CG:
            d.update(ratio_of_medians=x.median() / e[CG].dropna().median(),
                     ratio_of_geomeans=gm(x) / gm(e[CG].dropna()))
        out.append(d)
    save_table(pd.DataFrame(out), "reorder_time")
    return f


def spmm_common(a):
    b = a[a.spmm_method.isin(K)]
    cnt = b.groupby("matrix").size()
    return sorted(cnt[cnt == len(M) * len(K) * len(W)].index)


def speedup(a, C, S):
    out, per = [], []
    for variant, KK in [("Acc-SpMM default (non-balanced)", K), ("Acc-SpMM load-balanced", KB)]:
        for sset, mats in [(f"common SpMM set ({len(C)})", C), (f"final ({len(S)})", S)]:
            b = a[a.spmm_method.isin(KK) & a.matrix.isin(mats)]
            for m in M:
                x = b[b.reorder_method == m].speedup
                if len(x) != len(mats) * len(KK) * len(W):
                    continue
                out.append(dict(acc_spmm_variant=variant, matrix_set=sset, method=m, **desc(x),
                                frac_improved=(x > 1).mean(), median_speedup_when_not_improved=x[x <= 1].median()))
    b = a[a.matrix.isin(C)]
    for (m, k, w), g in b.groupby(["reorder_method", "spmm_method", "feature_dim"]):
        per.append(dict(method=m, kernel=k, w=w, **desc(g.speedup), frac_improved=(g.speedup > 1).mean()))
    save_table(pd.DataFrame(out), "speedup_overall")
    save_table(pd.DataFrame(per), "speedup_kernel_w")


def fail_rate(a, C):
    rows = []
    for sset in ["common", "own"]:
        for m in M:
            d = dict(method=m, matrix_set=f"common SpMM set ({len(C)})" if sset == "common"
                     else "each method on all its measured matrices", n=0)
            for k in K + ["acc-spmm-balance"]:
                g = a[(a.reorder_method == m) & (a.spmm_method == k)]
                if sset == "common":
                    g = g[g.matrix.isin(C)]
                d["n"] = max(d["n"], g.matrix.nunique())
                d[k] = 100 * np.mean([(g[g.feature_dim == w].speedup <= 1).mean() for w in W])
            rows.append(d)
    f = pd.DataFrame(rows)
    save_table(f, "fail_rate")
    # submitted: baselines on the common set, CG on its own matrices; revision: common set for all
    common = f[f.matrix_set.str.startswith("common")].set_index("method").loc[M, K]
    own = f[f.matrix_set.str.startswith("each method")].set_index("method").loc[M, K]
    submitted = common.copy()
    submitted.loc[CG] = own.loc[CG]
    t3 = pd.concat([submitted.assign(version="submitted"), common.assign(version=f"revision (common {len(C)})")])
    save_table(t3.round(1), "table3_not_sped_up", index=True)


def amort(a, w):
    b = a[a.spmm_method.isin(K) & (a.feature_dim == w)]
    b = b[np.isfinite(b.amortization) & (b.amortization > 0)]
    cnt = b.groupby("matrix").size()
    d = b[b.matrix.isin(cnt[cnt == len(K) * len(M)].index)]
    rows = []
    for k in K:
        f = d[d.spmm_method == k].pivot(index="matrix", columns="reorder_method", values="amortization")
        for m in M:
            r = dict(w=w, kernel=k, method=m, **desc(f[m]))
            if m != CG:
                pr = f[m] / f[CG]
                r.update(ratio_of_means=f[m].mean() / f[CG].mean(), ratio_of_medians=f[m].median() / f[CG].median(),
                         per_matrix_ratio_geomean=gm(pr), per_matrix_ratio_median=pr.median())
            rows.append(r)
    save_table(pd.DataFrame(rows), f"amort_w{w}")


def amort_own(a, w):
    b = a[a.spmm_method.isin(K) & (a.feature_dim == w)]
    b = b[np.isfinite(b.amortization) & (b.amortization > 0)]
    rows = []
    for k in K:
        g = b[b.spmm_method == k]
        cg_mean = g[g.reorder_method == CG].amortization.mean()
        for m in M:
            x = g[g.reorder_method == m].amortization
            rows.append(dict(kernel=k, method=m, n=len(x), mean=x.mean(), median=x.median(), geomean=gm(x),
                             mean_ratio_vs_CG=x.mean() / cg_mean))
    save_table(pd.DataFrame(rows), f"amort_own_positive_w{w}")


def cg_misc(a, t):
    rows = []
    c = a[(a.reorder_method == CG) & a.spmm_method.isin(K)]
    pos = c[np.isfinite(c.amortization) & (c.amortization > 0)]
    rows.append(dict(item="CG amortization, median over positive (kernel, w, matrix)", value=pos.amortization.median(), n=len(pos)))
    for w in W:
        p = pos[pos.feature_dim == w]
        rows.append(dict(item=f"CG amortization median, w={w}, positive", value=p.amortization.median(), n=len(p)))
    for k in K:
        p = pos[pos.spmm_method == k]
        rows.append(dict(item=f"CG amortization median, {k}, positive", value=p.amortization.median(), n=len(p)))
    rows.append(dict(item="CG reorder time median (ms)", value=t[CG].median(), n=len(t)))
    rows.append(dict(item="CG reorder time geomean (ms)", value=gm(t[CG]), n=len(t)))
    rows.append(dict(item="CG reorder time mean (ms)", value=t[CG].mean(), n=len(t)))
    save_table(pd.DataFrame(rows), "cg_misc")


def main():
    S, S475 = sorted(matrix_set("final")), set(matrix_set("candidates"))
    t = reorder_time(S, S475)
    a = load_amort(methods=M, matrices=S)
    a["speedup"] = a.spmm_orig_ms / a.spmm_reord_ms
    C = spmm_common(a)
    save_table(pd.DataFrame({"matrix": C}), "spmm_common_set")
    print("common SpMM set:", len(C))
    speedup(a, C, S)
    fail_rate(a, C)
    for w in W:
        amort(a, w)
    amort_own(a, 256)
    cg_misc(a, t)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for f in ["reorder_time", "speedup_overall", "table3_not_sped_up", "amort_w256", "cg_misc"]:
        print("==", f)
        print(pd.read_csv(TABLES / f"{f}.csv").round(3).to_string())


if __name__ == "__main__":
    main()
