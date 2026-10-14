"""Table 1: pillar count after global greedy vs community-local greedy."""
from common import load_quality, matrix_set, save_table


def main():
    mats = matrix_set("table1")
    q = load_quality(orders=["ORIGIN", "global-greedy", "rabbit-local-greedy"], matrices=mats)
    t = q.pivot(index="matrix", columns="order", values="pillars").loc[mats]
    t = t[["ORIGIN", "global-greedy", "rabbit-local-greedy"]]
    t["local_vs_global_pct"] = (t["rabbit-local-greedy"] / t["global-greedy"] - 1) * 100
    t.insert(0, "rows", q[q.order == "ORIGIN"].set_index("matrix").loc[mats, "n"])
    save_table(t.round(2), "table1_global_vs_local", index=True)
    print(t.round(2).to_string())


if __name__ == "__main__":
    main()
