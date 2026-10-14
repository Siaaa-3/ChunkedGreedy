"""Sec. 5.3: timeout / OOM rates of each method on the candidate matrices."""
import pandas as pd

from common import MAIN_METHODS, METHOD_LABELS, RESULTS, matrix_set, save_table


def main():
    st = pd.read_csv(RESULTS / "reorder_status.csv")
    n = len(matrix_set("candidates"))
    rates = st.groupby("method")[["timeout", "oom"]].sum().reindex(MAIN_METHODS[::-1], fill_value=0)
    rates["timeout_pct"] = 100 * rates.timeout / n
    rates["oom_pct"] = 100 * rates.oom / n
    rates.insert(0, "label", [METHOD_LABELS[m] for m in rates.index])
    save_table(rates.round(2), "timeout_rates", index=True)
    print(f"candidates: {n}")
    print(rates.round(2).to_string())


if __name__ == "__main__":
    main()
