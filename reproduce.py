#!/usr/bin/env python3
"""Reproduce the paper's results.

  python reproduce.py figures                  # tables + figures from the released results/
  python reproduce.py setup                    # build extensions
  python reproduce.py download --set final     # SuiteSparse matrices -> data/
  python reproduce.py reorder  --set final     # reorder with every method (writes runs/)
  python reproduce.py spmm     --set final     # SpMM times and amortization
  python reproduce.py quality | e2e | sketch   # rebuttal experiments
  python reproduce.py collect                  # runs/ -> results/ (replaces released results)
  python reproduce.py all                      # every stage above on the paper's matrix sets

Matrices are stored under data/ (override with SPMATKIT_DATA); raw outputs go to runs/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT / "configs" / "paper.yaml").read_text())
RUNS = ROOT / "runs"
DATA = Path(os.environ.get("SPMATKIT_DATA", ROOT / "data"))
ANALYSES = ["paper_numbers", "status", "table1", "fig2_pillars_vs_spmm", "fig4_amortization", "fig5_reorder_time",
            "fig6_quality_vs_time", "fig7_ablation", "speedup_by_kernel", "quality", "e2e", "sketch"]


def env() -> dict[str, str]:
    e = dict(os.environ)
    paths = [ROOT / "spmatkit", ROOT / "bench" / "reorder", ROOT / "bench" / "spmm"]
    e["PYTHONPATH"] = os.pathsep.join([*map(str, paths), e.get("PYTHONPATH", "")])
    e["SPMATKIT_DATA"] = str(DATA)
    # thread counts enter the configuration hash of each reorder run
    e["OMP_NUM_THREADS"] = str(CONFIG["reorder"]["threads"])
    return e


def run(cmd: list, cwd: Path = ROOT) -> None:
    cmd = [str(c) for c in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env(), check=True)


def set_file(name: str) -> Path:
    return ROOT / "datasets" / CONFIG["sets"][name]


def dataset_name(name: str) -> str:
    """spmatkit looks up datasets by name under <data>/datasets/."""
    dst = DATA / "datasets" / CONFIG["sets"][name]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(set_file(name).read_text())
    return dst.stem


def stage_setup(args) -> None:
    run([sys.executable, "-m", "pip", "install", "-e", "sporder"])
    run(["bash", "scripts/build_native.sh", "all"], cwd=ROOT / "bench" / "reorder")


def stage_download(args) -> None:
    run([sys.executable, "datasets/download.py", "--list", set_file(args.set), "--data", DATA])


def stage_reorder(args) -> None:
    methods = args.methods or CONFIG["main_methods"] + ["sporder-cg"]
    timeout_h = CONFIG["reorder"]["timeout_s"] / 3600
    if args.set == "table1":
        # small matrices; the greedy baselines run without the timeout subprocess
        methods = args.methods or ["global-greedy", "rabbit-local-greedy"]
        timeout_h = -1
    params = {m: CONFIG["methods"][m]["params"] for m in methods}
    run([sys.executable, "scripts/run_bench.py", "--dataset", dataset_name(args.set), "--methods", *methods,
         "--mode", "save", "--params-json", json.dumps(params), "--timeout-h", timeout_h,
         "--output-csv", RUNS / "reorder_bench.csv"], cwd=ROOT / "bench" / "reorder")


def stage_spmm(args) -> None:
    kernels = CONFIG["spmm"]["kernels"] + CONFIG["spmm"]["extra_kernels"]
    run([sys.executable, "scripts/compute_amortization.py", "--dataset", dataset_name(args.set),
         "--reorder-methods", *CONFIG["main_methods"], "--all-param-sets", "--methods", *kernels,
         "--dims", *CONFIG["spmm"]["widths"], "--bench-csv", RUNS / "spmm_bench.csv",
         "--output-csv", RUNS / "reorder_spmm_amortization.csv"], cwd=ROOT / "bench" / "spmm")


def stage_quality(args) -> None:
    run([sys.executable, "experiments/quality_metrics.py", "--matrix-list", set_file("final"),
         "--orders", "ORIGIN", *CONFIG["main_methods"], "sporder-cg", "--out", RUNS / "quality_metrics.csv"])
    run([sys.executable, "experiments/quality_metrics.py", "--matrix-list", set_file("table1"),
         "--orders", "ORIGIN", "global-greedy", "rabbit-local-greedy", "--out", RUNS / "quality_metrics_table1.csv"])


def stage_e2e(args) -> None:
    graphs = set_file("graphs").read_text().split()
    e2e = CONFIG["e2e"]
    run([sys.executable, "experiments/e2e_gcn.py", "--matrices", *graphs, "--kernels", *e2e["kernels"],
         "--epochs", e2e["epochs"], "--hidden", e2e["hidden"], "--classes", e2e["classes"],
         "--out", RUNS / "e2e_gcn.csv"])
    run([sys.executable, "experiments/perm_time.py", set_file("graphs"), RUNS / "perm_time_sporder.csv"])


def stage_sketch(args) -> None:
    run([sys.executable, "experiments/sketch_accuracy.py", "--matrix-list", set_file("final"),
         "--accuracy-every", CONFIG["sketch"]["accuracy_every"], "--out", RUNS / "sketch_accuracy.csv"])


def stage_collect(args) -> None:
    quality = [RUNS / "quality_metrics.csv", RUNS / "quality_metrics_table1.csv"]
    run([sys.executable, "analysis/collect.py", "--quality", *quality])


def stage_figures(args) -> None:
    for name in ANALYSES:
        run([sys.executable, f"analysis/{name}.py"])
    print(f"\ntables: {ROOT / 'out' / 'tables'}\nfigures: {ROOT / 'out' / 'figures'}")


def stage_all(args) -> None:
    """Paper flow: reorder every candidate (timeouts/OOM decide the final set), SpMM on the final set."""
    stage_setup(args)
    for name in ["candidates", "table1"]:
        sub = argparse.Namespace(**{**vars(args), "set": name})
        stage_download(sub)
        stage_reorder(sub)
    stage_spmm(argparse.Namespace(**{**vars(args), "set": "final"}))
    for stage in [stage_quality, stage_e2e, stage_sketch, stage_collect, stage_figures]:
        stage(args)


def main() -> None:
    stages = {name[len("stage_"):]: fn for name, fn in globals().items() if name.startswith("stage_")}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=list(stages))
    ap.add_argument("--set", default="final", choices=list(CONFIG["sets"]),
                    help="matrix set for download / reorder / spmm (default: final)")
    ap.add_argument("--methods", nargs="+", help="reorder: subset of methods")
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    stages[args.stage](args)


if __name__ == "__main__":
    main()
