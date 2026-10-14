# Chunked Greedy: artifact

Code, data and scripts that reproduce the tables and figures of the paper
"Chunked Greedy" (PPoPP 2027), including the experiments added in the revision.

## Quick start: tables and figures from the released results

```bash
conda env create -f environment.yml && conda activate chunked-greedy
python reproduce.py figures
```

Tables go to `out/tables/`, figures to `out/figures/`. This step only needs
pandas/matplotlib and reads the measurements in `results/`.

## Full reproduction

```bash
python reproduce.py setup                     # build extensions
python reproduce.py download --set candidates # SuiteSparse matrices -> data/
python reproduce.py reorder  --set candidates # every reordering method
python reproduce.py spmm     --set final      # SpMM times and amortization
python reproduce.py quality                   # structural quality metrics
python reproduce.py e2e                       # end-to-end GCN training
python reproduce.py sketch                    # sketch-size sweep and accuracy
python reproduce.py collect                   # runs/ -> results/
python reproduce.py figures
```

`python reproduce.py all` runs the whole sequence. All parameters (method
parameters, thread count, timeout, SpMM kernels and widths, ...) are in
`configs/paper.yaml`. The paper's numbers were measured on an NVIDIA GH200 node
(72-core Grace CPU, 32 OpenMP threads, CUDA 12.0). Reordering all candidates
with the slow baselines takes several days; `--methods sporder-csg` limits the
run to Chunked Greedy.

## Matrix sets (`datasets/`)

| File | Matrices | Use |
|------|----------|-----|
| `candidates_475.txt` | 475 | all SuiteSparse candidates; timeout/OOM rates |
| `final_300.txt` | 300 | main evaluation |
| `graphs_60.txt` | 60 | end-to-end GCN |
| `table1_8.txt` | 8 | Table 1 (global vs. local greedy) |

## Paper item -> script (`analysis/`)

| Item | Script |
|------|--------|
| Table 1 | `table1.py` |
| Fig. 2 | `fig2_pillars_vs_spmm.py` |
| Fig. 4 | `fig4_amortization.py` |
| Fig. 5 | `fig5_reorder_time.py` |
| Fig. 6 | `fig6_quality_vs_time.py` |
| Fig. 7 | `fig7_ablation.py` |
| Table 3, speedups, reorder times, fail rates, numbers in the text | `paper_numbers.py` |
| Timeout / OOM rates | `status.py` |
| SpMM speedup per kernel and width | `speedup_by_kernel.py` |
| Revision: end-to-end GCN | `e2e.py` |
| Revision: structural quality metrics | `quality.py` |
| Revision: sketch size and accuracy | `sketch.py` |

## Notes

- Fig. 2 uses all 300 final matrices. The submitted figure was drawn from the
  295 for which SpMM metadata was available at the time (it lacked NACA0015,
  gsm_106857, roadNet-CA, roadNet-PA and web-Google); on those 295 the fit is
  identical to the submitted one.
- Timeouts (1 h) and out-of-memory failures were recorded over several
  machines; `results/reorder_status.csv` is the curated list used in the paper.
## License

MIT (see `LICENSE`) for the code in this repository, except third-party code,
which keeps its own license (see `third_party/README.md`).
