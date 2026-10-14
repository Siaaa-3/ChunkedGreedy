# Third-party code

Shipped in this repository, under their own licenses:

| Code | Location | License |
|------|----------|---------|
| Groot (`sparse.py` only) | `bench/reorder/Reordering_methods/Groot-EuroSys25/` | Apache-2.0 |
| clusterwise-spgemm | `bench/reorder/Reordering_methods/clusterwise-spgemm/` | BSD (LBNL / Tokyo Tech) |
| AccSpMM | `bench/spmm/SpMM_methods/AccSpMM/` | MIT |
| Voltrix-SpMM | `bench/spmm/SpMM_methods/Voltrix-SpMM/` | MIT |

`bench/spmm/SpMM_methods/VERSIONS.md` lists the orchestration-only changes
made for benchmarking (timing hooks; kernels are unchanged).

Fetched by `fetch.sh` (no license that permits redistribution), at the commits
pinned in `sources.txt`, then patched with `patches/<name>.patch`:

| Code | Used for | Patch |
|------|----------|-------|
| Acc_spmm_reorder (`order/`: AccReorder, Rabbit Order) | AccReorder, rabbit-local-greedy | none |
| DTC-SpMM_ASPLOS24 | DTC-SpMM kernel | timing hooks in `DTCSpMM_kernel.cu` |
| FlashSparse | FlashSparse kernel | `prepare`/`run` entry points in `benchmark.cpp` |

```bash
bash third_party/fetch.sh               # all
bash third_party/fetch.sh FlashSparse   # one entry
```
