# Third-party code

Third-party code shipped in this repository keeps its original license.

| Code | Location | License |
|------|----------|---------|
| Groot (`sparse.py` only) | `bench/reorder/Reordering_methods/Groot-EuroSys25/` | Apache-2.0 |
| clusterwise-spgemm | `bench/reorder/Reordering_methods/clusterwise-spgemm/` | BSD (LBNL / Tokyo Tech) |
| AccSpMM | `bench/spmm/SpMM_methods/AccSpMM/` | MIT |
| Voltrix-SpMM | `bench/spmm/SpMM_methods/Voltrix-SpMM/` | MIT |
| Acc_spmm_reorder (`order/`: AccReorder, Rabbit Order) | `bench/reorder/Reordering_methods/Acc_spmm_reorder/` | none stated upstream; Rabbit Order (c) NTT |
| DTC-SpMM ([HPMLL/DTC-SpMM_ASPLOS24](https://github.com/HPMLL/DTC-SpMM_ASPLOS24) @ `66eca26`) | `bench/spmm/SpMM_methods/DTC-SpMM_ASPLOS24/` | none stated upstream |
| FlashSparse ([ParCIS/FlashSparse](https://github.com/ParCIS/FlashSparse) @ `1687646`) | `bench/spmm/SpMM_methods/FlashSparse/` | none stated upstream |

Code without a stated license is included only so that the paper's results can
be reproduced; all rights remain with its authors.
`bench/spmm/SpMM_methods/VERSIONS.md` lists the orchestration-only changes made
for benchmarking (timing hooks; kernels are unchanged).
