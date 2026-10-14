# Vendored SpMM method sources

Method sources live under `SpMM_methods/` and ship with this repository.
They include small **orchestration-only** changes for unified benchmarking; core SpMM kernels and format-conversion algorithms are unchanged.

| Directory | Upstream | Bench-specific changes |
|-----------|----------|------------------------|
| `AccSpMM/` | AccSpMM | `mma_tf32.cu`: `acc_spmm_prepare` / `acc_spmm_run` / `acc_spmm_release` |
| `DTC-SpMM_ASPLOS24/` | DTC-SpMM | `DTCSpMM_kernel.cu`: `EXE_TIME=1`, return `kernel_ms`, no CSV write |
| `FlashSparse/` | FlashSparse | `benchmark.cpp`: `prepare_tf32_map` / `run_tf32_map`; `spmmKernel.cu`: remove dead warmup block |
| `Voltrix-SpMM/` | Voltrix-SpMM | none in source (runtime profiling patch in Python wrapper) |
| `clusterwise-spgemm-main/` | [PASSIONLab/clusterwise-spgemm](https://github.com/PASSIONLab/clusterwise-spgemm) | used for `gustavson-spgemm` via `RowSpGEMM` in `hash_mult.h` (no source edits) |
