#pragma once
#include "common/context.h"
#include <cstdint>

namespace sporder {
float bench_cusparse_spmm(context &ctx, int32_t n, int32_t nnz,
                          const int32_t *d_ptr, const int32_t *d_ind,
                          int32_t cols, int32_t warmup, int32_t repeat,
                          uint64_t seed);
}
