#pragma once
#include "common/context.h"
#include <cstdint>
#include <sporder.h>

namespace sporder::csg {

void chunk_similarity_greedy(context &ctx, int32_t n, int32_t nnz,
                             const int32_t *d_ptr, const int32_t *d_ind,
                             bool is_symmetric, int32_t k, float oversampling,
                             int32_t chunk_size, int32_t *d_new_order,
                             sporder_csg_timing_t *timing,
                             sporder_csg_stats_t *stats);

} // namespace sporder::csg
