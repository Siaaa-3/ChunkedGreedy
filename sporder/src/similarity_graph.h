#pragma once
#include "common/context.h"
#include <cstdint>
#include <sporder.h>

namespace sporder {

void similarity_graph(context &ctx, int32_t n, int32_t nnz,
                      const int32_t *d_ptr, const int32_t *d_ind,
                      bool is_symmetric, int32_t k, float oversampling,
                      int32_t *d_indices, float *d_weights,
                      sporder_sim_timing_t *timing, sporder_sim_stats_t *stats);

} // namespace sporder
