#pragma once
#include "common/context.h"
#include <cstdint>
#include <sporder.h>

namespace sporder::cg {

/*
 * chunk_greedy -- one-shot chunked greedy seriation.
 *
 * Computes the Dice-similarity top-k graph, partitions rows into
 * fixed-size chunks (chunk_size >= 1), and runs a warp-cooperative
 * nearest-neighbour greedy chain inside each chunk (one warp per
 * chunk, visited bitmap in shared memory).  The chain order is
 * returned as a new2old permutation in d_new_order (length n),
 * suitable for reorder().
 *
 * Synchronizing: waits on the context stream before returning;
 * d_new_order, timing, and stats are valid when the call returns.
 * stats fields k_used, m_used, and chunk_size_used are left
 * untouched (the C bridge fills them).
 */
void chunk_greedy(context &ctx, int32_t n, int32_t nnz, const int32_t *d_ptr,
                  const int32_t *d_ind, bool is_symmetric, int32_t k,
                  float oversampling, int32_t chunk_size, int32_t *d_new_order,
                  sporder_cg_timing_t *timing, sporder_cg_stats_t *stats);

} // namespace sporder::cg
