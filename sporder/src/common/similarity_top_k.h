#pragma once
#include "common/context.h"
#include "common/weighted_index.h"
#include <cstdint>

namespace sporder {

/*
 * similarity_top_k -- top-k Dice-similar rows for every row.
 *
 * For each row r, finds up to k candidate rows s maximizing the Dice
 * similarity of their column sets, 2|r&s| / (|r|+|s|), approximated
 * with a warp-level space-saving sketch of capacity k * oversampling
 * (<= 512).  d_neighbors[r*k + j] holds the j-th best (index, weight)
 * pair in descending weight order; unused entries are (-1, 0).
 *
 * Parameters:
 *   d_ptr, d_ind -- CSR pattern (original)
 *   d_col_ptr, d_row_ind -- CSC pattern (transposed; pass same as
 *     CSR for symmetric patterns -- no copy needed)
 *
 * Stream-ordered: enqueues onto the context stream and returns without
 * synchronizing; d_neighbors is ready after the stream is synced.
 */
void similarity_top_k(context &ctx, int32_t n, int32_t k, const int32_t *d_ptr,
                      const int32_t *d_ind, const int32_t *d_col_ptr,
                      const int32_t *d_row_ind, float oversampling,
                      weighted_index *d_neighbors);

/*
 * similarity_top_k_stats -- diagnostics over a top-k neighbour table.
 *
 * Accumulates (atomically, so pre-zero the outputs) the total degree
 * of the top-k graph (number of valid entries), the total weight of
 * its edges, and the number of degree-zero rows (isolated: no
 * candidates at all).  Valid entries form a prefix of each row of
 * d_neighbors.
 *
 * Stream-ordered: enqueues onto the context stream and returns
 * without synchronizing.
 */
void similarity_top_k_stats(context &ctx, int32_t n, int32_t k,
                            const weighted_index *d_neighbors,
                            unsigned long long *d_total_degree,
                            double *d_total_weight,
                            int32_t *d_zero_degree_rows);

} // namespace sporder
