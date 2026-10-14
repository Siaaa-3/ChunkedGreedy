#pragma once
#include "common/context.h"
#include <sporder.h>

namespace sporder::hm {

/*
 * hierarchical_matching -- compute a similarity-grouping row permutation.
 *
 * Builds a binary cluster hierarchy over the rows of the CSR pattern:
 * a Dice-similarity top-k graph is computed once at the leaves
 * (similarity_top_k), then each of the num_levels levels runs suitor
 * matching between super-nodes (leaf_driven_matching) and contracts
 * matched pairs into parents (advance_level).  The DFS order of the
 * resulting tree is written to d_new_order: row i of the input should
 * become row d_new_order[i], which places similar rows in the same
 * aligned window of 2^num_levels rows.
 *
 * Parameters:
 *   ctx          - CUDA context providing the stream
 *   n            - number of rows (and columns; the pattern is square)
 *   nnz          - total number of nonzeros (must equal d_ptr[n])
 *   d_ptr        - device, CSR row pointers, length n+1
 *   d_ind        - device, CSR column indices, length nnz
 *   is_symmetric - pattern equals its transpose (skips the csr2csc)
 *   num_levels   - hierarchy levels, in [1, 30]
 *   k            - similarity candidates kept per row
 *   oversampling - sketch capacity factor, k * oversampling <= 512
 *   max_rounds   - suitor round limit per level (<= 0 = to convergence)
 *   d_new_order    - device, output permutation of 0..n-1, length n
 *   timing       - optional stage timings, NULL to skip
 *   stats        - optional diagnostics, NULL to skip
 *
 * Synchronizing: waits on the context stream before returning (also
 * once per suitor round for loop control); d_new_order, timing, and
 * stats are valid when the call returns.
 */
void hierarchical_matching(context &ctx, int32_t n, int32_t nnz,
                           const int32_t *d_ptr, const int32_t *d_ind,
                           bool is_symmetric, int32_t num_levels, int32_t k,
                           float oversampling, int32_t max_rounds,
                           int32_t *d_new_order, sporder_hm_timing_t *timing,
                           sporder_hm_stats_t *stats);

} // namespace sporder::hm
