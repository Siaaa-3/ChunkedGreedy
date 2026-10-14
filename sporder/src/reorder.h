#pragma once
#include "common/context.h"
#include <cstdint>
#include <sporder.h>

namespace sporder {

/*
 * reorder -- apply a row (or symmetric row+column) permutation to a
 * CSR sparsity pattern, optionally with values.
 *
 * Modes:
 *   SPORDER_REORDER_ROWS       new[nr][j]    = old[new2old[nr]][j]
 *   SPORDER_REORDER_SYMMETRIC  new[nr1][nr2] = old[new2old[nr1]][new2old[nr2]]
 *
 * d_new_order must be a permutation of 0..n-1: the output's row nr
 * contains the input's row new_order[nr].  When d_val_old is non-NULL,
 * values are reordered alongside the column indices.  Assuming the
 * input has sorted column indices within each row, so does the output:
 * row mode copies rows verbatim, symmetric mode re-sorts each row
 * after remapping the column indices.
 *
 * Parameters:
 *   ctx       - CUDA context providing the stream
 *   n         - number of rows (and columns; the pattern is square)
 *   nnz       - total number of nonzeros (must equal d_ptr_old[n])
 *   d_ptr_old - device pointer, input CSR row pointers, length n+1
 *   d_ind_old - device pointer, input CSR column indices, length nnz
 *   d_val_old - device pointer, input values, length nnz (NULL if none)
 *   d_new_order - device pointer, permutation, length n (new2old)
 *   mode      - which sides of the pattern to permute
 *   d_ptr_new - device pointer, output CSR row pointers, length n+1
 *   d_ind_new - device pointer, output CSR column indices, length nnz
 *   d_val_new - device pointer, output values, length nnz (NULL if none)
 *
 * Stream-ordered: enqueues onto the context stream and returns without
 * synchronizing; outputs are ready after the stream is synced.
 */
void reorder(context &ctx, int32_t n, int32_t nnz, const int32_t *d_ptr_old,
             const int32_t *d_ind_old, const float *d_val_old,
             const int32_t *d_new_order, sporder_reorder_mode_t mode,
             int32_t *d_ptr_new, int32_t *d_ind_new, float *d_val_new);

} // namespace sporder
