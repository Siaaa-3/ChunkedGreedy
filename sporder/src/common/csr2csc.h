#pragma once
#include "common/context.h"
#include <cstdint>

namespace sporder {

/*
 * csr2csc -- convert CSR representation to CSC representation
 *
 * Transposes the sparse-matrix storage layout: the row-pointer and
 * column-index arrays in CSR become column-pointer and row-index arrays
 * in CSC.  For a square n x n matrix with nnz nonzeros:
 *
 *   Input:  d_row_ptr[n+1], d_col_ind[nnz]
 *   Output: d_col_ptr[n+1], d_row_ind[nnz]
 *
 * Parameters:
 *   ctx        - CUDA context
 *   n          - number of rows / columns (square matrix)
 *   nnz        - number of nonzeros
 *   d_row_ptr  - device, CSR row pointers, length n+1
 *   d_col_ind  - device, CSR column indices, length nnz
 *   d_col_ptr  - device, output CSC column pointers, length n+1
 *   d_row_ind  - device, output CSC row indices, length nnz
 *
 * Stream-ordered: enqueues onto the context stream and returns without
 * synchronizing; outputs are ready after the stream is synced.
 */
void csr2csc(context &ctx, int32_t n, int32_t nnz, const int32_t *d_row_ptr,
             const int32_t *d_col_ind, int32_t *d_col_ptr, int32_t *d_row_ind);

/*
 * resolved_csc -- CSC access to a pattern, produced by resolve_csc().
 * Read-only pointers; owned tells release_csc() whether they were
 * allocated (transpose built) or alias the caller's CSR input.
 */
struct resolved_csc {
  const int32_t *d_ptr;
  const int32_t *d_ind;
  bool owned;
};

/*
 * resolve_csc -- obtain CSC pointers, building the transpose if needed.
 * For symmetric patterns the result aliases the CSR input (no
 * allocation).  For non-symmetric a CSC is allocated and computed.
 * Call release_csc() when the result is no longer needed.
 */
inline resolved_csc resolve_csc(context &ctx, int32_t n, int32_t nnz,
                                const int32_t *d_ptr, const int32_t *d_ind,
                                bool is_symmetric) {
  if (is_symmetric) {
    return {.d_ptr = d_ptr, .d_ind = d_ind, .owned = false};
  }
  auto *cp = ctx.device_alloc_async<int32_t>(n + 1);
  auto *ri = ctx.device_alloc_async<int32_t>(nnz);
  csr2csc(ctx, n, nnz, d_ptr, d_ind, cp, ri);
  return {.d_ptr = cp, .d_ind = ri, .owned = true};
}

/*
 * release_csc -- free the CSC buffers if resolve_csc allocated them.
 */
inline void release_csc(context &ctx, const resolved_csc &csc) {
  if (csc.owned) {
    ctx.device_free_async(const_cast<int32_t *>(csc.d_ptr));
    ctx.device_free_async(const_cast<int32_t *>(csc.d_ind));
  }
}

} // namespace sporder
