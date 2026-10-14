#pragma once
#include "common/context.h"
#include <cstdint>

namespace sporder {

/*
 * count_pillars -- count nonzero columns over row windows.
 *
 * Rows are grouped into ceil(n / height) consecutive windows of
 * `height` rows (the last window may be shorter).  A pillar is a
 * column that holds at least one nonzero inside a window; the return
 * value is the total number of pillars over all windows.  This is the
 * column-tile count of a blocked SpMM with tile width 1.
 *
 * Parameters:
 *   ctx    - CUDA context providing the stream
 *   n      - number of rows
 *   nnz    - total number of nonzeros (must equal d_ptr[n])
 *   d_ptr  - device pointer, CSR row pointers, length n+1
 *   d_ind  - device pointer, CSR column indices, length nnz
 *   height - window height, >= 1
 *
 * Synchronizing: waits on the context stream (forced by the host
 * return value); the count is valid when the call returns.
 */
int32_t count_pillars(context &ctx, int32_t n, int32_t nnz,
                      const int32_t *d_ptr, const int32_t *d_ind,
                      int32_t height);

} // namespace sporder
