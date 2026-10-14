#include "common/csr2csc.h"
#include "common/cuda_util.cuh"
#include "common/error_checking.h"
#ifdef SPORDER_CUSPARSE_CSR2CSC
#include <cusparse.h>
#else
#include <cub/cub.cuh>
#endif

namespace sporder {

#ifndef SPORDER_CUSPARSE_CSR2CSC
namespace {

__global__ void fill_row_ids(int32_t n, const int32_t *__restrict__ row_ptr,
                             int32_t *__restrict__ row_ids) {
  for (int32_t row = thread_in_grid(); row < n;
       row += total_threads_in_grid()) {
    int32_t start = row_ptr[row];
    int32_t end = row_ptr[row + 1];
    for (int32_t j = start; j < end; ++j) {
      row_ids[j] = row;
    }
  }
}

__global__ void count_column_degrees(int32_t nnz,
                                     const int32_t *__restrict__ sorted_keys,
                                     int32_t *__restrict__ col_deg) {
  for (int32_t i = thread_in_grid(); i < nnz; i += total_threads_in_grid()) {
    atomicAdd(&col_deg[sorted_keys[i]], 1);
  }
}

} // namespace
#endif // SPORDER_CUSPARSE_CSR2CSC

void csr2csc(context &ctx, int32_t n, int32_t nnz, const int32_t *d_row_ptr,
             const int32_t *d_col_ind, int32_t *d_col_ptr, int32_t *d_row_ind) {
  cudaStream_t stream = ctx.stream();

#ifdef SPORDER_CUSPARSE_CSR2CSC
  size_t buffer_size = 0;
  ensure(cusparseCsr2cscEx2_bufferSize(
      ctx.cusparse_handle(), n, n, nnz,
      /*csrVal=*/nullptr, d_row_ptr, d_col_ind,
      /*cscVal=*/nullptr, d_col_ptr, d_row_ind, CUDA_R_32F,
      CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO,
      CUSPARSE_CSR2CSC_ALG_DEFAULT, &buffer_size));

  void *buffer = ctx.device_alloc_async(buffer_size);

  ensure(cusparseCsr2cscEx2(ctx.cusparse_handle(), n, n, nnz,
                            /*csrVal=*/nullptr, d_row_ptr, d_col_ind,
                            /*cscVal=*/nullptr, d_col_ptr, d_row_ind,
                            CUDA_R_32F, CUSPARSE_ACTION_SYMBOLIC,
                            CUSPARSE_INDEX_BASE_ZERO,
                            CUSPARSE_CSR2CSC_ALG_DEFAULT, buffer));

  ctx.device_free_async(buffer);
#else
  // 1. Copy column indices to a temporary key array (input is const).
  auto d_keys = ctx.device_alloc_async<int32_t>(nnz);
  ctx.memcpy_device_to_device_async(d_keys, d_col_ind,
                                    static_cast<size_t>(nnz) * sizeof(int32_t));

  // 2. Generate row-id values from row_ptr into d_row_ind directly.
  {
    auto occ = ctx.max_potential_block_size(fill_row_ids);
    int32_t bd = (occ.block_dim / 32) * 32;
    if (bd < 32) {
      bd = 32;
    }
    int32_t grid = ceil_div(n, bd);
    fill_row_ids<<<grid, bd, 0, stream>>>(n, d_row_ptr, d_row_ind);
    ctx.check_last_cuda_error();
  }

  // 3. Sort (col_ind, row_id) pairs by column index.
  size_t temp_bytes = 0;
  ensure(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, d_keys, d_keys,
                                         d_row_ind, d_row_ind, nnz, 0,
                                         sizeof(int32_t) * 8, stream));
  auto d_temp = ctx.device_alloc_async<char>(temp_bytes);
  ensure(cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, d_keys, d_keys,
                                         d_row_ind, d_row_ind, nnz, 0,
                                         sizeof(int32_t) * 8, stream));

  // 4. Histogram of sorted column indices → exclusive scan.
  ctx.device_memset_async(d_col_ptr, 0,
                          static_cast<size_t>(n + 1) * sizeof(int32_t));
  {
    auto occ = ctx.max_potential_block_size(count_column_degrees);
    int32_t bd = (occ.block_dim / 32) * 32;
    if (bd < 32) {
      bd = 32;
    }
    int32_t grid = ceil_div(nnz, bd);
    count_column_degrees<<<grid, bd, 0, stream>>>(nnz, d_keys, d_col_ptr);
    ctx.check_last_cuda_error();
  }

  temp_bytes = 0;
  ensure(cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_col_ptr,
                                       d_col_ptr, n + 1, stream));
  auto d_scan_temp = ctx.device_alloc_async<char>(temp_bytes);
  ensure(cub::DeviceScan::ExclusiveSum(d_scan_temp, temp_bytes, d_col_ptr,
                                       d_col_ptr, n + 1, stream));

  // 5. Clean up.
  ctx.device_free_async(d_keys);
  ctx.device_free_async(d_temp);
  ctx.device_free_async(d_scan_temp);
#endif
}

} // namespace sporder
