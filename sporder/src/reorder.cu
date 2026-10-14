#include "common/cuda_util.cuh"
#include "reorder.h"

#include <cub/cub.cuh>

namespace sporder {

namespace {

/*
 * Gather old row lengths by new row order.  Thread `new_row` looks up its
 * source old row `new2old[new_row]` and reads that row's length.
 */
__global__ void gather_row_lengths(int32_t n,
                                   const int32_t *__restrict__ ptr_old,
                                   const int32_t *__restrict__ new2old,
                                   int32_t *__restrict__ ptr_new) {
  for (int32_t new_row = thread_in_grid(); new_row < n;
       new_row += total_threads_in_grid()) {
    int32_t src_row = new2old[new_row];
    ptr_new[new_row + 1] = ptr_old[src_row + 1] - ptr_old[src_row];
  }
}

/*
 * Gather rows into output order.  One warp per new row `new_row`: lane 0
 * looks up the source old row via new2old[new_row] (broadcast), then the
 * full warp strides through that row's contiguous columns in ind_old
 * (coalesced reads) and writes consecutively to ind_new (coalesced
 * writes).  col_map, when non-null, remaps old column indices to new
 * column space.  d_val_old/d_val_new, when non-null, copy values
 * alongside the columns.
 */
__global__ void gather_rows(int32_t n, const int32_t *__restrict__ ptr_old,
                            const int32_t *__restrict__ ind_old,
                            const float *__restrict__ val_old,
                            const int32_t *__restrict__ new2old,
                            const int32_t *__restrict__ col_map,
                            const int32_t *__restrict__ ptr_new,
                            int32_t *__restrict__ ind_new,
                            float *__restrict__ val_new) {
  for (int32_t new_row = warp_in_grid(); new_row < n;
       new_row += total_warps_in_grid()) {
    int32_t src_row = (lane_in_warp() == 0) ? new2old[new_row] : 0;
    src_row = __shfl_sync(0xFFFFFFFFu, src_row, 0);
    int32_t src = ptr_old[src_row], src_end = ptr_old[src_row + 1];
    int32_t dst = ptr_new[new_row];
    for (int32_t p = src + lane_in_warp(); p < src_end; p += 32) {
      int32_t col = ind_old[p];
      float v = val_old ? val_old[p] : 0.0f;
      int32_t offset = dst + (p - src);
      ind_new[offset] = col_map ? col_map[col] : col;
      if (val_new) {
        val_new[offset] = v;
      }
    }
  }
}

/*
 * Invert a permutation: dst[src[i]] = i for i in 0..n-1.
 */
__global__ void invert_perm(int32_t n, const int32_t *__restrict__ src,
                            int32_t *__restrict__ dst) {
  for (int32_t i = thread_in_grid(); i < n; i += total_threads_in_grid()) {
    dst[src[i]] = i;
  }
}

} // namespace

void reorder(context &ctx, int32_t n, int32_t nnz, const int32_t *d_ptr_old,
             const int32_t *d_ind_old, const float *d_val_old,
             const int32_t *d_new_order, sporder_reorder_mode_t mode,
             int32_t *d_ptr_new, int32_t *d_ind_new, float *d_val_new) {
  bool symmetric = mode == SPORDER_REORDER_SYMMETRIC;
  bool with_vals = d_val_old != nullptr && d_val_new != nullptr;
  cudaStream_t stream = ctx.stream();

  int32_t block_dim = 256;
  int32_t row_grid =
      ctx.grid_dim_by_block_dim(block_dim, 4, ceil_div(n, block_dim));
  int32_t warps_per_block = block_dim / 32;
  int32_t warp_grid =
      ctx.grid_dim_by_block_dim(block_dim, 4, ceil_div(n, warps_per_block));

  ctx.device_memset_async(d_ptr_new, 0, sizeof(*d_ptr_new)); // ptr_new[0] = 0

  gather_row_lengths<<<row_grid, block_dim, 0, stream>>>(
      n, d_ptr_old, d_new_order, d_ptr_new);
  ctx.check_last_cuda_error();

  size_t temp_bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d_ptr_new, d_ptr_new,
                                n + 1, stream);
  auto d_temp = ctx.device_alloc_async(temp_bytes);
  cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_ptr_new, d_ptr_new, n + 1,
                                stream);
  ctx.device_free_async(d_temp);

  if (!symmetric) {
    gather_rows<<<warp_grid, block_dim, 0, stream>>>(
        n, d_ptr_old, d_ind_old, d_val_old, d_new_order, nullptr, d_ptr_new,
        d_ind_new, d_val_new);
    ctx.check_last_cuda_error();
    return;
  }

  // Symmetric mode needs old2new to remap columns.  Invert new2old.
  auto d_old2new = ctx.device_alloc_async<int32_t>(n);
  invert_perm<<<row_grid, block_dim, 0, stream>>>(n, d_new_order, d_old2new);
  ctx.check_last_cuda_error();

  auto d_unsorted = ctx.device_alloc_async<int32_t>(nnz);
  auto d_val_unsorted =
      with_vals ? ctx.device_alloc_async<float>(nnz) : nullptr;

  gather_rows<<<warp_grid, block_dim, 0, stream>>>(
      n, d_ptr_old, d_ind_old, d_val_old, d_new_order, d_old2new, d_ptr_new,
      d_unsorted, d_val_unsorted);
  ctx.check_last_cuda_error();

  if (with_vals) {
    temp_bytes = 0;
    cub::DeviceSegmentedSort::SortPairs(
        nullptr, temp_bytes, d_unsorted, d_ind_new, d_val_unsorted, d_val_new,
        nnz, n, d_ptr_new, d_ptr_new + 1, stream);
    d_temp = ctx.device_alloc_async(temp_bytes);
    cub::DeviceSegmentedSort::SortPairs(
        d_temp, temp_bytes, d_unsorted, d_ind_new, d_val_unsorted, d_val_new,
        nnz, n, d_ptr_new, d_ptr_new + 1, stream);
  } else {
    temp_bytes = 0;
    cub::DeviceSegmentedSort::SortKeys(nullptr, temp_bytes, d_unsorted,
                                       d_ind_new, nnz, n, d_ptr_new,
                                       d_ptr_new + 1, stream);
    d_temp = ctx.device_alloc_async(temp_bytes);
    cub::DeviceSegmentedSort::SortKeys(d_temp, temp_bytes, d_unsorted,
                                       d_ind_new, nnz, n, d_ptr_new,
                                       d_ptr_new + 1, stream);
  }
  ctx.device_free_async(d_temp);

  ctx.device_free_async(d_unsorted);
  if (d_val_unsorted) {
    ctx.device_free_async(d_val_unsorted);
  }
  ctx.device_free_async(d_old2new);
}

} // namespace sporder
