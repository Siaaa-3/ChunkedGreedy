#include "common/cuda_util.cuh"
#include "count_pillars.h"

#include <bit>
#include <cub/cub.cuh>

namespace sporder {

namespace {

__global__ void build_pillar_keys(int32_t n, int32_t height,
                                  const int32_t *__restrict__ ptr,
                                  const int32_t *__restrict__ ind,
                                  uint64_t *__restrict__ keys) {
  for (int32_t row = warp_in_grid(); row < n; row += total_warps_in_grid()) {
    uint64_t window = static_cast<uint32_t>(row / height);
    int32_t start = ptr[row], end = ptr[row + 1];
    for (int32_t p = start + lane_in_warp(); p < end; p += 32) {
      keys[p] = (window << 32) | static_cast<uint32_t>(ind[p]);
    }
  }
}

__global__ void count_unique(int32_t nnz, const uint64_t *__restrict__ keys,
                             int32_t *__restrict__ count) {
  int32_t thread_count = 0;
  for (int32_t e = thread_in_grid(); e < nnz; e += total_threads_in_grid()) {
    thread_count += (e == 0 || keys[e] != keys[e - 1]);
  }
  thread_count = warp_sum(thread_count);
  if (lane_in_warp() == 0 && thread_count > 0) {
    atomicAdd(count, thread_count);
  }
}

} // namespace

int32_t count_pillars(context &ctx, int32_t n, int32_t nnz,
                      const int32_t *d_ptr, const int32_t *d_ind,
                      int32_t height) {
  if (nnz == 0) {
    return 0;
  }
  cudaStream_t stream = ctx.stream();

  auto d_keys = ctx.device_alloc_async<uint64_t>(nnz);
  auto d_keys_sorted = ctx.device_alloc_async<uint64_t>(nnz);
  auto d_count = ctx.device_alloc_async<int32_t>();
  ctx.device_memset_async(d_count, 0, sizeof(*d_count));

  int32_t block_dim = 256;
  int32_t warps_per_block = block_dim / 32;
  int32_t warp_grid =
      ctx.grid_dim_by_block_dim(block_dim, 4, ceil_div(n, warps_per_block));
  build_pillar_keys<<<warp_grid, block_dim, 0, stream>>>(n, height, d_ptr,
                                                         d_ind, d_keys);
  ctx.check_last_cuda_error();

  // Keys are (window << 32) | column; sorting only the bits actually
  // used by the window index cuts radix passes.
  int32_t num_windows = ceil_div(n, height);
  int end_bit = 32 + std::bit_width(static_cast<uint32_t>(num_windows - 1));

  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, d_keys, d_keys_sorted,
                                 nnz, 0, end_bit, stream);
  auto d_temp = ctx.device_alloc_async(temp_bytes);
  cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys, d_keys_sorted, nnz,
                                 0, end_bit, stream);
  ctx.device_free_async(d_temp);

  int32_t elem_grid =
      ctx.grid_dim_by_block_dim(block_dim, 4, ceil_div(nnz, block_dim));
  count_unique<<<elem_grid, block_dim, 0, stream>>>(nnz, d_keys_sorted,
                                                    d_count);
  ctx.check_last_cuda_error();

  int32_t count = 0;
  ctx.memcpy_device_to_host_async(&count, d_count);
  ctx.device_free_async(d_keys);
  ctx.device_free_async(d_keys_sorted);
  ctx.device_free_async(d_count);
  ctx.wait();
  return count;
}

} // namespace sporder
