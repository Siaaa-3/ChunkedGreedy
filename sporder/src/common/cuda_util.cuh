#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#ifdef __CUDACC__
#include <cuda/std/functional>
#endif

namespace sporder {

__host__ __device__ __forceinline__ constexpr int32_t ceil_div(int32_t a,
                                                               int32_t b) {
  return (a + b - 1) / b;
}

#ifdef __CUDACC__
__device__ __forceinline__ int32_t warp_in_block() { return threadIdx.x >> 5; }

__device__ __forceinline__ int32_t total_warps_in_block() {
  return blockDim.x >> 5;
}

__device__ __forceinline__ int32_t lane_in_warp() { return threadIdx.x & 0x1F; }

__device__ __forceinline__ int32_t thread_in_grid() {
  return threadIdx.x + blockIdx.x * blockDim.x;
}

__device__ __forceinline__ int32_t total_threads_in_grid() {
  return blockDim.x * gridDim.x;
}

__device__ __forceinline__ int32_t warp_in_grid() {
  return blockIdx.x * total_warps_in_block() + warp_in_block();
}

__device__ __forceinline__ int32_t total_warps_in_grid() {
  return gridDim.x * total_warps_in_block();
}

template <typename T>
__device__ __forceinline__ T warp_sum(T value) {
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T warp_max(T value) {
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1) {
    T other = __shfl_down_sync(0xFFFFFFFFu, value, offset);
    value = value < other ? other : value;
  }
  return __shfl_sync(0xFFFFFFFFu, value, 0);
}

/*
 * warp_arg_reduce -- warp-wide argument reduction over (key, value) pairs.
 *
 * Selects the value winning under cmp (ties broken by the smaller
 * key) and returns its key, broadcast to all 32 lanes.  All lanes
 * must participate.
 */
template <typename Comparator>
__device__ __forceinline__ int32_t warp_arg_reduce(int32_t key, float value,
                                                   Comparator cmp) {
#pragma unroll
  for (int32_t offset = 16; offset > 0; offset >>= 1) {
    int32_t other_key = __shfl_down_sync(0xFFFFFFFFu, key, offset);
    float other_value = __shfl_down_sync(0xFFFFFFFFu, value, offset);
    if (cmp(other_value, value) || (other_value == value && other_key < key)) {
      key = other_key;
      value = other_value;
    }
  }
  return __shfl_sync(0xFFFFFFFFu, key, 0);
}

__device__ __forceinline__ int32_t warp_arg_min(int32_t key, float value) {
  return warp_arg_reduce(key, value, cuda::std::less<>());
}

__device__ __forceinline__ int32_t warp_arg_max(int32_t key, float value) {
  return warp_arg_reduce(key, value, cuda::std::greater<>());
}
#endif

} // namespace sporder
