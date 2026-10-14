#pragma once

#include "common/cuda_util.cuh"
#include "common/error_checking.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <source_location>

namespace sporder {

/*
 * context -- CUDA execution context owning a stream, cuSPARSE handle,
 *           and cached device properties.
 *
 * Each context wraps a cudaStream_t and provides stream-ordered memory
 * allocation (cudaMallocAsync / cudaFreeAsync).  The stream, allocator,
 * and cuSPARSE handle are tied to the currently active device; the
 * constructor queries that device with cudaGetDevice.
 *
 * Cached device properties -- queried once at construction:
 *   num_sms              cudaDevAttrMultiProcessorCount
 *   max_threads_per_sm   cudaDevAttrMaxThreadsPerMultiProcessor
 *
 * Grid-dimension helpers:
 *
 *   grid_dim_by_block_dim(block_dim, waves=4, ub=0)
 *     Thread-saturating grid.  Launches enough blocks to fill all SMs
 *     waves times at the given block size.  Optionally capped by ub
 *     (ceil_div(n, block_dim)) for element-coverage kernels.
 *
 *   grid_dim_by_blocks_per_sm(blocks_per_sm, waves=4, ub=0)
 *     Explicit-occupancy grid.  For SMEM- or register-bound kernels
 *     where blocks_per_sm (e.g. 1) is known.  Also accepts ub.
 *
 * Occupancy helpers (thin wrappers over the CUDA occupancy API):
 *
 *   max_potential_block_size(kernel, dynamic_smem_size=0,
 *                            block_size_limit=0)
 *     cudaOccupancyMaxPotentialBlockSize.  Returns
 *     {min_grid_dim, block_dim} maximizing occupancy.
 *
 *   max_active_blocks_per_sm(kernel, block_dim, dynamic_smem_size=0)
 *     cudaOccupancyMaxActiveBlocksPerMultiprocessor.  Returns the
 *     number of resident blocks per SM at the given block size.
 *
 * Utility:
 *   ceil_div(a, b) -- __host__ __device__ constexpr integer ceiling
 *                     division, e.g. ceil_div(n, block_dim).
 *
 * Not copyable, not movable.  Destructor destroys the stream and
 * cuSPARSE handle.
 *
 * Usage:
 *   context ctx;
 *   int32_t *buf = ctx.device_alloc_async<int32_t>(n);
 *
 *   int32_t block_dim = 256;
 *   int32_t grid_dim  = ctx.grid_dim_by_block_dim(
 *       block_dim, 4, ceil_div(n, block_dim));
 *   kernel<<<grid_dim, block_dim, 0, ctx.stream()>>>(buf);
 *
 *   ctx.device_free_async(buf);
 *   ctx.wait();
 */
class context {
  int device_;
  cudaStream_t stream_;
  cusparseHandle_t cusparse_handle_;
  int num_sms_;
  int max_threads_per_sm_;

  using src_loc = std::source_location;

public:
  context() {
    ensure(cudaGetDevice(&device_));
    ensure(cudaStreamCreate(&stream_));
    ensure(cusparseCreate(&cusparse_handle_));
    ensure(cusparseSetStream(cusparse_handle_, stream_));
    ensure(cudaDeviceGetAttribute(&num_sms_, cudaDevAttrMultiProcessorCount,
                                  device_));
    ensure(cudaDeviceGetAttribute(
        &max_threads_per_sm_, cudaDevAttrMaxThreadsPerMultiProcessor, device_));
  }

  ~context() {
    ensure(cudaSetDevice(device_));
    ensure(cusparseDestroy(cusparse_handle_));
    ensure(cudaStreamDestroy(stream_));
  }

  cudaStream_t stream() const noexcept { return stream_; }
  cusparseHandle_t cusparse_handle() const noexcept { return cusparse_handle_; }
  int num_sms() const noexcept { return num_sms_; }
  int max_threads_per_sm() const noexcept { return max_threads_per_sm_; }

  [[nodiscard]] int32_t grid_dim_by_block_dim(int32_t block_dim,
                                              int32_t waves = 4,
                                              int32_t ub = 0) const {
    int32_t grid =
        static_cast<int32_t>(static_cast<int64_t>(num_sms_) *
                             max_threads_per_sm_ * waves / block_dim);
    return ub > 0 ? std::min(grid, ub) : grid;
  }

  [[nodiscard]] int32_t grid_dim_by_blocks_per_sm(int32_t blocks_per_sm,
                                                  int32_t waves = 4,
                                                  int32_t ub = 0) const {
    int32_t grid = num_sms_ * blocks_per_sm * waves;
    return ub > 0 ? std::min(grid, ub) : grid;
  }

  struct occupancy_block_size {
    int min_grid_dim;
    int block_dim;
  };

  template <typename Kernel>
  [[nodiscard]] occupancy_block_size
  max_potential_block_size(Kernel kernel, size_t dynamic_smem_size = 0,
                           int block_size_limit = 0,
                           src_loc loc = src_loc::current()) const {
    occupancy_block_size result{};
    ensure(cudaOccupancyMaxPotentialBlockSize(
               &result.min_grid_dim, &result.block_dim, kernel,
               dynamic_smem_size, block_size_limit),
           loc);
    return result;
  }

  template <typename Kernel>
  [[nodiscard]] int
  max_active_blocks_per_sm(Kernel kernel, int block_dim,
                           size_t dynamic_smem_size = 0,
                           src_loc loc = src_loc::current()) const {
    int num_blocks = 0;
    ensure(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
               &num_blocks, kernel, block_dim, dynamic_smem_size),
           loc);
    return num_blocks;
  }

  context(const context &) = delete;
  context &operator=(const context &) = delete;
  context(context &&) = delete;
  context &operator=(context &&) = delete;

  void wait(src_loc loc = src_loc::current()) {
    ensure(cudaStreamSynchronize(stream_), loc);
  }

  void check_last_cuda_error(src_loc loc = src_loc::current()) {
    ensure(cudaGetLastError(), loc);
  }

  void device_memset_async(void *ptr, int value, size_t count,
                           src_loc loc = src_loc::current()) {
    ensure(cudaMemsetAsync(ptr, value, count, stream_), loc);
  }

  template <typename T>
  void memcpy_device_to_host_async(T *dst, const T *src, size_t count = 1,
                                   src_loc loc = src_loc::current()) {
    ensure(cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost,
                           stream_),
           loc);
  }

  void memcpy_device_to_device_async(void *dst, const void *src, size_t bytes,
                                     src_loc loc = src_loc::current()) {
    ensure(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream_),
           loc);
  }

  [[nodiscard]] void *device_alloc_async(size_t size,
                                         src_loc loc = src_loc::current()) {
    void *ptr = nullptr;
    if (size > 0) {
      ensure(cudaMallocAsync(&ptr, size, stream_), loc);
    }
    return ptr;
  }

  template <typename T>
  [[nodiscard]] T *device_alloc_async(size_t count = 1,
                                      src_loc loc = src_loc::current()) {
    return static_cast<T *>(device_alloc_async(count * sizeof(T), loc));
  }

  void device_free_async(void *ptr, src_loc loc = src_loc::current()) {
    if (ptr) {
      ensure(cudaFreeAsync(ptr, stream_), loc);
    }
  }
};

} // namespace sporder
