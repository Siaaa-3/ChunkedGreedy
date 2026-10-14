#include "common/csr2csc.h"
#include "common/cuda_util.cuh"
#include "common/similarity_top_k.h"
#include "common/timer.h"
#include "common/weighted_index.h"
#include "similarity_graph.h"
#include <chrono>

namespace sporder {

namespace {

__global__ void unpack_weighted_index(int32_t n, int32_t k,
                                      const weighted_index *__restrict__ sim,
                                      int32_t *__restrict__ indices,
                                      float *__restrict__ weights) {
  for (int32_t idx = thread_in_grid(); idx < n * k;
       idx += total_threads_in_grid()) {
    weighted_index wi = sim[idx];
    indices[idx] = wi.index();
    weights[idx] = wi.weight();
  }
}

} // namespace

void similarity_graph(context &ctx, int32_t n, int32_t nnz,
                      const int32_t *d_ptr, const int32_t *d_ind,
                      bool is_symmetric, int32_t k, float oversampling,

                      int32_t *d_indices, float *d_weights,
                      sporder_sim_timing_t *timing,
                      sporder_sim_stats_t *stats) {
  auto t0 = std::chrono::steady_clock::now();
  cudaStream_t stream = ctx.stream();

  auto d_sim =
      ctx.device_alloc_async<weighted_index>(static_cast<size_t>(n) * k);
  ctx.device_memset_async(d_sim, 0,
                          static_cast<size_t>(n) * k * sizeof(*d_sim));

  cuda_timer t_csr2csc(stream);
  t_csr2csc.begin();
  resolved_csc csc = resolve_csc(ctx, n, nnz, d_ptr, d_ind, is_symmetric);
  t_csr2csc.end();

  cuda_timer t_sim(stream);
  t_sim.begin();
  similarity_top_k(ctx, n, k, d_ptr, d_ind, csc.d_ptr, csc.d_ind, oversampling,
                   d_sim);
  t_sim.end();

  auto d_total_degree =
      stats ? ctx.device_alloc_async<unsigned long long>() : nullptr;
  auto d_total_weight = stats ? ctx.device_alloc_async<double>() : nullptr;
  auto d_zero_degree_rows = stats ? ctx.device_alloc_async<int32_t>() : nullptr;

  if (stats) {
    ctx.device_memset_async(d_total_degree, 0, sizeof(*d_total_degree));
    ctx.device_memset_async(d_total_weight, 0, sizeof(*d_total_weight));
    ctx.device_memset_async(d_zero_degree_rows, 0, sizeof(*d_zero_degree_rows));
    similarity_top_k_stats(ctx, n, k, d_sim, d_total_degree, d_total_weight,
                           d_zero_degree_rows);
  }

  int32_t bdim = 256;
  int32_t grid = ctx.grid_dim_by_block_dim(
      bdim, 4, ceil_div(static_cast<int64_t>(n) * k, bdim));

  cuda_timer t_unpack(stream);
  t_unpack.begin();
  unpack_weighted_index<<<grid, bdim, 0, stream>>>(n, k, d_sim, d_indices,
                                                   d_weights);
  ctx.check_last_cuda_error();
  t_unpack.end();

  ctx.device_free_async(d_sim);
  release_csc(ctx, csc);

  ctx.wait();

  if (timing) {
    timing->csr2csc_ms = t_csr2csc.elapsed_ms();
    timing->similarity_ms = t_sim.elapsed_ms();
    timing->unpack_ms = t_unpack.elapsed_ms();
    timing->wall_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
  }

  if (stats) {
    unsigned long long h_total_degree = 0;
    double h_total_weight = 0.0;
    int32_t h_zero_degree_rows = 0;
    ctx.memcpy_device_to_host_async(&h_total_degree, d_total_degree);
    ctx.memcpy_device_to_host_async(&h_total_weight, d_total_weight);
    ctx.memcpy_device_to_host_async(&h_zero_degree_rows, d_zero_degree_rows);
    // Already waited above; D2H copies are done
    stats->avg_degree = n > 0 ? static_cast<double>(h_total_degree) / n : 0.0;
    stats->avg_edge_weight =
        h_total_degree > 0
            ? h_total_weight / static_cast<double>(h_total_degree)
            : 0.0;
    stats->zero_degree_rows = h_zero_degree_rows;

    ctx.device_free_async(d_total_degree);
    ctx.device_free_async(d_total_weight);
    ctx.device_free_async(d_zero_degree_rows);
  }
}

} // namespace sporder
