#include "common/cuda_util.cuh"
#include "common/error_checking.h"
#include "common/similarity_top_k.h"
#include "common/warp_spacesaving.cuh"
#include <algorithm>
#include <cuda_runtime.h>

namespace sporder {

template <int M>
__global__ void similarity_top_k_kernel(int32_t n, int32_t k,
                                        const int32_t *__restrict__ row_ptr,
                                        const int32_t *__restrict__ col_ind,
                                        const int32_t *__restrict__ col_ptr,
                                        const int32_t *__restrict__ row_ind,
                                        weighted_index *sim) {
  warp_spacesaving<M> store;

  for (int32_t row = warp_in_grid(); row < n; row += total_warps_in_grid()) {
    int32_t col_ind_start = row_ptr[row], col_ind_end = row_ptr[row + 1];
    int32_t row_size = col_ind_end - col_ind_start;
    if (row_size == 0) {
      continue;
    }

    // 1: all lanes count intersections in parallel
    store.clear();
    for (int32_t p = 0; p < row_size; ++p) {
      int32_t col = col_ind[col_ind_start + p];
      int32_t cstart = col_ptr[col], cend = col_ptr[col + 1];

      for (int32_t base = cstart; base < cend; base += 32) {
        int32_t q = base + lane_in_warp();
        int32_t cand = -1;
        float alpha = 0.0f;
        if (q < cend) {
          int32_t other = row_ind[q];
          if (other != row) {
            int32_t sz = row_ptr[other + 1] - row_ptr[other];
            cand = other;
            alpha = 2.0f / (row_size + sz);
          }
        }
        store.add(cand, alpha); // all 32 lanes always participate
      }
    }

    // 2: compute dice from pre-counted weighted intersections
    store.select_top_k(k, sim + static_cast<size_t>(row) * k);
  }
}

template <int M>
static void launch_similarity_top_k(context &ctx, int32_t n, int32_t k,
                                    const int32_t *d_ptr, const int32_t *d_ind,
                                    const int32_t *d_col_ptr,
                                    const int32_t *d_row_ind,
                                    weighted_index *d_neighbors) {
  auto kernel = similarity_top_k_kernel<M>;

  // Occupancy API picks block size for THIS instantiation's register footprint.
  auto occ = ctx.max_potential_block_size(kernel);
  int32_t block_dim = (occ.block_dim / 32) * 32;
  if (block_dim < 32) {
    block_dim = 32;
  }

  int blocks_per_sm = ctx.max_active_blocks_per_sm(kernel, block_dim);

  int32_t warps_per_block = block_dim / 32;
  int32_t grid_dim = ctx.grid_dim_by_blocks_per_sm(
      blocks_per_sm, 4, ceil_div(n, warps_per_block));

  kernel<<<grid_dim, block_dim, 0, ctx.stream()>>>(
      n, k, d_ptr, d_ind, d_col_ptr, d_row_ind, d_neighbors);
  ctx.check_last_cuda_error();
}

template <int LO, int HI>
static void dispatch_by_m(int32_t m, context &ctx, int32_t n, int32_t k,
                          const int32_t *d_ptr, const int32_t *d_ind,
                          const int32_t *d_col_ptr, const int32_t *d_row_ind,
                          weighted_index *d_neighbors) {
  static_assert(LO % 32 == 0 && HI % 32 == 0 && LO <= HI,
                "range must be a multiple of 32");

  if constexpr (LO >= HI) {
    launch_similarity_top_k<LO>(ctx, n, k, d_ptr, d_ind, d_col_ptr, d_row_ind,
                                d_neighbors);
  } else {
    constexpr int steps = (HI - LO) / 32;
    constexpr int MID = LO + (steps / 2) * 32;
    if (m <= MID) {
      dispatch_by_m<LO, MID>(m, ctx, n, k, d_ptr, d_ind, d_col_ptr, d_row_ind,
                             d_neighbors);
    } else {
      dispatch_by_m<MID + 32, HI>(m, ctx, n, k, d_ptr, d_ind, d_col_ptr,
                                  d_row_ind, d_neighbors);
    }
  }
}

void similarity_top_k(context &ctx, int32_t n, int32_t k, const int32_t *d_ptr,
                      const int32_t *d_ind, const int32_t *d_col_ptr,
                      const int32_t *d_row_ind, float oversampling,
                      weighted_index *d_neighbors) {
  expect(k > 0, "expect k > 0");
  expect(oversampling >= 1, "expect oversampling >= 1");
  int32_t m = static_cast<int32_t>(k * oversampling);
  expect(m <= 512, "expect k * oversampling <= 512");

  ctx.device_memset_async(d_neighbors, 0, sizeof(*d_neighbors) * n * k);

  dispatch_by_m<32, 512>(m, ctx, n, k, d_ptr, d_ind, d_col_ptr, d_row_ind,
                         d_neighbors);
}

namespace {

__global__ void count_top_k_degree(
    int32_t n, int32_t k, const weighted_index *__restrict__ neighbors,
    unsigned long long *__restrict__ total_degree,
    double *__restrict__ total_weight, int32_t *__restrict__ zero_degree_rows) {
  unsigned long long thread_degree = 0;
  double thread_weight = 0.0;
  int32_t thread_zero_degree = 0;
  for (int32_t row = thread_in_grid(); row < n;
       row += total_threads_in_grid()) {
    const weighted_index *row_neighbors =
        neighbors + static_cast<size_t>(row) * k;
    int32_t degree = 0;
    float row_weight = 0.0f;
    for (int32_t j = 0; j < k; ++j) {
      weighted_index entry = row_neighbors[j];
      if (entry.index() == -1) {
        break; // valid entries form a prefix
      }
      degree += 1;
      row_weight += entry.weight();
    }
    thread_degree += degree;
    thread_weight += row_weight;
    thread_zero_degree += (degree == 0);
  }
  thread_degree = warp_sum(thread_degree);
  thread_weight = warp_sum(thread_weight);
  thread_zero_degree = warp_sum(thread_zero_degree);
  if (lane_in_warp() == 0) {
    if (thread_degree > 0) {
      atomicAdd(total_degree, thread_degree);
      atomicAdd(total_weight, thread_weight);
    }
    if (thread_zero_degree > 0) {
      atomicAdd(zero_degree_rows, thread_zero_degree);
    }
  }
}

} // namespace

void similarity_top_k_stats(context &ctx, int32_t n, int32_t k,
                            const weighted_index *d_neighbors,
                            unsigned long long *d_total_degree,
                            double *d_total_weight,
                            int32_t *d_zero_degree_rows) {
  int32_t block_dim = 256;
  int32_t grid_dim =
      ctx.grid_dim_by_block_dim(block_dim, 4, ceil_div(n, block_dim));
  count_top_k_degree<<<grid_dim, block_dim, 0, ctx.stream()>>>(
      n, k, d_neighbors, d_total_degree, d_total_weight, d_zero_degree_rows);
  ctx.check_last_cuda_error();
}

} // namespace sporder
