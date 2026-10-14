#include "common/chunk_similarity_top_k.h"
#include "common/cuda_util.cuh"
#include "common/error_checking.h"
#include "common/warp_spacesaving.cuh"
#include <cuda_runtime.h>

namespace sporder {

namespace {

// Scalar binary search: first index in [lo, hi) where arr[idx] >= target.
// Used only for columns with deg_c > 512, where log d beats d/32.
__device__ int32_t lower_bound(const int32_t *arr, int32_t lo, int32_t hi,
                               int32_t target) {
  while (lo < hi) {
    int32_t mid = lo + (hi - lo) / 2;
    if (arr[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

} // namespace

// ---- multi-chunk kernel (single launch) -----------------------------

template <int M>
__global__ void multi_chunk_similarity_kernel(
    int32_t n, int32_t k, int32_t chunk_size,
    const int32_t *__restrict__ row_ptr, const int32_t *__restrict__ col_ind,
    const int32_t *__restrict__ col_ptr, const int32_t *__restrict__ row_ind,
    weighted_index *sim) {

  warp_spacesaving<M> store;

  for (int32_t row = warp_in_grid(); row < n; row += total_warps_in_grid()) {
    int32_t col_ind_start = row_ptr[row], col_ind_end = row_ptr[row + 1];
    int32_t row_size = col_ind_end - col_ind_start;
    if (row_size == 0) {
      continue;
    }

    int32_t chunk_start = (row / chunk_size) * chunk_size;
    int32_t chunk_end = chunk_start + chunk_size;
    if (chunk_end > n) {
      chunk_end = n;
    }

    store.clear();
    for (int32_t p = 0; p < row_size; ++p) {
      int32_t col = col_ind[col_ind_start + p];
      int32_t cstart = col_ptr[col], cend = col_ptr[col + 1];
      int32_t deg_c = cend - cstart;

      int32_t base;
      if (deg_c > 512) {
        base = lower_bound(row_ind, cstart, cend, chunk_start);
      } else {
        base = cstart;
      }

      for (; base < cend; base += 32) {
        int32_t q = base + lane_in_warp();
        int32_t cand = -1;
        float alpha = 0.0f;
        if (q < cend) {
          int32_t other = row_ind[q];
          if (other >= chunk_start && other < chunk_end && other != row) {
            int32_t sz = row_ptr[other + 1] - row_ptr[other];
            cand = other;
            alpha = 2.0f / (row_size + sz);
          }
        }
        store.add(cand, alpha);

        bool past = (q >= cend || row_ind[q] >= chunk_end);
        if (__all_sync(0xFFFFFFFFu, past)) {
          break;
        }
      }
    }

    store.select_top_k(k, sim + static_cast<size_t>(row) * k);
  }
}

template <int M>
static void
launch_multi_chunk(context &ctx, int32_t n, int32_t k, int32_t chunk_size,
                   int32_t num_chunks, const int32_t *d_ptr,
                   const int32_t *d_ind, const int32_t *d_col_ptr,
                   const int32_t *d_row_ind, weighted_index *d_neighbors) {
  auto kernel = multi_chunk_similarity_kernel<M>;

  auto occ = ctx.max_potential_block_size(kernel);
  int32_t block_dim = (occ.block_dim / 32) * 32;
  if (block_dim < 32) {
    block_dim = 32;
  }

  int32_t blocks = ctx.grid_dim_by_block_dim(block_dim, 2);

  kernel<<<blocks, block_dim, 0, ctx.stream()>>>(
      n, k, chunk_size, d_ptr, d_ind, d_col_ptr, d_row_ind, d_neighbors);
  ctx.check_last_cuda_error();
}

template <int LO, int HI>
static void dispatch_multi_by_m(int32_t m, context &ctx, int32_t n, int32_t k,
                                int32_t chunk_size, int32_t num_chunks,
                                const int32_t *d_ptr, const int32_t *d_ind,
                                const int32_t *d_col_ptr,
                                const int32_t *d_row_ind,
                                weighted_index *d_neighbors) {
  static_assert(LO % 32 == 0 && HI % 32 == 0 && LO <= HI,
                "range must be a multiple of 32");
  if constexpr (LO >= HI) {
    launch_multi_chunk<LO>(ctx, n, k, chunk_size, num_chunks, d_ptr, d_ind,
                           d_col_ptr, d_row_ind, d_neighbors);
  } else {
    constexpr int steps = (HI - LO) / 32;
    constexpr int MID = LO + (steps / 2) * 32;
    if (m <= MID) {
      dispatch_multi_by_m<LO, MID>(m, ctx, n, k, chunk_size, num_chunks, d_ptr,
                                   d_ind, d_col_ptr, d_row_ind, d_neighbors);
    } else {
      dispatch_multi_by_m<MID + 32, HI>(m, ctx, n, k, chunk_size, num_chunks,
                                        d_ptr, d_ind, d_col_ptr, d_row_ind,
                                        d_neighbors);
    }
  }
}

void multi_chunk_similarity_top_k(context &ctx, int32_t n, int32_t k,
                                  int32_t chunk_size, float oversampling,
                                  const int32_t *d_ptr, const int32_t *d_ind,
                                  const int32_t *d_col_ptr,
                                  const int32_t *d_row_ind,
                                  weighted_index *d_neighbors) {
  expect(k > 0, "expect k > 0");
  expect(oversampling >= 1, "expect oversampling >= 1");
  expect(chunk_size >= 1, "expect chunk_size >= 1");
  int32_t m = static_cast<int32_t>(k * oversampling);
  expect(m <= 512, "expect k * oversampling <= 512");

  int32_t num_chunks = ceil_div(n, chunk_size);
  ctx.device_memset_async(d_neighbors, 0, sizeof(*d_neighbors) * n * k);

  dispatch_multi_by_m<32, 512>(m, ctx, n, k, chunk_size, num_chunks, d_ptr,
                               d_ind, d_col_ptr, d_row_ind, d_neighbors);
}

} // namespace sporder
