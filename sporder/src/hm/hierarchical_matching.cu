#include "common/context.h"
#include <sporder.h>

#include <cub/cub.cuh>

#include "common/csr2csc.h"
#include "common/cuda_util.cuh"
#include "common/similarity_top_k.h"
#include "common/weighted_index.h"
#include "hm/leaf_driven_matching.h"
#include "hm/stats_collector.h"
#include "hm/timing_collector.h"

namespace sporder::hm {

/*
 * invert_perm -- permutation inversion: dst[src[i]] = i.
 */
__global__ void invert_perm(int32_t n, const int32_t *__restrict__ src,
                            int32_t *__restrict__ dst) {
  for (int32_t i = thread_in_grid(); i < n; i += total_threads_in_grid()) {
    dst[src[i]] = i;
  }
}

/*
 * iota -- fills d_array[0..n-1] with start, start+1, ..., start+n-1.
 */
void iota(context &ctx, int32_t n, int32_t *d_array, int32_t start);

/*
 * advance_level -- advances the contraction hierarchy by one level.
 *
 * Given a matching d_partners on level_size nodes (d_partners[i] == -1
 * for unmatched; otherwise d_partners[i] is i's partner and the pair
 * is symmetric), this function:
 *
 *   1. Ranks matched pairs and unmatched singletons so that every pair
 *      gets one parent index and consecutive unmatched nodes are paired
 *      arbitrarily.  Odd-count singletons get a solo parent at the end,
 *      guaranteeing ceil(level_size / 2) parents overall.
 *
 *   2. Writes the packed level map:
 *        d_level_map[i] = (parent << 1) | side
 *      for each node i at the current level.  The side bit is 0 for
 *      left / singleton children and 1 for right children of a pair.
 *
 *   3. Propagates the accumulated DFS ordering into d_old2new for
 *      every original leaf.  d_old2new[leaf] is read-modify-written
 *      so that after all num_levels iterations it holds the final
 *      DFS permutation.
 */
void advance_level(context &ctx, int32_t level, int32_t level_size,
                   int32_t num_leaves, int32_t *d_partners,
                   int32_t *d_level_map, int32_t *d_old2new);

void hierarchical_matching(context &ctx, int32_t n, int32_t nnz,
                           const int32_t *d_ptr, const int32_t *d_ind,
                           bool is_symmetric, int32_t num_levels, int32_t k,
                           float oversampling, int32_t max_rounds,
                           int32_t *d_new_order, sporder_hm_timing_t *timing,
                           sporder_hm_stats_t *stats) {
  timing_collector timing_collect(ctx, timing, num_levels);
  stats_collector stats_collect(ctx, stats, num_levels);

  auto d_sim =
      ctx.device_alloc_async<weighted_index>(static_cast<size_t>(n) * k);
  auto d_partners = ctx.device_alloc_async<int32_t>(n);
  auto d_level_map = ctx.device_alloc_async<int32_t>(n);
  auto d_old2new = ctx.device_alloc_async<int32_t>(n);

  iota(ctx, n, d_old2new, 0); // 0, 1, ..., n-1

  timing_collect.begin_csr2csc();
  resolved_csc csc = resolve_csc(ctx, n, nnz, d_ptr, d_ind, is_symmetric);
  timing_collect.end_csr2csc();

  timing_collect.begin_similarity();
  similarity_top_k(ctx, n, k, d_ptr, d_ind, csc.d_ptr, csc.d_ind, oversampling,
                   d_sim);
  timing_collect.end_similarity();
  stats_collect.on_similarity(n, k, d_sim);

  int32_t level_size = n;
  for (int32_t level = 0; level < num_levels; ++level) {
    timing_collect.begin_matching(level);
    int32_t rounds = leaf_driven_matching(
        ctx, n, level_size, level, k, max_rounds, d_sim, d_old2new, d_partners);
    timing_collect.end_matching(level);
    stats_collect.on_matching(level, level_size, rounds, d_partners);

    timing_collect.begin_advance(level);
    advance_level(ctx, level, level_size, n, d_partners, d_level_map,
                  d_old2new);
    timing_collect.end_advance(level);

    level_size = ceil_div(level_size, 2);
  }

  ctx.device_free_async(d_partners);
  ctx.device_free_async(d_level_map);
  release_csc(ctx, csc);

  int32_t block_dim = 256;
  int32_t leaf_grid =
      ctx.grid_dim_by_block_dim(block_dim, 4, ceil_div(n, block_dim));
  invert_perm<<<leaf_grid, block_dim, 0, ctx.stream()>>>(n, d_old2new,
                                                         d_new_order);
  ctx.check_last_cuda_error();

  ctx.device_free_async(d_old2new);
  ctx.device_free_async(d_sim);

  ctx.wait();
  stats_collect.publish();
  timing_collect.publish();
}

// ==========================================================================
// advance_level
// ==========================================================================

struct match_unmatch_pair {
  int32_t matched, unmatched;
  __host__ __device__ match_unmatch_pair
  operator+(const match_unmatch_pair &b) const {
    return {matched + b.matched, unmatched + b.unmatched};
  }
};

__global__ void advance_level_count(int32_t level_size,
                                    const int32_t *__restrict__ partners,
                                    match_unmatch_pair *__restrict__ counts) {
  for (int32_t i = thread_in_grid(); i < level_size;
       i += total_threads_in_grid()) {
    int32_t p = partners[i];
    if (p == -1) {
      counts[i] = {0, 1};
    } else if (i < p) {
      counts[i] = {1, 0};
    } else {
      counts[i] = {0, 0};
    }
  }
}

__global__ void
advance_level_update_level_map(int32_t level_size,
                               const int32_t *__restrict__ partners,
                               const match_unmatch_pair *__restrict__ counts,
                               int32_t *__restrict__ level_map) {
  int32_t matched_total = counts[level_size - 1].matched;
  for (int32_t i = thread_in_grid(); i < level_size;
       i += total_threads_in_grid()) {
    int32_t p = partners[i];
    if (p == -1) {
      int32_t r = counts[i].unmatched;
      level_map[i] = 2 * matched_total + r - 1;
    } else if (i < p) {
      int32_t r = counts[i].matched;
      int32_t parent = r - 1;
      level_map[i] = (parent << 1) | 0;
      level_map[p] = (parent << 1) | 1;
    }
  }
}

__global__ void
advance_level_update_old2new(int32_t num_leaves, int32_t level,
                             const int32_t *__restrict__ level_map,
                             int32_t *__restrict__ old2new) {
  for (int32_t leaf = thread_in_grid(); leaf < num_leaves;
       leaf += total_threads_in_grid()) {
    int32_t cur_root_sides = old2new[leaf];
    int32_t cur_root = cur_root_sides >> level;
    int32_t cur_sides = cur_root_sides & ((1 << level) - 1);

    int32_t next_root_side = level_map[cur_root];
    int32_t next_root = next_root_side >> 1;
    int32_t next_side = next_root_side & 1;

    old2new[leaf] =
        (next_root << (level + 1)) | (next_side << level) | cur_sides;
  }
}

void advance_level(context &ctx, int32_t level, int32_t level_size,
                   int32_t num_leaves, int32_t *d_partners,
                   int32_t *d_level_map, int32_t *d_old2new) {
  cudaStream_t stream = ctx.stream();
  auto d_counts = ctx.device_alloc_async<match_unmatch_pair>(level_size);

  int32_t block_dim = 256;
  int32_t node_grid =
      ctx.grid_dim_by_block_dim(block_dim, 4, ceil_div(level_size, block_dim));
  int32_t leaf_grid =
      ctx.grid_dim_by_block_dim(block_dim, 4, ceil_div(num_leaves, block_dim));

  advance_level_count<<<node_grid, block_dim, 0, stream>>>(
      level_size, d_partners, d_counts);
  ctx.check_last_cuda_error();

  size_t temp_bytes = 0;
  cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d_counts, d_counts,
                                level_size, stream);
  auto d_temp = ctx.device_alloc_async(temp_bytes);
  cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_counts, d_counts,
                                level_size, stream);
  ctx.device_free_async(d_temp);

  advance_level_update_level_map<<<node_grid, block_dim, 0, stream>>>(
      level_size, d_partners, d_counts, d_level_map);
  ctx.check_last_cuda_error();

  advance_level_update_old2new<<<leaf_grid, block_dim, 0, stream>>>(
      num_leaves, level, d_level_map, d_old2new);
  ctx.check_last_cuda_error();

  ctx.device_free_async(d_counts);
}

// ==========================================================================
// iota
// ==========================================================================

__global__ void iota_fill(int32_t n, int32_t start,
                          int32_t *__restrict__ array) {
  for (int32_t i = thread_in_grid(); i < n; i += total_threads_in_grid()) {
    array[i] = start + i;
  }
}

void iota(context &ctx, int32_t n, int32_t *d_array, int32_t start) {
  int32_t block_dim = 256;
  int32_t grid_dim =
      ctx.grid_dim_by_block_dim(block_dim, 4, ceil_div(n, block_dim));
  iota_fill<<<grid_dim, block_dim, 0, ctx.stream()>>>(n, start, d_array);
  ctx.check_last_cuda_error();
}

} // namespace sporder::hm
