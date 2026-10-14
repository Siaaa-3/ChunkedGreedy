#include "cg/chunk_greedy.h"
#include "cg/stats_collector.h"
#include "cg/timing_collector.h"
#include "common/csr2csc.h"
#include "common/cuda_util.cuh"
#include "common/similarity_top_k.h"
#include "common/warp_visited_bitmap.cuh"
#include "common/weighted_index.h"

namespace sporder::cg {

namespace {

// ---- one warp per chunk -----------------------------------------------

__global__ void
chunk_greedy_kernel(int32_t n, int32_t k, int32_t chunk_size,
                    const weighted_index *__restrict__ neighbors,
                    int32_t *__restrict__ order,
                    chain_counters *__restrict__ counters) {
  int32_t chunk_id = blockIdx.x;
  int32_t start = chunk_id * chunk_size;
  int32_t end = (start + chunk_size > n) ? n : start + chunk_size;
  if (start >= end) {
    return;
  }
  int32_t size = end - start;

  extern __shared__ char s_buf[];
  warp_visited_bitmap visited(reinterpret_cast<uint32_t *>(s_buf), size);

  visited.clear();
  __syncwarp();

  int32_t cur = start;
  int32_t pos = start; // only lane 0 writes; other lanes keep their own copy

  // per-warp stats, accumulated only when counters != nullptr
  float lane_in_chunk_weight = 0.0f; // per lane, summed at exit
  int32_t restarts = 0, links = 0;   // lane 0 only
  float link_weight_sum = 0.0f;      // lane 0 only

  while (true) {
    // --- lane 0: mark visited, emit to output ---
    if (lane_in_warp() == 0) {
      visited.set(cur - start);
      order[pos++] = cur;
    }
    __syncwarp();

    // --- all 32 lanes: cooperative candidate scan ---
    const weighted_index *row = neighbors + static_cast<size_t>(cur) * k;
    float best_w = -1.0f;
    int32_t best_c = -1;

    for (int32_t j = lane_in_warp(); j < k; j += 32) {
      int32_t cand = row[j].index();
      if (cand < 0) {
        break;
      }
      if (cand < start || cand >= end) {
        continue;
      }
      if (counters) {
        lane_in_chunk_weight += row[j].weight();
      }
      if (visited.test(cand - start)) {
        continue;
      }
      float w = row[j].weight();
      if (w > best_w || (w == best_w && cand < best_c)) {
        best_w = w;
        best_c = cand;
      }
    }

    // --- warp reduction: pick the highest-weight candidate ---
    best_c = warp_arg_max(best_c, best_w);

    if (best_c >= 0) {
      if (counters) {
        float w = warp_max(best_w);
        if (lane_in_warp() == 0) {
          links += 1;
          link_weight_sum += w;
        }
      }
      cur = best_c;
      continue;
    }

    // --- exhausted -- lane 0 finds the next unvisited ---
    if (lane_in_warp() == 0) {
      int32_t nxt = visited.find_first_unvisited();
      cur = (nxt >= 0) ? start + nxt : -1;
      restarts += (nxt >= 0);
    }
    cur = __shfl_sync(0xFFFFFFFFu, cur, 0);
    if (cur < 0) {
      break;
    }
  }

  if (counters) {
    float in_chunk_weight = warp_sum(lane_in_chunk_weight);
    if (lane_in_warp() == 0) {
      atomicAdd(&counters->restarts, restarts);
      atomicAdd(&counters->links, links);
      atomicAdd(&counters->link_weight_sum,
                static_cast<double>(link_weight_sum));
      atomicAdd(&counters->in_chunk_edge_weight,
                static_cast<double>(in_chunk_weight));
    }
  }
}

} // namespace

// ---- public entry point -----------------------------------------------

void chunk_greedy(context &ctx, int32_t n, int32_t nnz, const int32_t *d_ptr,
                  const int32_t *d_ind, bool is_symmetric, int32_t k,
                  float oversampling, int32_t chunk_size, int32_t *d_new_order,
                  sporder_cg_timing_t *timing, sporder_cg_stats_t *stats) {
  timing_collector timing_collect(ctx, timing);
  stats_collector stats_collect(ctx, stats);
  cudaStream_t stream = ctx.stream();

  auto d_sim =
      ctx.device_alloc_async<weighted_index>(static_cast<size_t>(n) * k);

  timing_collect.begin_csr2csc();
  resolved_csc csc = resolve_csc(ctx, n, nnz, d_ptr, d_ind, is_symmetric);
  timing_collect.end_csr2csc();

  timing_collect.begin_similarity();
  similarity_top_k(ctx, n, k, d_ptr, d_ind, csc.d_ptr, csc.d_ind, oversampling,
                   d_sim);
  timing_collect.end_similarity();
  stats_collect.on_similarity(n, k, d_sim);

  int32_t num_chunks = ceil_div(n, chunk_size);
  size_t smem =
      (static_cast<size_t>(chunk_size) + 31u) / 32u * sizeof(uint32_t);

  timing_collect.begin_chunk_greedy();
  chunk_greedy_kernel<<<num_chunks, 32, smem, stream>>>(
      n, k, chunk_size, d_sim, d_new_order, stats_collect.device_counters());
  ctx.check_last_cuda_error();
  timing_collect.end_chunk_greedy();

  ctx.device_free_async(d_sim);
  release_csc(ctx, csc);

  ctx.wait();
  stats_collect.publish();
  timing_collect.publish();
}

} // namespace sporder::cg
