#include "common/chunk_similarity_top_k.h"
#include "common/csr2csc.h"
#include "common/cuda_util.cuh"
#include "common/warp_visited_bitmap.cuh"
#include "common/weighted_index.h"
#include "csg/chunk_similarity_greedy.h"
#include "csg/stats_collector.h"
#include "csg/timing_collector.h"

namespace sporder::csg {

namespace {

__global__ void greedy_kernel(int32_t n, int32_t k, int32_t chunk_size,
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

  int32_t cur = start, pos = start;
  int32_t restarts = 0, links = 0;
  float link_weight_sum = 0.0f;

  while (true) {
    if (lane_in_warp() == 0) {
      visited.set(cur - start);
      order[pos++] = cur;
    }
    __syncwarp();

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
      if (visited.test(cand - start)) {
        continue;
      }
      float w = row[j].weight();
      if (w > best_w || (w == best_w && cand < best_c)) {
        best_w = w;
        best_c = cand;
      }
    }

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
    if (lane_in_warp() == 0) {
      atomicAdd(&counters->restarts, restarts);
      atomicAdd(&counters->links, links);
      atomicAdd(&counters->link_weight_sum,
                static_cast<double>(link_weight_sum));
    }
  }
}

} // namespace

void chunk_similarity_greedy(context &ctx, int32_t n, int32_t nnz,
                             const int32_t *d_ptr, const int32_t *d_ind,
                             bool is_symmetric, int32_t k, float oversampling,
                             int32_t chunk_size, int32_t *d_new_order,
                             sporder_csg_timing_t *timing,
                             sporder_csg_stats_t *stats) {
  timing_collector tc(ctx, timing);
  stats_collector sc(ctx, stats);
  cudaStream_t stream = ctx.stream();

  auto d_sim =
      ctx.device_alloc_async<weighted_index>(static_cast<size_t>(n) * k);

  tc.begin_csr2csc();
  resolved_csc csc = resolve_csc(ctx, n, nnz, d_ptr, d_ind, is_symmetric);
  tc.end_csr2csc();

  int32_t num_chunks = ceil_div(n, chunk_size);
  tc.begin_similarity();
  multi_chunk_similarity_top_k(ctx, n, k, chunk_size, oversampling, d_ptr,
                               d_ind, csc.d_ptr, csc.d_ind, d_sim);
  tc.end_similarity();
  sc.on_similarity(n, k, d_sim);

  size_t smem =
      (static_cast<size_t>(chunk_size) + 31u) / 32u * sizeof(uint32_t);
  tc.begin_greedy();
  greedy_kernel<<<num_chunks, 32, smem, stream>>>(
      n, k, chunk_size, d_sim, d_new_order, sc.device_counters());
  ctx.check_last_cuda_error();
  tc.end_greedy();

  ctx.device_free_async(d_sim);
  release_csc(ctx, csc);

  ctx.wait();
  sc.publish();
  tc.publish();
}

} // namespace sporder::csg
