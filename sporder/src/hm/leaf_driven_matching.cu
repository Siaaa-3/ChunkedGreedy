#include "common/cuda_util.cuh"
#include "hm/leaf_driven_matching.h"

namespace sporder::hm {

namespace {

__device__ __forceinline__ bool leaf_propose(
    int32_t leaf, int32_t level, int32_t max_neighbors,
    int32_t incomplete_super, const weighted_index *__restrict__ neighbors,
    const int32_t *__restrict__ old2new, int32_t *__restrict__ neighbor_iters,
    weighted_index *__restrict__ best_suitors) {
  bool changed = false;
  int32_t my_super = old2new[leaf] >> level; // this leaf's super-node
  if (my_super == incomplete_super) {
    return false;
  }
  neighbors += static_cast<size_t>(leaf) * max_neighbors;
  int iter = neighbor_iters[leaf];
  for (; iter < max_neighbors; ++iter) {
    weighted_index weighted_u = neighbors[iter];
    int32_t u = weighted_u.index();
    if (u == -1) {
      iter = max_neighbors;
      break;
    }
    int32_t target_super = old2new[u] >> level; // neighbour's super-node
    if (target_super == my_super || target_super == incomplete_super) {
      continue;
    }
    weighted_index new_proposal = weighted_u.replace_index(my_super);
    if (new_proposal >
        best_suitors[target_super]) { // read-before-atomic filter
      weighted_index old_proposal =
          best_suitors[target_super].atomic_update_if_greater(new_proposal);
      if (new_proposal > old_proposal) {
        changed = true;
        iter += 1;
        break;
      }
    }
  }
  neighbor_iters[leaf] = iter;
  return changed;
}

__global__ void suitor_round(int32_t num_leaves, int32_t level,
                             int32_t max_neighbors, int32_t incomplete_super,
                             const weighted_index *__restrict__ neighbors,
                             const int32_t *__restrict__ old2new,
                             int32_t *__restrict__ neighbor_iters,
                             weighted_index *__restrict__ best_suitors,
                             bool *__restrict__ changed) {
  bool thread_changed = false;
  for (int32_t leaf = thread_in_grid(); leaf < num_leaves;
       leaf += total_threads_in_grid()) {
    thread_changed |=
        leaf_propose(leaf, level, max_neighbors, incomplete_super, neighbors,
                     old2new, neighbor_iters, best_suitors);
  }

  bool warp_changed = __any_sync(0xFFFFFFFFu, thread_changed);
  if (lane_in_warp() == 0 && warp_changed) {
    *changed = true;
  }
}

__global__ void
finalize_partners(int32_t level_size,
                  const weighted_index *__restrict__ best_suitors,
                  int32_t *__restrict__ partners) {
  for (int32_t v = thread_in_grid(); v < level_size;
       v += total_threads_in_grid()) {
    int32_t u = best_suitors[v].index();
    if (u != -1 && v < u) {
      int32_t w = best_suitors[u].index();
      if (v == w) {
        partners[v] = u;
        partners[u] = v;
      }
    }
  }
}

} // namespace

int32_t leaf_driven_matching(context &ctx, int32_t num_leaves,
                             int32_t level_size, int32_t level,
                             int32_t max_neighbors, int32_t max_rounds,
                             const weighted_index *d_neighbors,
                             const int32_t *d_old2new, int32_t *d_partners) {
  cudaStream_t stream = ctx.stream();

  auto d_suitors = ctx.device_alloc_async<weighted_index>(level_size);
  auto d_neighbor_iters = ctx.device_alloc_async<int32_t>(num_leaves);
  auto d_changed = ctx.device_alloc_async<bool>();

  ctx.device_memset_async(d_suitors, 0, level_size * sizeof(*d_suitors));
  ctx.device_memset_async(d_neighbor_iters, 0,
                          num_leaves * sizeof(*d_neighbor_iters));
  ctx.device_memset_async(d_partners, 0xFF, level_size * sizeof(*d_partners));

  int32_t block_dim = 256;
  dim3 block(block_dim);
  dim3 grid_leaves(
      ctx.grid_dim_by_block_dim(block_dim, 4, ceil_div(num_leaves, block_dim)));
  dim3 grid_super(
      ctx.grid_dim_by_block_dim(block_dim, 4, ceil_div(level_size, block_dim)));

  // The last super-node is under-full iff 2^level does not divide num_leaves.
  // Such incomplete super-node is ignored during matching so that it becomes
  // an unmatched node in the output of this function. This allows the caller
  // (i.e. advance_level() in hierarchical_matching.cu) to keep this super-node
  // ALWAYS at the end of old2new and ensures that old2new is always a
  // permutation of 0,1,...,n-1 without any holes.
  int32_t incomplete_super =
      (num_leaves % (1 << level)) != 0 ? level_size - 1 : -1;

  int32_t rounds = 0;
  while (max_rounds <= 0 || rounds < max_rounds) {
    ctx.device_memset_async(d_changed, 0, sizeof(*d_changed));

    suitor_round<<<grid_leaves, block, 0, stream>>>(
        num_leaves, level, max_neighbors, incomplete_super, d_neighbors,
        d_old2new, d_neighbor_iters, d_suitors, d_changed);
    ctx.check_last_cuda_error();
    rounds += 1;

    bool changed;
    ctx.memcpy_device_to_host_async(&changed, d_changed);
    ctx.wait();
    if (!changed) {
      break;
    }
  }

  finalize_partners<<<grid_super, block, 0, stream>>>(level_size, d_suitors,
                                                      d_partners);
  ctx.check_last_cuda_error();

  ctx.device_free_async(d_suitors);
  ctx.device_free_async(d_neighbor_iters);
  ctx.device_free_async(d_changed);

  return rounds;
}

} // namespace sporder::hm
