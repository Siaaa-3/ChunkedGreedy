#pragma once
#include "common/cuda_util.cuh"
#include "common/weighted_index.h"
#include <cstdint>

namespace sporder {

namespace {
const uint32_t MASK = 0xFFFFFFFFu;
} // namespace

template <int32_t M>
class warp_spacesaving {
  static_assert(M >= 32 && M % 32 == 0, "M must be a multiple of 32");

  static constexpr int32_t CELLS = M / 32;

  int32_t keys_[CELLS];
  float weights_[CELLS];

public:
  __device__ warp_spacesaving() {}

  __device__ __forceinline__ void clear() {
    for (int32_t c = 0; c < CELLS; ++c) {
      keys_[c] = -1;
      weights_[c] = 0.0f;
    }
  }

  __device__ __forceinline__ void add(int32_t key, float weight) {
    bool todo = (key != -1);
    unsigned todos = __ballot_sync(MASK, todo);

    while (todos) {
      int32_t cur = __ffs(todos) - 1; // position of lsb - 1
      int32_t cur_k = __shfl_sync(MASK, key, cur);
      float cur_w = __shfl_sync(MASK, weight, cur);

      _cooperative_single_add(cur_k, cur_w);

      todos &= ~(1u << cur); // set the cur bit to 0
    }
  }

  __device__ __forceinline__ void
  select_top_k(int32_t k, weighted_index *__restrict__ out) __restrict__ {
    _local_sort_descending();

    int32_t pos = 0; // pos in CELLS per thread

    for (int32_t i = 0; i < k; ++i) {
      // stop if all threads exhausted
      if (__all_sync(MASK, pos >= CELLS || keys_[pos] == -1)) {
        break;
      }
      int32_t max_lane =
          warp_arg_max(lane_in_warp(), pos < CELLS ? weights_[pos] : 0.0f);
      if (lane_in_warp() == max_lane) {
        out[i] = weighted_index(keys_[pos], weights_[pos]);
        pos += 1;
      }
    }

    // assume out array is initialized as {-1,0} for early break
  }

private:
  __device__ __forceinline__ void _local_sort_descending() {
#pragma unroll
    for (int32_t i = 0; i < CELLS - 1; ++i) {
      int32_t best = i;
      if (keys_[best] == -1) {
        break;
      }
      for (int32_t j = i + 1; j < CELLS; ++j) {
        if (keys_[j] == -1) {
          break;
        }
        if (weights_[j] > weights_[best]) {
          best = j;
        }
      }
      if (best != i) {
        _swap(i, best);
      }
    }
  }

  __device__ __forceinline__ void _swap(int32_t a, int32_t b) {
    int32_t tk = keys_[a];
    keys_[a] = keys_[b];
    keys_[b] = tk;
    float tw = weights_[a];
    weights_[a] = weights_[b];
    weights_[b] = tw;
  }

  __device__ __forceinline__ int32_t find(int32_t k) {
#pragma unroll
    for (int32_t c = 0; c < CELLS; ++c) {
      if (keys_[c] == k) {
        return c;
      }
    }
    return -1;
  }

  __device__ __forceinline__ void _cooperative_single_add(int32_t k, float w) {
    // try to find exisiting k and increase its w
    int32_t k_cell = find(k);
    if (k_cell >= 0) {
      weights_[k_cell] += w;
    }

    unsigned warp_found_k = __ballot_sync(MASK, k_cell >= 0);
    if (warp_found_k) {
      return;
    }

    // k not found, try to find an empty cell
    int32_t empty_cell = find(-1);
    unsigned warp_found_empty = __ballot_sync(MASK, empty_cell >= 0);
    if (warp_found_empty) {
      int32_t owner = __ffs(warp_found_empty) - 1;
      if (lane_in_warp() == owner) {
        keys_[empty_cell] = k;
        weights_[empty_cell] = w;
      }
      return;
    }

    // k not found and no empty cell, evict smallest
    int32_t min_cell_lane = (0 << 5) | lane_in_warp();
    float min_weight = weights_[0];
#pragma unroll
    for (int32_t c = 1; c < CELLS; ++c) {
      if (weights_[c] < min_weight) {
        min_weight = weights_[c];
        min_cell_lane = (c << 5) | lane_in_warp();
      }
    }
    int32_t warp_min_cell_lane = warp_arg_min(min_cell_lane, min_weight);

    if (lane_in_warp() == (warp_min_cell_lane & 31)) {
      int warp_min_cell = warp_min_cell_lane >> 5;
      keys_[warp_min_cell] = k;
      weights_[warp_min_cell] += w;
    }
  }
};

} // namespace sporder
