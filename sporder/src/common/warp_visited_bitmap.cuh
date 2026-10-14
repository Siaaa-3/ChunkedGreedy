#pragma once
#include "common/cuda_util.cuh"
#include <cstdint>

namespace sporder {

/*
 * warp_visited_bitmap — warp-cooperative visited-set in shared memory.
 *
 * Tracks up to `size` elements as visited / unvisited bits.  All 32
 * lanes of the warp participate; every method assumes the warp is
 * converged and the caller has already called __syncwarp() where
 * necessary (see individual methods).
 *
 * Shared memory requirement:  `((size + 31) / 32) * sizeof(uint32_t)`
 * bytes.  Allocate it from the kernel's dynamic shared memory pool and
 * pass the pointer to the constructor:
 *
 *     extern __shared__ char s_buf[];
 *     warp_visited_bitmap visited(reinterpret_cast<uint32_t *>(s_buf), size);
 *
 * With `size = 1024` rows this needs only 128 bytes of shared memory,
 * leaving the rest for other per-block uses.
 *
 * --- Warp-sync contract ------------------------------------------------
 *
 *   - After `clear()`:  call __syncwarp() before any `set()` or `test()`
 *                        so that all lanes see the zeroed bitmap.
 *   - After one or more `set()` calls from different lanes:  call
 *     __syncwarp() before `find_first_unvisited()` so the scan observes
 *     all writes.
 *   - `test()` does not need its own sync if the warp is already
 *     converged (no other lane has written to the same word since the
 *     last sync).
 *
 * --- Thread-safety notes ----------------------------------------------
 *
 *   `set()` and `test()` are expected to be called on *distinct* indices
 *   by different lanes; concurrent writes to the same 32-bit word are
 *   safe (all lanes set different bits via |=), but if two lanes write
 *   the same index simultaneously the result is a benign duplicate set.
 *
 *   `find_first_unvisited()` is a read-only collective: all lanes
 *   independently scan the same shared memory and reach the same
 *   conclusion; no warp shuffle is needed.
 */

class warp_visited_bitmap {
  uint32_t *data_;
  int32_t size_;
  int32_t words_;

public:
  /*
   * Constructor.  `smem` must point to at least
   *   ((size + 31) / 32) * sizeof(uint32_t)
   * bytes of shared memory.
   */
  __device__ warp_visited_bitmap(uint32_t *smem, int32_t size)
      : data_(smem), size_(size), words_((size + 31) >> 5) {}

  /*
   * Zero every word of the bitmap.  Each lane clears one word per
   * iteration (lane `w` handles word `w + 32 * k`), so all `words_`
   * words are cleared in `ceil(words_ / 32)` iterations.
   *
   * Call __syncwarp() after this before any `set()` or `test()`.
   */
  __device__ void clear() {
    for (int32_t w = lane_in_warp(); w < words_; w += 32) {
      data_[w] = 0;
    }
  }

  /*
   * Mark index `idx` as visited.  Safe to call concurrently from all
   * 32 lanes with distinct indices; uses bitwise OR to set the
   * corresponding bit.
   */
  __device__ void set(int32_t idx) { data_[idx >> 5] |= 1u << (idx & 31); }

  /*
   * Test whether index `idx` has been marked.  Caller must ensure the
   * warp is converged (no pending writes to the same word from other
   * lanes since the last sync).
   */
  __device__ bool test(int32_t idx) const {
    return data_[idx >> 5] & (1u << (idx & 31));
  }

  /*
   * Return the smallest unvisited index, or -1 if all entries are
   * visited.  The bitmask for the last word is clipped to `size % 32`
   * bits so that out-of-range indices are treated as already-visited.
   *
   * Uses the __ffs() intrinsic (find-first-set, a single instruction
   * on all supported architectures).
   */
  __device__ int32_t find_first_unvisited() const {
    int32_t limit32 = size_ & 31;
    for (int32_t w = 0; w < words_; ++w) {
      uint32_t mask = data_[w];
      uint32_t bitmask =
          (w == words_ - 1 && limit32) ? ((1u << limit32) - 1u) : 0xFFFFFFFFu;
      uint32_t free = (~mask) & bitmask;
      if (free) {
        return w * 32 + __ffs(free) - 1;
      }
    }
    return -1;
  }
};

} // namespace sporder
