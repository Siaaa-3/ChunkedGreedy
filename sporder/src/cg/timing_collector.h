#pragma once
#include "common/context.h"
#include "common/timer.h"
#include <chrono>
#include <optional>
#include <sporder.h>

namespace sporder::cg {

/*
 * timing_collector -- optional timings for chunk_greedy.
 *
 * Sole writer of the sporder_cg_timing_t struct: wall_ms spans from
 * construction to publish() on the host clock, and one cuda_timer per
 * stage covers csc resolve, similarity, and the greedy kernel.  Every
 * method is a no-op when constructed with timing == nullptr, so call
 * sites in the main algorithm stay unconditional.  Call begin_xxx()
 * immediately before a timed call and end_xxx() immediately after, so
 * only that call's device work is attributed to the stage.  publish()
 * must be called after the stream has been synchronized and writes
 * the results into the user-provided struct.
 */
class timing_collector {
public:
  timing_collector(context &ctx, sporder_cg_timing_t *timing)
      : timing_(timing) {
    if (!timing_) {
      return;
    }
    wall_begin_ = std::chrono::steady_clock::now();
    cudaStream_t stream = ctx.stream();
    csr2csc_.emplace(stream);
    similarity_.emplace(stream);
    chunk_greedy_.emplace(stream);
  }

  void begin_csr2csc() {
    if (timing_) {
      csr2csc_->begin();
    }
  }

  void end_csr2csc() {
    if (timing_) {
      csr2csc_->end();
    }
  }

  void begin_similarity() {
    if (timing_) {
      similarity_->begin();
    }
  }

  void end_similarity() {
    if (timing_) {
      similarity_->end();
    }
  }

  void begin_chunk_greedy() {
    if (timing_) {
      chunk_greedy_->begin();
    }
  }

  void end_chunk_greedy() {
    if (timing_) {
      chunk_greedy_->end();
    }
  }

  void publish() {
    if (!timing_) {
      return;
    }
    timing_->wall_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - wall_begin_)
                           .count();
    timing_->csr2csc_ms = csr2csc_->elapsed_ms();
    timing_->similarity_ms = similarity_->elapsed_ms();
    timing_->chunk_greedy_ms = chunk_greedy_->elapsed_ms();
  }

private:
  sporder_cg_timing_t *timing_;
  std::chrono::steady_clock::time_point wall_begin_;
  std::optional<cuda_timer> csr2csc_;
  std::optional<cuda_timer> similarity_;
  std::optional<cuda_timer> chunk_greedy_;
};

} // namespace sporder::cg
