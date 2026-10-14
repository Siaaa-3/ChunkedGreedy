#pragma once
#include "common/context.h"
#include "common/timer.h"
#include <chrono>
#include <optional>
#include <sporder.h>
#include <vector>

namespace sporder::hm {

/*
 * timing_collector -- optional timings for hierarchical_matching.
 *
 * Sole writer of the sporder_hm_timing_t struct: wall_ms spans from
 * construction to publish() on the host clock, and one cuda_timer per
 * timed computation covers the stages, all known up front: one csc
 * resolve, one similarity pass, plus a matching and an advance per
 * level.  Every method is a no-op when constructed with timing ==
 * nullptr, so call sites in the main algorithm stay unconditional.
 * Call begin_xxx() immediately before a timed call and end_xxx()
 * immediately after, so only that call's device work is attributed to
 * the stage.  publish() must be called after the stream has been
 * synchronized and writes the results into the user-provided struct.
 */
class timing_collector {
public:
  timing_collector(context &ctx, sporder_hm_timing_t *timing,
                   int32_t num_levels)
      : timing_(timing), num_levels_(num_levels) {
    if (!timing_) {
      return;
    }
    wall_begin_ = std::chrono::steady_clock::now();
    cudaStream_t stream = ctx.stream();
    csr2csc_.emplace(stream);
    similarity_.emplace(stream);
    matching_.reserve(num_levels_);
    advance_.reserve(num_levels_);
    for (int32_t level = 0; level < num_levels_; ++level) {
      matching_.emplace_back(stream);
      advance_.emplace_back(stream);
    }
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

  void begin_matching(int32_t level) {
    if (timing_) {
      matching_[level].begin();
    }
  }

  void end_matching(int32_t level) {
    if (timing_) {
      matching_[level].end();
    }
  }

  void begin_advance(int32_t level) {
    if (timing_) {
      advance_[level].begin();
    }
  }

  void end_advance(int32_t level) {
    if (timing_) {
      advance_[level].end();
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
    timing_->matching_ms = 0.0f;
    timing_->advance_ms = 0.0f;
    for (int32_t level = 0; level < num_levels_; ++level) {
      float m = matching_[level].elapsed_ms();
      float a = advance_[level].elapsed_ms();
      timing_->matching_ms += m;
      timing_->advance_ms += a;
      if (timing_->matching_per_level_ms) {
        timing_->matching_per_level_ms[level] = m;
      }
      if (timing_->advance_per_level_ms) {
        timing_->advance_per_level_ms[level] = a;
      }
    }
  }

private:
  sporder_hm_timing_t *timing_;
  int32_t num_levels_;
  std::chrono::steady_clock::time_point wall_begin_;
  std::optional<cuda_timer> csr2csc_;
  std::optional<cuda_timer> similarity_;
  std::vector<cuda_timer> matching_;
  std::vector<cuda_timer> advance_;
};

} // namespace sporder::hm
