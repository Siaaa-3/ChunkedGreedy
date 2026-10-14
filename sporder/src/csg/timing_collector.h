#pragma once
#include "common/context.h"
#include "common/timer.h"
#include <chrono>
#include <optional>
#include <sporder.h>

namespace sporder::csg {

class timing_collector {
public:
  timing_collector(context &ctx, sporder_csg_timing_t *t) : timing_(t) {
    if (!timing_) {
      return;
    }
    wall_begin_ = std::chrono::steady_clock::now();
    cudaStream_t s = ctx.stream();
    csr2csc_.emplace(s);
    similarity_.emplace(s);
    greedy_.emplace(s);
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
  void begin_greedy() {
    if (timing_) {
      greedy_->begin();
    }
  }
  void end_greedy() {
    if (timing_) {
      greedy_->end();
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
    timing_->chunk_greedy_ms = greedy_->elapsed_ms();
  }

private:
  sporder_csg_timing_t *timing_;
  std::chrono::steady_clock::time_point wall_begin_;
  std::optional<cuda_timer> csr2csc_, similarity_, greedy_;
};

} // namespace sporder::csg
