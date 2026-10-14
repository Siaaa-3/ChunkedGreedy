#pragma once
#include "common/error_checking.h"
#include <cuda_runtime.h>

namespace sporder {

/*
 * cuda_timer -- one timed span on a stream, tracked by a start/stop
 * event pair.  Call begin() and end() around the work to measure;
 * elapsed_ms() may only be called after the stream is synced.
 */
class cuda_timer {
  cudaStream_t stream_;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;

public:
  explicit cuda_timer(cudaStream_t stream) : stream_(stream) {
    ensure(cudaEventCreate(&start_));
    ensure(cudaEventCreate(&stop_));
  }

  cuda_timer(cuda_timer &&other) noexcept
      : stream_(other.stream_), start_(other.start_), stop_(other.stop_) {
    other.start_ = nullptr;
    other.stop_ = nullptr;
  }

  cuda_timer(const cuda_timer &) = delete;
  cuda_timer &operator=(const cuda_timer &) = delete;
  cuda_timer &operator=(cuda_timer &&) = delete;

  ~cuda_timer() {
    if (start_) {
      ensure(cudaEventDestroy(start_));
    }
    if (stop_) {
      ensure(cudaEventDestroy(stop_));
    }
  }

  void begin() { ensure(cudaEventRecord(start_, stream_)); } // async, no stall
  void end() { ensure(cudaEventRecord(stop_, stream_)); }    // async, no stall

  // ms between begin() and end() -- call ONLY after the stream is synced
  float elapsed_ms() const {
    float ms;
    ensure(cudaEventElapsedTime(&ms, start_, stop_));
    return ms;
  }
};

} // namespace sporder
