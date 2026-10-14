#pragma once
#include "common/context.h"
#include "common/weighted_index.h"
#include <sporder.h>

namespace sporder::cg {

/*
 * chain_counters -- device-side counters filled by chunk_greedy_kernel.
 *
 * One global instance accumulated atomically across all chunks:
 *   restarts             chain breaks beyond the first chain of each chunk
 *   links                greedy links followed
 *   link_weight_sum      sum of followed link weights
 *   in_chunk_edge_weight sum of edge weights whose target lies in the
 *                        source's chunk (complement of the cut)
 */
struct chain_counters {
  int32_t restarts;
  int32_t links;
  double link_weight_sum;
  double in_chunk_edge_weight;
};

/*
 * stats_collector -- optional diagnostics for chunk_greedy.
 *
 * Every method is a no-op when constructed with stats == nullptr, so
 * call sites in the main algorithm stay unconditional.  on_similarity
 * enqueues a small device pass on the context stream;
 * device_counters() exposes a zeroed chain_counters instance for
 * chunk_greedy_kernel (nullptr when disabled, which skips in-kernel
 * collection).  publish() must be called after the stream has been
 * synchronized and writes the results into the user-provided struct;
 * k_used, m_used, and chunk_size_used are filled by the
 * C bridge instead.
 */
class stats_collector {
public:
  stats_collector(context &ctx, sporder_cg_stats_t *stats);
  ~stats_collector();

  stats_collector(const stats_collector &) = delete;
  stats_collector &operator=(const stats_collector &) = delete;

  void on_similarity(int32_t n, int32_t k, const weighted_index *d_neighbors);
  chain_counters *device_counters() const { return d_counters_; }
  void publish();

private:
  context &ctx_;
  sporder_cg_stats_t *stats_;
  int32_t n_ = 0;

  chain_counters h_counters_{};
  unsigned long long h_total_degree_ = 0;
  double h_total_weight_ = 0.0;
  int32_t h_zero_degree_rows_ = 0;

  chain_counters *d_counters_ = nullptr;
  unsigned long long *d_total_degree_ = nullptr;
  double *d_total_weight_ = nullptr;
  int32_t *d_zero_degree_rows_ = nullptr;
};

} // namespace sporder::cg
