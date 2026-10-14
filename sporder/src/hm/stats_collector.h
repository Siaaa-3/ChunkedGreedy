#pragma once
#include "common/context.h"
#include "common/weighted_index.h"
#include <sporder.h>
#include <vector>

namespace sporder::hm {

/*
 * stats_collector -- optional diagnostics for hierarchical_matching.
 *
 * Every method is a no-op when constructed with stats == nullptr, so
 * call sites in the main algorithm stay unconditional.  The on_*
 * methods enqueue small device passes and async D2H copies on the
 * context stream; publish() must be called after the stream has been
 * synchronized and writes the results into the user-provided struct.
 */
class stats_collector {
public:
  stats_collector(context &ctx, sporder_hm_stats_t *stats, int32_t num_levels);
  ~stats_collector();

  stats_collector(const stats_collector &) = delete;
  stats_collector &operator=(const stats_collector &) = delete;

  void on_similarity(int32_t n, int32_t k, const weighted_index *d_sim);
  void on_matching(int32_t level, int32_t level_size, int32_t rounds,
                   const int32_t *d_partners);
  void publish();

private:
  context &ctx_;
  sporder_hm_stats_t *stats_;
  int32_t num_levels_;
  int32_t n_ = 0;

  std::vector<int32_t> level_size_;
  std::vector<int32_t> matched_pairs_;
  std::vector<int32_t> rounds_;
  unsigned long long h_total_degree_ = 0;
  double h_total_weight_ = 0.0;
  int32_t h_zero_degree_rows_ = 0;

  unsigned long long *d_total_degree_ = nullptr;
  double *d_total_weight_ = nullptr;
  int32_t *d_zero_degree_rows_ = nullptr;
  int32_t *d_matched_pairs_ = nullptr;
};

} // namespace sporder::hm
