#pragma once
#include "common/context.h"
#include "common/weighted_index.h"
#include <sporder.h>

namespace sporder::csg {

struct chain_counters {
  int32_t restarts;
  int32_t links;
  double link_weight_sum;
};

class stats_collector {
public:
  stats_collector(context &ctx, sporder_csg_stats_t *s);
  ~stats_collector();

  stats_collector(const stats_collector &) = delete;
  stats_collector &operator=(const stats_collector &) = delete;

  void on_similarity(int32_t n, int32_t k, const weighted_index *d_neighbors);
  chain_counters *device_counters() const { return d_counters_; }
  void publish();

private:
  context &ctx_;
  sporder_csg_stats_t *s_;
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

} // namespace sporder::csg
