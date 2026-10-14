#include "cg/stats_collector.h"
#include "common/similarity_top_k.h"
#include <algorithm>

namespace sporder::cg {

stats_collector::stats_collector(context &ctx, sporder_cg_stats_t *stats)
    : ctx_(ctx), stats_(stats) {
  if (!stats_) {
    return;
  }
  d_counters_ = ctx_.device_alloc_async<chain_counters>();
  d_total_degree_ = ctx_.device_alloc_async<unsigned long long>();
  d_total_weight_ = ctx_.device_alloc_async<double>();
  d_zero_degree_rows_ = ctx_.device_alloc_async<int32_t>();
  ctx_.device_memset_async(d_counters_, 0, sizeof(*d_counters_));
  ctx_.device_memset_async(d_total_degree_, 0, sizeof(*d_total_degree_));
  ctx_.device_memset_async(d_total_weight_, 0, sizeof(*d_total_weight_));
  ctx_.device_memset_async(d_zero_degree_rows_, 0,
                           sizeof(*d_zero_degree_rows_));
}

stats_collector::~stats_collector() {
  if (!stats_) {
    return;
  }
  ctx_.device_free_async(d_counters_);
  ctx_.device_free_async(d_total_degree_);
  ctx_.device_free_async(d_total_weight_);
  ctx_.device_free_async(d_zero_degree_rows_);
}

void stats_collector::on_similarity(int32_t n, int32_t k,
                                    const weighted_index *d_neighbors) {
  if (!stats_) {
    return;
  }
  n_ = n;
  similarity_top_k_stats(ctx_, n, k, d_neighbors, d_total_degree_,
                         d_total_weight_, d_zero_degree_rows_);
}

void stats_collector::publish() {
  if (!stats_) {
    return;
  }
  // Deferred D2H: keeping the hot path free of copies avoids pageable
  // memcpy stalls that would otherwise skew the stage timings.
  ctx_.memcpy_device_to_host_async(&h_counters_, d_counters_);
  ctx_.memcpy_device_to_host_async(&h_total_degree_, d_total_degree_);
  ctx_.memcpy_device_to_host_async(&h_total_weight_, d_total_weight_);
  ctx_.memcpy_device_to_host_async(&h_zero_degree_rows_, d_zero_degree_rows_);
  ctx_.wait();

  stats_->avg_degree = n_ > 0 ? static_cast<double>(h_total_degree_) / n_ : 0.0;
  stats_->avg_edge_weight =
      h_total_degree_ > 0
          ? h_total_weight_ / static_cast<double>(h_total_degree_)
          : 0.0;
  stats_->zero_degree_rows = h_zero_degree_rows_;
  stats_->chain_restarts = h_counters_.restarts;
  stats_->avg_followed_weight =
      h_counters_.links > 0 ? h_counters_.link_weight_sum / h_counters_.links
                            : 0.0;
  // The two weight sums accumulate in different orders, so their
  // ratio can drift past [0, 1] by float noise; clamp to the
  // documented range.
  double cut = h_total_weight_ > 0.0
                   ? 1.0 - h_counters_.in_chunk_edge_weight / h_total_weight_
                   : 0.0;
  stats_->cut_weight_fraction = std::min(1.0, std::max(0.0, cut));
}

} // namespace sporder::cg
