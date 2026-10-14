#include "common/similarity_top_k.h"
#include "csg/stats_collector.h"

namespace sporder::csg {

stats_collector::stats_collector(context &ctx, sporder_csg_stats_t *s)
    : ctx_(ctx), s_(s) {
  if (!s_) {
    return;
  }
  d_total_degree_ = ctx_.device_alloc_async<unsigned long long>(1);
  d_total_weight_ = ctx_.device_alloc_async<double>(1);
  d_zero_degree_rows_ = ctx_.device_alloc_async<int32_t>(1);
  d_counters_ = ctx_.device_alloc_async<chain_counters>(1);
  ctx_.device_memset_async(d_total_degree_, 0, sizeof(*d_total_degree_));
  ctx_.device_memset_async(d_total_weight_, 0, sizeof(*d_total_weight_));
  ctx_.device_memset_async(d_zero_degree_rows_, 0,
                           sizeof(*d_zero_degree_rows_));
  ctx_.device_memset_async(d_counters_, 0, sizeof(*d_counters_));
}

stats_collector::~stats_collector() {
  if (!s_) {
    return;
  }
  ctx_.wait();
  ctx_.device_free_async(d_total_degree_);
  ctx_.device_free_async(d_total_weight_);
  ctx_.device_free_async(d_zero_degree_rows_);
  ctx_.device_free_async(d_counters_);
}

void stats_collector::on_similarity(int32_t n, int32_t k,
                                    const weighted_index *d_neighbors) {
  if (!s_) {
    return;
  }
  n_ = n;
  similarity_top_k_stats(ctx_, n, k, d_neighbors, d_total_degree_,
                         d_total_weight_, d_zero_degree_rows_);
}

void stats_collector::publish() {
  if (!s_) {
    return;
  }
  ctx_.memcpy_device_to_host_async(&h_total_degree_, d_total_degree_);
  ctx_.memcpy_device_to_host_async(&h_total_weight_, d_total_weight_);
  ctx_.memcpy_device_to_host_async(&h_zero_degree_rows_, d_zero_degree_rows_);
  ctx_.memcpy_device_to_host_async(&h_counters_, d_counters_);
  ctx_.wait();

  s_->avg_degree = (n_ > 0) ? static_cast<double>(h_total_degree_) / n_ : 0.0;
  s_->avg_edge_weight =
      (h_total_degree_ > 0)
          ? h_total_weight_ / static_cast<double>(h_total_degree_)
          : 0.0;
  s_->zero_degree_rows = h_zero_degree_rows_;
  s_->chain_restarts = h_counters_.restarts;
  s_->avg_followed_weight =
      (h_counters_.links > 0) ? h_counters_.link_weight_sum / h_counters_.links
                              : 0.0;
}

} // namespace sporder::csg
