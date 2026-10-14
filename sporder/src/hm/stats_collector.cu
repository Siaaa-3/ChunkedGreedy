#include "common/cuda_util.cuh"
#include "common/similarity_top_k.h"
#include "hm/stats_collector.h"

namespace sporder::hm {

namespace {

__global__ void count_matched_pairs(int32_t level_size,
                                    const int32_t *__restrict__ partners,
                                    int32_t *__restrict__ matched_pairs) {
  int32_t thread_pairs = 0;
  for (int32_t i = thread_in_grid(); i < level_size;
       i += total_threads_in_grid()) {
    thread_pairs += (partners[i] > i);
  }
  thread_pairs = warp_sum(thread_pairs);
  if (lane_in_warp() == 0 && thread_pairs > 0) {
    atomicAdd(matched_pairs, thread_pairs);
  }
}

} // namespace

stats_collector::stats_collector(context &ctx, sporder_hm_stats_t *stats,
                                 int32_t num_levels)
    : ctx_(ctx), stats_(stats), num_levels_(num_levels) {
  if (!stats_) {
    return;
  }
  level_size_.assign(num_levels_, 0);
  matched_pairs_.assign(num_levels_, 0);
  rounds_.assign(num_levels_, 0);
  d_total_degree_ = ctx_.device_alloc_async<unsigned long long>();
  d_total_weight_ = ctx_.device_alloc_async<double>();
  d_zero_degree_rows_ = ctx_.device_alloc_async<int32_t>();
  d_matched_pairs_ = ctx_.device_alloc_async<int32_t>(num_levels_);
  ctx_.device_memset_async(d_total_degree_, 0, sizeof(*d_total_degree_));
  ctx_.device_memset_async(d_total_weight_, 0, sizeof(*d_total_weight_));
  ctx_.device_memset_async(d_zero_degree_rows_, 0,
                           sizeof(*d_zero_degree_rows_));
  ctx_.device_memset_async(d_matched_pairs_, 0,
                           num_levels_ * sizeof(*d_matched_pairs_));
}

stats_collector::~stats_collector() {
  if (!stats_) {
    return;
  }
  ctx_.device_free_async(d_total_degree_);
  ctx_.device_free_async(d_total_weight_);
  ctx_.device_free_async(d_zero_degree_rows_);
  ctx_.device_free_async(d_matched_pairs_);
}

void stats_collector::on_similarity(int32_t n, int32_t k,
                                    const weighted_index *d_sim) {
  if (!stats_) {
    return;
  }
  n_ = n;
  similarity_top_k_stats(ctx_, n, k, d_sim, d_total_degree_, d_total_weight_,
                         d_zero_degree_rows_);
}

void stats_collector::on_matching(int32_t level, int32_t level_size,
                                  int32_t rounds, const int32_t *d_partners) {
  if (!stats_) {
    return;
  }
  level_size_[level] = level_size;
  rounds_[level] = rounds;
  int32_t block_dim = 256;
  int32_t grid_dim =
      ctx_.grid_dim_by_block_dim(block_dim, 4, ceil_div(level_size, block_dim));
  count_matched_pairs<<<grid_dim, block_dim, 0, ctx_.stream()>>>(
      level_size, d_partners, d_matched_pairs_ + level);
  ctx_.check_last_cuda_error();
}

void stats_collector::publish() {
  if (!stats_) {
    return;
  }
  // Deferred D2H: keeping the hot loop free of copies avoids pageable
  // memcpy stalls that would otherwise skew the stage timings.
  ctx_.memcpy_device_to_host_async(&h_total_degree_, d_total_degree_);
  ctx_.memcpy_device_to_host_async(&h_total_weight_, d_total_weight_);
  ctx_.memcpy_device_to_host_async(&h_zero_degree_rows_, d_zero_degree_rows_);
  ctx_.memcpy_device_to_host_async(matched_pairs_.data(), d_matched_pairs_,
                                   num_levels_);
  ctx_.wait();

  stats_->avg_degree = n_ > 0 ? static_cast<double>(h_total_degree_) / n_ : 0.0;
  stats_->avg_edge_weight =
      h_total_degree_ > 0
          ? h_total_weight_ / static_cast<double>(h_total_degree_)
          : 0.0;
  stats_->zero_degree_rows = h_zero_degree_rows_;
  for (int32_t level = 0; level < num_levels_; ++level) {
    if (stats_->level_size) {
      stats_->level_size[level] = level_size_[level];
    }
    if (stats_->matched_pairs) {
      stats_->matched_pairs[level] = matched_pairs_[level];
    }
    if (stats_->rounds) {
      stats_->rounds[level] = rounds_[level];
    }
  }
}

} // namespace sporder::hm
