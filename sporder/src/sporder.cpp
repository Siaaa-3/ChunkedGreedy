#include "bench_cusparse_spmm.h"
#include "cg/chunk_greedy.h"
#include "common/context.h"
#include "count_pillars.h"
#include "csg/chunk_similarity_greedy.h"
#include "hm/hierarchical_matching.h"
#include "reorder.h"
#include "similarity_graph.h"
#include "similarity_graph_exact.h"
#include <sporder.h>

struct sporder_context_struct {
  sporder::context ctx_impl;
};

sporder_context_t sporder_context_create(void) {
  return new sporder_context_struct;
}

void sporder_context_destroy(sporder_context_t h) { delete h; }

void sporder_hierarchical_matching(
    sporder_context_t ctx, int32_t n, int32_t nnz, const int32_t *d_ptr,
    const int32_t *d_ind, bool is_symmetric, int32_t num_levels,
    const sporder_options_t *opts, int32_t *d_new_order,
    sporder_hm_timing_t *timing, sporder_hm_stats_t *stats) {
  // old2new packs `level` side bits plus the super-node index into an int32,
  // and the kernels compute (1 << level); both require num_levels <= 30.
  sporder::expect(num_levels >= 1 && num_levels <= 30,
                  "num_levels must be in [1, 30]");

  sporder_options_t o = opts ? *opts : sporder_options_t{};

  int32_t k = o.k > 0 ? o.k : std::min(32, std::max(16, 2 * (nnz / n)));
  float ovs = o.oversampling > 0.0f ? o.oversampling : 2.0f;
  int32_t mr = o.max_rounds > 0 ? o.max_rounds : 0;

  if (stats) {
    stats->k_used = k;
    // the similarity sketch rounds its capacity up to a multiple of 32
    stats->m_used = ((static_cast<int32_t>(k * ovs) + 31) / 32) * 32;
  }

  sporder::hm::hierarchical_matching(ctx->ctx_impl, n, nnz, d_ptr, d_ind,
                                     is_symmetric, num_levels, k, ovs, mr,
                                     d_new_order, timing, stats);
}

void sporder_reorder(sporder_context_t ctx, int32_t n, int32_t nnz,
                     const int32_t *d_ptr_old, const int32_t *d_ind_old,
                     const float *d_val_old, const int32_t *d_new_order,
                     sporder_reorder_mode_t mode, int32_t *d_ptr_new,
                     int32_t *d_ind_new, float *d_val_new) {
  sporder::expect(mode == SPORDER_REORDER_ROWS ||
                      mode == SPORDER_REORDER_SYMMETRIC,
                  "invalid reorder mode");
  sporder::reorder(ctx->ctx_impl, n, nnz, d_ptr_old, d_ind_old, d_val_old,
                   d_new_order, mode, d_ptr_new, d_ind_new, d_val_new);
  ctx->ctx_impl.wait();
}

void sporder_count_pillars(sporder_context_t ctx, int32_t n, int32_t nnz,
                           const int32_t *d_ptr, const int32_t *d_ind,
                           int32_t height, int32_t *count) {
  sporder::expect(height >= 1, "height must be >= 1");
  *count = sporder::count_pillars(ctx->ctx_impl, n, nnz, d_ptr, d_ind, height);
}

void sporder_similarity_graph(sporder_context_t ctx, int32_t n, int32_t nnz,
                              const int32_t *d_ptr, const int32_t *d_ind,
                              bool is_symmetric, int32_t k, float oversampling,
                              int32_t *d_indices, float *d_weights,
                              sporder_sim_timing_t *timing,
                              sporder_sim_stats_t *stats) {
  if (k <= 0) {
    k = std::min(32, std::max(16, 2 * (nnz / n)));
  }
  if (oversampling <= 0.0f) {
    oversampling = 2.0f;
  }

  if (stats) {
    stats->k_used = k;
    stats->m_used = ((static_cast<int32_t>(k * oversampling) + 31) / 32) * 32;
  }

  sporder::similarity_graph(ctx->ctx_impl, n, nnz, d_ptr, d_ind, is_symmetric,
                            k, oversampling, d_indices, d_weights, timing,
                            stats);
}

void sporder_similarity_graph_exact(sporder_context_t ctx, int32_t n,
                                    int32_t nnz, const int32_t *d_ptr,
                                    const int32_t *d_ind, bool is_symmetric,
                                    int32_t k, int32_t *d_indices,
                                    float *d_weights,
                                    sporder_sim_exact_timing_t *timing,
                                    sporder_sim_exact_stats_t *stats) {
  if (k <= 0) {
    k = std::min(32, std::max(16, 2 * (nnz / n)));
  }

  if (stats) {
    stats->k_used = k;
  }

  sporder::similarity_graph_exact(ctx->ctx_impl, n, nnz, d_ptr, d_ind,
                                  is_symmetric, k, d_indices, d_weights, timing,
                                  stats);
}

void sporder_chunk_greedy(sporder_context_t ctx, int32_t n, int32_t nnz,
                          const int32_t *d_ptr, const int32_t *d_ind,
                          bool is_symmetric, int32_t k, float oversampling,
                          int32_t chunk_size, int32_t *d_new_order,
                          sporder_cg_timing_t *timing,
                          sporder_cg_stats_t *stats) {
  if (k <= 0) {
    k = std::min(256, std::max(16, 1 * (nnz / n)));
  }
  if (oversampling <= 0.0f) {
    oversampling = 4.0f;
  }
  if (chunk_size <= 0) {
    chunk_size = 1024;
  }

  if (stats) {
    stats->k_used = k;
    // the similarity sketch rounds its capacity up to a multiple of 32
    stats->m_used = ((static_cast<int32_t>(k * oversampling) + 31) / 32) * 32;
    stats->chunk_size_used = chunk_size;
  }

  sporder::cg::chunk_greedy(ctx->ctx_impl, n, nnz, d_ptr, d_ind, is_symmetric,
                            k, oversampling, chunk_size, d_new_order, timing,
                            stats);
}

void sporder_chunk_similarity_greedy(sporder_context_t ctx, int32_t n,
                                     int32_t nnz, const int32_t *d_ptr,
                                     const int32_t *d_ind, bool is_symmetric,
                                     int32_t k, float oversampling,
                                     int32_t chunk_size, int32_t *d_new_order,
                                     sporder_csg_timing_t *timing,
                                     sporder_csg_stats_t *stats) {
  if (k <= 0) {
    k = std::min(256, std::max(16, 1 * (nnz / n)));
  }
  if (oversampling <= 0.0f) {
    oversampling = 4.0f;
  }
  if (chunk_size <= 0) {
    chunk_size = 1024;
  }

  if (stats) {
    stats->k_used = k;
    stats->m_used = ((static_cast<int32_t>(k * oversampling) + 31) / 32) * 32;
    stats->chunk_size_used = chunk_size;
  }

  sporder::csg::chunk_similarity_greedy(ctx->ctx_impl, n, nnz, d_ptr, d_ind,
                                        is_symmetric, k, oversampling,
                                        chunk_size, d_new_order, timing, stats);
}

void sporder_bench_cusparse_spmm(sporder_context_t ctx, int32_t n, int32_t nnz,
                                 const int32_t *d_ptr, const int32_t *d_ind,
                                 int32_t cols, int32_t warmup, int32_t repeat,
                                 uint64_t seed, float *wall_ms) {
  sporder::expect(warmup >= 0 && repeat >= 0, "warmup and repeat must be >= 0");
  sporder::expect(cols >= 1, "cols must be >= 1");
  *wall_ms = sporder::bench_cusparse_spmm(ctx->ctx_impl, n, nnz, d_ptr, d_ind,
                                          cols, warmup, repeat, seed);
}
