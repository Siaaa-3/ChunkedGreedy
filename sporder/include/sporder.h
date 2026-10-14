#ifndef SPORDER_H
#define SPORDER_H

#include <stdbool.h>
#include <stdint.h>

#define SPORDER_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

/*
 * sporder -- GPU sparse matrix reordering algorithms.
 *
 * Conventions shared by every entry point:
 *
 *   - Synchronous: calls return only after their results (device
 *     outputs and host out-parameters) are ready to use.
 *   - d_-prefixed pointers are device memory on the context's device;
 *     all other pointers are host memory.
 *   - Inputs must already be resident on the device when a call is
 *     made.  The library runs on a private CUDA stream, so inputs
 *     produced asynchronously on a non-default stream must be
 *     synchronized by the caller first.
 *   - CSR patterns are square (n rows and columns), zero-based, with
 *     column indices sorted within each row; values are not needed.
 *   - Invalid arguments and CUDA errors abort the process with a
 *     diagnostic on stderr.
 *   - Permutations are in new2old form: d_new_order[p] = the original
 *     row index that belongs in new position p.  So new row p
 *     contains old row d_new_order[p].
 *   - Timing and diagnostics: every entry point accepts optional
 *     timing and/or stats structs (NULL to skip).  wall_ms always
 *     reflects the full cost of the configured call: with diagnostics
 *     enabled it includes their overhead, without them it measures
 *     computation only.  CUDA-event stage spans are unaffected by
 *     diagnostics (they run between stages, not inside timed spans),
 *     but may shift slightly due to cache interference.  For precise
 *     performance measurement, collect timing and stats separately.
 */

typedef struct sporder_context_struct *sporder_context_t;

/*
 * A context owns a CUDA stream and a cuSPARSE handle, bound to the
 * device that is current at creation time.  Use a context from one
 * thread at a time; create one context per thread (or per device) for
 * concurrent calls.
 */
SPORDER_API sporder_context_t sporder_context_create(void);
SPORDER_API void sporder_context_destroy(sporder_context_t h);

typedef struct {
  int32_t k;          /* 0 = auto (clamp(2*nnz/n, 16, 32)) */
  float oversampling; /* 0 = auto (2.0); k * oversampling <= 512 */
  int32_t max_rounds; /* 0 = run to convergence */
} sporder_options_t;

/*
 * Per-stage timings for hierarchical matching; see conventions.
 * Per-level arrays must be NULL or hold num_levels elements.
 */
typedef struct {
  double wall_ms;
  float csr2csc_ms;
  float similarity_ms;
  float matching_ms;
  float advance_ms;
  float *matching_per_level_ms;
  float *advance_per_level_ms;
} sporder_hm_timing_t;

/*
 * Per-level diagnostics for hierarchical matching; see conventions.
 * Per-level arrays must be NULL or hold num_levels elements.
 *
 * Useful derived quantities (n = number of rows):
 *   total degree           = avg_degree * n
 *   isolated fraction      = zero_degree_rows / n
 *                            (rows no matching can improve)
 *   matched fraction of l  = 2 * matched_pairs[l] / level_size[l]
 *   singletons of level l  = level_size[l] - 2 * matched_pairs[l]
 */
typedef struct {
  int32_t k_used;           /* top-k after auto-selection */
  int32_t m_used;           /* space-saving sketch capacity */
  double avg_degree;        /* mean top-k graph degree per row */
  double avg_edge_weight;   /* mean weight over top-k graph edges */
  int32_t zero_degree_rows; /* isolated in the top-k graph; unmatchable */
  int32_t *level_size;      /* [num_levels] nodes entering the level */
  int32_t *matched_pairs;   /* [num_levels] mutual matches formed */
  int32_t *rounds;          /* [num_levels] suitor rounds run */
} sporder_hm_stats_t;

/*
 * Compute a similarity-grouping row permutation of a CSR pattern.
 *
 * Builds a binary cluster hierarchy over the rows: a Dice-similarity
 * top-k graph is computed once, then each of the num_levels levels
 * pairs up (super-)nodes by suitor matching.  The DFS order is
 * written to d_new_order (a permutation of 0..n-1 in new2old form):
 * the output's row p contains the input's row d_new_order[p], which
 * places similar rows in the same aligned window of 2^num_levels
 * rows.  Apply it with sporder_reorder().
 *
 * Arguments:
 *   n, nnz       - pattern size; nnz must equal d_ptr[n]
 *   d_ptr, d_ind - CSR pattern, lengths n+1 and nnz
 *   is_symmetric - the pattern equals its transpose (skips an
 *                  internal transposition)
 *   num_levels   - hierarchy levels, in [1, 30]; choose log2 of the
 *                  target row-window height
 *   opts         - tuning knobs; NULL or zeroed fields = defaults
 *   d_new_order  - output permutation, length n (new2old)
 *   timing       - optional timings, NULL to skip
 *   stats        - optional diagnostics, NULL to skip
 */
SPORDER_API void sporder_hierarchical_matching(
    sporder_context_t ctx, int32_t n, int32_t nnz, const int32_t *d_ptr,
    const int32_t *d_ind, bool is_symmetric, int32_t num_levels,
    const sporder_options_t *opts, int32_t *d_new_order,
    sporder_hm_timing_t *timing, sporder_hm_stats_t *stats);

/*
 * Per-stage timings for chunk_greedy; see conventions.
 */
typedef struct {
  double wall_ms;
  float csr2csc_ms;
  float similarity_ms;
  float chunk_greedy_ms;
} sporder_cg_timing_t;

/*
 * Diagnostics for chunk_greedy; see conventions.  Chain counters
 * are collected inside the greedy kernel at negligible cost.
 *
 * Useful derived quantities (n = number of rows):
 *   num_chunks   = ceil(n / chunk_size_used)
 *   chains       = num_chunks + chain_restarts
 *   links        = n - chains        (edges the greedy followed)
 *   break rate   = chain_restarts / (n - num_chunks)
 *                  (0 = perfect chaining; no links exist when
 *                  chunk_size_used == 1)
 *   selection advantage = avg_followed_weight / avg_edge_weight
 *                  (> 1: the greedy exploits the graph; <= 1: links
 *                  are leftovers -- partition damage or exhaustion)
 *   random-order cut baseline = 1 - (chunk_size_used - 1) / (n - 1)
 *                  (cut_weight_fraction near this baseline means the
 *                  input order has no locality for chunking to keep;
 *                  well below it means rows are already clustered)
 */
typedef struct {
  int32_t k_used;             /* top-k after auto-selection */
  int32_t m_used;             /* space-saving sketch capacity */
  double avg_degree;          /* mean top-k graph degree per row */
  double avg_edge_weight;     /* mean weight over top-k graph edges */
  int32_t zero_degree_rows;   /* isolated in the top-k graph */
  int32_t chunk_size_used;    /* chunk size after auto-selection */
  int32_t chain_restarts;     /* chain breaks beyond one chain per chunk */
  double avg_followed_weight; /* mean weight of the edges the greedy
                                 followed, one per chain step;
                                 restart transitions excluded */
  double cut_weight_fraction; /* edge weight crossing chunk boundaries
                                over total edge weight */
} sporder_cg_stats_t;

/*
 * One-shot chunked greedy seriation: similarity_top_k is computed
 * internally, then rows are partitioned into fixed-size chunks and a
 * warp-cooperative greedy chain runs inside each chunk.  The chain
 * order is returned as a new2old permutation.  chunk_size=0 selects
 * a default (1024).  The output is suitable for sporder_reorder().
 */
SPORDER_API void sporder_chunk_greedy(sporder_context_t ctx, int32_t n,
                                      int32_t nnz, const int32_t *d_ptr,
                                      const int32_t *d_ind, bool is_symmetric,
                                      int32_t k, float oversampling,
                                      int32_t chunk_size, int32_t *d_new_order,
                                      sporder_cg_timing_t *timing,
                                      sporder_cg_stats_t *stats);

/*
 * One-shot per-chunk similarity + greedy seriation.
 *
 * Partitions rows into fixed-size chunks, computes the top-k
 * Dice-similarity graph within each chunk independently (single
 * kernel launch using multi-chunk similarity), then runs
 * warp-cooperative greedy chaining inside each chunk.
 *
 * The similarity work is O(chunk_size * d^2) per chunk, totalling
 * O(chunk_size * d^2) independent of n -- a fraction C/n of global
 * similarity cost.  Chunk quality is bounded: rows only find
 * neighbours inside their own chunk.
 *
 * Arguments match chunk_greedy.  For precise performance
 * measurement, collect timing and stats separately.
 */
/*
 * Per-stage timings for chunk_similarity_greedy; see conventions.
 */
typedef struct {
  double wall_ms;
  float csr2csc_ms;
  float similarity_ms;
  float chunk_greedy_ms;
} sporder_csg_timing_t;

/*
 * Diagnostics for chunk_similarity_greedy; see conventions.
 */
typedef struct {
  int32_t k_used;
  int32_t m_used;
  double avg_degree;
  double avg_edge_weight;
  int32_t zero_degree_rows;
  int32_t chunk_size_used;
  int32_t chain_restarts;
  double avg_followed_weight;
} sporder_csg_stats_t;

SPORDER_API void sporder_chunk_similarity_greedy(
    sporder_context_t ctx, int32_t n, int32_t nnz, const int32_t *d_ptr,
    const int32_t *d_ind, bool is_symmetric, int32_t k, float oversampling,
    int32_t chunk_size, int32_t *d_new_order, sporder_csg_timing_t *timing,
    sporder_csg_stats_t *stats);

typedef enum {
  SPORDER_REORDER_ROWS = 0,
  SPORDER_REORDER_SYMMETRIC = 1,
} sporder_reorder_mode_t;

/*
 * Apply a row (or symmetric row+column) permutation to a CSR pattern.
 *
 * d_new_order must be a permutation of 0..n-1 in new2old form:
 * new row p contains old row d_new_order[p].  The output has sorted
 * column indices within each row in both modes (SYMMETRIC re-sorts
 * after remapping the columns).  When d_val_old is non-NULL, the
 * corresponding values are reordered alongside the column indices.
 *
 * Arguments:
 *   n, nnz               - pattern size; nnz must equal d_ptr_old[n]
 *   d_ptr_old, d_ind_old - input CSR pattern, lengths n+1 and nnz
 *   d_val_old            - input values, length nnz, NULL if none
 *   d_new_order          - permutation, length n (new2old)
 *   mode                 - which sides of the pattern to permute
 *   d_ptr_new, d_ind_new - output CSR pattern, lengths n+1 and nnz
 *   d_val_new            - output values, length nnz, NULL if none
 */
SPORDER_API void
sporder_reorder(sporder_context_t ctx, int32_t n, int32_t nnz,
                const int32_t *d_ptr_old, const int32_t *d_ind_old,
                const float *d_val_old, const int32_t *d_new_order,
                sporder_reorder_mode_t mode, int32_t *d_ptr_new,
                int32_t *d_ind_new, float *d_val_new);

/*
 * Count nonzero columns over consecutive row windows.
 *
 * Rows are grouped into ceil(n / height) consecutive windows of
 * `height` rows (the last window may be shorter).  A pillar is a
 * column holding at least one nonzero inside a window; *count receives
 * the total over all windows -- the column-tile count of a blocked
 * SpMM with tile width 1, i.e. the quantity row reordering tries to
 * reduce.  Does not require sorted or duplicate-free column indices.
 *
 * Arguments:
 *   n, nnz       - pattern size; nnz must equal d_ptr[n]
 *   d_ptr, d_ind - CSR pattern, lengths n+1 and nnz
 *   height       - row window height, >= 1
 *   count        - host pointer receiving the pillar total
 */
SPORDER_API void sporder_count_pillars(sporder_context_t ctx, int32_t n,
                                       int32_t nnz, const int32_t *d_ptr,
                                       const int32_t *d_ind, int32_t height,
                                       int32_t *count);

/*
 * Benchmark cuSPARSE SpMM (sparse A x dense B) for a square CSR pattern.
 *
 * Generates reproducible random float values for A (CSR) and a cols-column
 * dense B (column-major) from the given seed, then runs cusparseSpMM
 * `warmup` + `repeat` times (between 0 and INT32_MAX each).  Returns the
 * average wall-clock time per repeat in milliseconds via *wall_ms.
 *
 * The metric measures raw cuSPARSE SpMM throughput, not including the
 * value-generation cost (values are generated once upfront).
 *
 * Pattern-only: d_ptr and d_ind define the sparsity; no value array is
 * consumed.  Seed reproducibility relies only on the pattern structure.
 */
SPORDER_API void sporder_bench_cusparse_spmm(sporder_context_t ctx, int32_t n,
                                             int32_t nnz, const int32_t *d_ptr,
                                             const int32_t *d_ind, int32_t cols,
                                             int32_t warmup, int32_t repeat,
                                             uint64_t seed, float *wall_ms);

/*
 * Build the weighted top-k Dice-similarity graph of a CSR pattern.
 *
 * For each row r, up to k neighbours s with the largest Dice
 * similarity 2|r&s|/(|r|+|s|) are selected.  The result is a flat
 * n x k array per output: d_indices[r*k + j] and d_weights[r*k + j].
 * Unused entries are marked with index = -1 and weight = 0.
 */
/*
 * Per-stage timings for similarity_graph; see conventions.
 */
typedef struct {
  double wall_ms;
  float csr2csc_ms;
  float similarity_ms;
  float unpack_ms;
} sporder_sim_timing_t;

/*
 * Graph-quality diagnostics for similarity_graph; see conventions.
 */
typedef struct {
  int32_t k_used;           /* top-k after auto-selection */
  int32_t m_used;           /* space-saving sketch capacity */
  double avg_degree;        /* mean graph degree per row */
  double avg_edge_weight;   /* mean weight over graph edges */
  int32_t zero_degree_rows; /* rows with no candidates */
} sporder_sim_stats_t;

SPORDER_API void sporder_similarity_graph(
    sporder_context_t ctx, int32_t n, int32_t nnz, const int32_t *d_ptr,
    const int32_t *d_ind, bool is_symmetric, int32_t k, float oversampling,
    int32_t *d_indices, // out, length n*k
    float *d_weights,   // out, length n*k
    sporder_sim_timing_t *timing, sporder_sim_stats_t *stats);

/*
 * Compute the exact top-k Dice-similarity graph via cuSPARSE SpGEMM.
 *
 * For each row r, the exact intersection size with every other row
 * is computed and converted to Dice similarity; then the top k are
 * retained.  The output format matches sporder_similarity_graph.
 *
 * This is intended as a correctness baseline and for small-to-medium
 * validation: the exact computation materializes O(n * d^2) edges
 * internally (its nnz is reported in stats->c_nnz), which is
 * dramatically more expensive than the sketch.  oversampling is not
 * applicable (no sketch); k is the only selectivity parameter.
 */
/*
 * Per-stage timings for similarity_graph_exact; see conventions.
 */
typedef struct {
  double wall_ms;
  float csr2csc_ms;
  float spgemm_ms; /* SpGEMM compute + copy */
  float dice_ms;   /* count-to-Dice conversion */
  float sort_ms;   /* per-row descending sort */
  float select_ms; /* top-k selection */
} sporder_sim_exact_timing_t;

/*
 * Diagnostics for similarity_graph_exact; see conventions.
 */
typedef struct {
  int32_t k_used;           /* top-k after auto-selection */
  int64_t c_nnz;            /* nonzeros of the intermediate C = A @ A^T */
  double avg_degree;        /* mean graph degree per row */
  double avg_edge_weight;   /* mean weight over graph edges */
  int32_t zero_degree_rows; /* rows with no candidates */
} sporder_sim_exact_stats_t;

SPORDER_API void sporder_similarity_graph_exact(
    sporder_context_t ctx, int32_t n, int32_t nnz, const int32_t *d_ptr,
    const int32_t *d_ind, bool is_symmetric, int32_t k, int32_t *d_indices,
    float *d_weights, sporder_sim_exact_timing_t *timing,
    sporder_sim_exact_stats_t *stats);

#ifdef __cplusplus
}
#endif

#endif // SPORDER_H
