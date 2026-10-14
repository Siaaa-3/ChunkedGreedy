/*
 * smoke.c -- C API contract guard.
 *
 * Verifies that sporder.h compiles as plain C and that libsporder
 * links into and runs from a C program; every public entry point is
 * called once.  This is deliberately minimal: functional coverage
 * belongs in a Python-level suite, not here.
 * (The documented synchronous-call contract is not verifiable from
 * outside the API -- default-stream reads are masked by implicit
 * synchronization, and the context stream is private -- so it is
 * guarded by code convention instead: every entry point ends in
 * exactly one wait.)
 */
#include <cuda_runtime.h>
#include <sporder.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA(call)                                                             \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(err_));                                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT(cond, msg)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "smoke: FAILED: %s\n", msg);                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

/* copies a device array back and checks it is a permutation of 0..n-1 */
static int valid_permutation(const int32_t *d_perm, int32_t n) {
  int32_t *h = malloc(n * sizeof(int32_t));
  char *seen = calloc(n, 1);
  int ok = cudaMemcpy(h, d_perm, n * sizeof(int32_t), cudaMemcpyDeviceToHost) ==
           cudaSuccess;
  for (int32_t i = 0; ok && i < n; ++i) {
    if (h[i] < 0 || h[i] >= n || seen[h[i]]) {
      ok = 0;
    } else {
      seen[h[i]] = 1;
    }
  }
  free(h);
  free(seen);
  return ok;
}

int main(void) {
  const int32_t n = 1000;

  /* symmetric band pattern: row i has columns {i-2..i+2} in [0, n) */
  int32_t *h_ptr = malloc((n + 1) * sizeof(int32_t));
  int32_t *h_ind = malloc((size_t)n * 5 * sizeof(int32_t));
  int32_t nnz = 0;
  h_ptr[0] = 0;
  for (int32_t i = 0; i < n; ++i) {
    for (int32_t j = i - 2; j <= i + 2; ++j) {
      if (j >= 0 && j < n) {
        h_ind[nnz++] = j;
      }
    }
    h_ptr[i + 1] = nnz;
  }

  int32_t *d_ptr, *d_ind, *d_new_order, *d_ptr_new, *d_ind_new;
  CUDA(cudaMalloc((void **)&d_ptr, (n + 1) * sizeof(int32_t)));
  CUDA(cudaMalloc((void **)&d_ind, nnz * sizeof(int32_t)));
  CUDA(cudaMalloc((void **)&d_new_order, n * sizeof(int32_t)));
  CUDA(cudaMalloc((void **)&d_ptr_new, (n + 1) * sizeof(int32_t)));
  CUDA(cudaMalloc((void **)&d_ind_new, nnz * sizeof(int32_t)));
  CUDA(cudaMemcpy(d_ptr, h_ptr, (n + 1) * sizeof(int32_t),
                  cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(d_ind, h_ind, nnz * sizeof(int32_t), cudaMemcpyHostToDevice));

  sporder_context_t ctx = sporder_context_create();

  /* Diagnostics are NULL throughout: populated timing/stats paths are
   * functional coverage, which does not belong in this file. */

  /* hierarchical matching */
  sporder_hierarchical_matching(ctx, n, nnz, d_ptr, d_ind, true, 3, NULL,
                                d_new_order, NULL, NULL);
  EXPECT(valid_permutation(d_new_order, n), "new_order is a valid permutation");

  /* chunk greedy */
  sporder_chunk_greedy(ctx, n, nnz, d_ptr, d_ind, true, 0, 0.0f, 0, d_new_order,
                       NULL, NULL);
  EXPECT(valid_permutation(d_new_order, n),
         "chunk_greedy new_order is a valid permutation");

  /* reorder + count_pillars round-trip on the last permutation */
  sporder_reorder(ctx, n, nnz, d_ptr, d_ind, NULL, d_new_order,
                  SPORDER_REORDER_ROWS, d_ptr_new, d_ind_new, NULL);
  int32_t count = -1;
  sporder_count_pillars(ctx, n, nnz, d_ptr_new, d_ind_new, 1, &count);
  EXPECT(count == nnz, "height=1 pillars == nnz (band has no duplicates)");

  /* similarity graph: every band row has at least one neighbour */
  const int32_t k = 8;
  int32_t *d_sim_idx;
  float *d_sim_wgt;
  CUDA(cudaMalloc((void **)&d_sim_idx, (size_t)n * k * sizeof(int32_t)));
  CUDA(cudaMalloc((void **)&d_sim_wgt, (size_t)n * k * sizeof(float)));
  sporder_similarity_graph(ctx, n, nnz, d_ptr, d_ind, true, k, 4.0f, d_sim_idx,
                           d_sim_wgt, NULL, NULL);
  int32_t best0 = -1;
  CUDA(cudaMemcpy(&best0, d_sim_idx, sizeof(best0), cudaMemcpyDeviceToHost));
  EXPECT(best0 > 0 && best0 < n, "row 0 has a valid best neighbour");

  sporder_context_destroy(ctx);
  cudaFree(d_ptr);
  cudaFree(d_ind);
  cudaFree(d_new_order);
  cudaFree(d_ptr_new);
  cudaFree(d_ind_new);
  cudaFree(d_sim_idx);
  cudaFree(d_sim_wgt);
  free(h_ptr);
  free(h_ind);

  printf("smoke: OK\n");
  return 0;
}
