#include "common/csr2csc.h"
#include "common/cuda_util.cuh"
#include "common/timer.h"
#include "similarity_graph_exact.h"
#include <chrono>
#include <cub/cub.cuh>
#include <cusparse_v2.h>

namespace sporder {

namespace {

__global__ void fill_ones(int32_t nnz, float *__restrict__ vals) {
  for (int32_t i = thread_in_grid(); i < nnz; i += total_threads_in_grid()) {
    vals[i] = 1.0f;
  }
}

__global__ void build_row_len(int32_t n, const int32_t *__restrict__ ptr,
                              int32_t *__restrict__ len) {
  for (int32_t i = thread_in_grid(); i < n; i += total_threads_in_grid()) {
    len[i] = ptr[i + 1] - ptr[i];
  }
}

__global__ void dice_in_place(int32_t n, const int32_t *__restrict__ c_ptr,
                              const int32_t *__restrict__ c_ind,
                              float *__restrict__ c_val,
                              const int32_t *__restrict__ row_len) {
  for (int32_t r = warp_in_grid(); r < n; r += total_warps_in_grid()) {
    int32_t deg_r = (lane_in_warp() == 0) ? row_len[r] : 0;
    deg_r = __shfl_sync(0xFFFFFFFFu, deg_r, 0);
    int32_t begin = c_ptr[r], end = c_ptr[r + 1];
    for (int32_t p = begin + lane_in_warp(); p < end; p += 32) {
      int32_t s = c_ind[p];
      float count = c_val[p];
      if (r == s) {
        c_val[p] = 0.0f;
      } else {
        int32_t deg_s = row_len[s];
        c_val[p] = 2.0f * count /
                   (static_cast<float>(deg_r) + static_cast<float>(deg_s));
      }
    }
  }
}

__global__ void
copy_top_k(int32_t n, int32_t k, const int32_t *__restrict__ c_ptr,
           const int32_t *__restrict__ c_ind, const float *__restrict__ c_val,
           int32_t *__restrict__ d_indices, float *__restrict__ d_weights) {
  for (int32_t r = thread_in_grid(); r < n; r += total_threads_in_grid()) {
    int32_t begin = c_ptr[r], end = c_ptr[r + 1];
    int32_t count = end - begin;
    if (count > k) {
      count = k;
    }
    int32_t dst = r * k;
    for (int32_t j = 0; j < count; ++j) {
      d_indices[dst + j] = c_ind[begin + j];
      d_weights[dst + j] = c_val[begin + j];
    }
    for (int32_t j = count; j < k; ++j) {
      d_indices[dst + j] = -1;
      d_weights[dst + j] = 0.0f;
    }
  }
}

} // namespace
void similarity_graph_exact(context &ctx, int32_t n, int32_t nnz,
                            const int32_t *d_ptr, const int32_t *d_ind,
                            bool is_symmetric, int32_t k, int32_t *d_indices,
                            float *d_weights,
                            sporder_sim_exact_timing_t *timing,
                            sporder_sim_exact_stats_t *stats) {
  auto t0 = std::chrono::steady_clock::now();
  cudaStream_t stream = ctx.stream();
  auto handle = ctx.cusparse_handle();
  cusparseSetStream(handle, stream);

  int32_t block_dim = 256;
  int32_t nnz_grid =
      ctx.grid_dim_by_block_dim(block_dim, 4, ceil_div(nnz, block_dim));
  int32_t row_grid =
      ctx.grid_dim_by_block_dim(block_dim, 4, ceil_div(n, block_dim));

  // --- resolve CSC (CSR of A^T) ---
  cuda_timer t_csr2csc(stream);
  t_csr2csc.begin();
  resolved_csc csc = resolve_csc(ctx, n, nnz, d_ptr, d_ind, is_symmetric);
  t_csr2csc.end();

  // --- fill value arrays with 1.0 ---
  auto d_a_val = ctx.device_alloc_async<float>(nnz);
  auto d_b_val = ctx.device_alloc_async<float>(nnz);
  fill_ones<<<nnz_grid, block_dim, 0, stream>>>(nnz, d_a_val);
  ctx.check_last_cuda_error();
  fill_ones<<<nnz_grid, block_dim, 0, stream>>>(nnz, d_b_val);
  ctx.check_last_cuda_error();

  // --- build row degrees for Dice conversion ---
  auto d_row_len = ctx.device_alloc_async<int32_t>(n);
  build_row_len<<<row_grid, block_dim, 0, stream>>>(n, d_ptr, d_row_len);
  ctx.check_last_cuda_error();

  // --- SpGEMM: C = A @ B = A @ A^T (intersection counts) ---
  cuda_timer t_spgemm(stream);
  t_spgemm.begin();

  cusparseSpMatDescr_t matA, matB, matC;
  cusparseCreateCsr(&matA, n, n, nnz, const_cast<int32_t *>(d_ptr),
                    const_cast<int32_t *>(d_ind), d_a_val, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&matB, n, n, nnz, const_cast<int32_t *>(csc.d_ptr),
                    const_cast<int32_t *>(csc.d_ind), d_b_val,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  auto d_c_ptr = ctx.device_alloc_async<int32_t>(n + 1);
  ctx.device_memset_async(d_c_ptr, 0, (n + 1) * sizeof(int32_t));
  cusparseCreateCsr(&matC, n, n, 0, d_c_ptr, nullptr, nullptr,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  cusparseSpGEMMDescr_t spgemm_desc;
  cusparseSpGEMM_createDescr(&spgemm_desc);
  float one = 1.0f, zero = 0.0f;

  size_t buf1_bytes = 0;
  ensure(cusparseSpGEMM_workEstimation(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB, &zero, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc, &buf1_bytes, nullptr));
  auto d_buf1 = ctx.device_alloc_async<char>(buf1_bytes > 0 ? buf1_bytes : 1);
  ensure(cusparseSpGEMM_workEstimation(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB, &zero, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc, &buf1_bytes, d_buf1));

  size_t buf2_bytes = 0;
  ensure(cusparseSpGEMM_compute(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB, &zero, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc, &buf2_bytes, nullptr));
  auto d_buf2 = ctx.device_alloc_async<char>(buf2_bytes > 0 ? buf2_bytes : 1);
  ensure(cusparseSpGEMM_compute(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB, &zero, matC,
      CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, spgemm_desc, &buf2_bytes, d_buf2));

  int64_t c_rows = 0, c_cols = 0, c_nnz = 0;
  ensure(cusparseSpMatGetSize(matC, &c_rows, &c_cols, &c_nnz));

  if (c_nnz > 0) {
    auto d_c_ind = ctx.device_alloc_async<int32_t>(c_nnz);
    auto d_c_val = ctx.device_alloc_async<float>(c_nnz);
    ensure(cusparseCsrSetPointers(matC, d_c_ptr, d_c_ind, d_c_val));
    ensure(cusparseSpGEMM_copy(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE, &one, matA,
                               matB, &zero, matC, CUDA_R_32F,
                               CUSPARSE_SPGEMM_DEFAULT, spgemm_desc));
    t_spgemm.end();

    // --- convert counts to Dice weights in place ---
    cuda_timer t_pack(stream);
    t_pack.begin();
    int32_t warp_dim = 256;
    int32_t warp_grid2 =
        ctx.grid_dim_by_block_dim(warp_dim, 4, ceil_div(n, warp_dim / 32));
    dice_in_place<<<warp_grid2, warp_dim, 0, stream>>>(n, d_c_ptr, d_c_ind,
                                                       d_c_val, d_row_len);
    ctx.check_last_cuda_error();
    t_pack.end();

    // --- sort each row by weight descending (values follow keys) ---
    cuda_timer t_sort(stream);
    t_sort.begin();
    auto d_c_val_sorted = ctx.device_alloc_async<float>(c_nnz);
    auto d_c_ind_sorted = ctx.device_alloc_async<int32_t>(c_nnz);
    size_t sort_temp = 0;
    cub::DeviceSegmentedSort::SortPairsDescending(
        nullptr, sort_temp, d_c_val, d_c_val_sorted, d_c_ind, d_c_ind_sorted,
        c_nnz, n, d_c_ptr, d_c_ptr + 1, stream);
    auto d_sort_temp = ctx.device_alloc_async(sort_temp);
    cub::DeviceSegmentedSort::SortPairsDescending(
        d_sort_temp, sort_temp, d_c_val, d_c_val_sorted, d_c_ind,
        d_c_ind_sorted, c_nnz, n, d_c_ptr, d_c_ptr + 1, stream);
    ctx.device_free_async(d_sort_temp);
    ctx.device_free_async(d_c_val);
    ctx.device_free_async(d_c_ind);
    d_c_val = d_c_val_sorted;
    d_c_ind = d_c_ind_sorted;
    t_sort.end();

    // --- copy first k entries per row to user output ---
    cuda_timer t_copy(stream);
    t_copy.begin();
    copy_top_k<<<row_grid, block_dim, 0, stream>>>(
        n, k, d_c_ptr, d_c_ind, d_c_val, d_indices, d_weights);
    ctx.check_last_cuda_error();
    t_copy.end();

    ctx.device_free_async(d_c_ind);
    ctx.device_free_async(d_c_val);

    ctx.wait();

    if (timing) {
      timing->csr2csc_ms = t_csr2csc.elapsed_ms();
      timing->spgemm_ms = t_spgemm.elapsed_ms();
      timing->dice_ms = t_pack.elapsed_ms();
      timing->sort_ms = t_sort.elapsed_ms();
      timing->select_ms = t_copy.elapsed_ms();
    }

    if (stats) {
      stats->c_nnz = c_nnz;
      // Count non-minus-one entries and total weight from the output.
      // D2H is fine here: this path is for small-to-medium validation.
      std::vector<int32_t> h_idx(n * k);
      std::vector<float> h_wgt(n * k);
      cudaMemcpy(h_idx.data(), d_indices, n * k * sizeof(int32_t),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(h_wgt.data(), d_weights, n * k * sizeof(float),
                 cudaMemcpyDeviceToHost);
      double total_deg = 0.0, total_wgt = 0.0;
      int32_t zero_deg = 0;
      for (int32_t r = 0; r < n; ++r) {
        int32_t deg = 0;
        for (int32_t j = 0; j < k && h_idx[r * k + j] >= 0; ++j) {
          deg += 1;
          total_wgt += h_wgt[r * k + j];
        }
        total_deg += deg;
        zero_deg += (deg == 0);
      }
      stats->avg_degree = n > 0 ? total_deg / n : 0.0;
      stats->avg_edge_weight = total_deg > 0 ? total_wgt / total_deg : 0.0;
      stats->zero_degree_rows = zero_deg;
    }
  } else {
    t_spgemm.end();
    ctx.device_memset_async(d_indices, 0xFF,
                            static_cast<size_t>(n) * k * sizeof(int32_t));
    ctx.device_memset_async(d_weights, 0,
                            static_cast<size_t>(n) * k * sizeof(float));
    if (stats) {
      stats->c_nnz = 0;
    }
  }

  // --- cleanup common ---
  ctx.device_free_async(d_a_val);
  ctx.device_free_async(d_b_val);
  ctx.device_free_async(d_c_ptr);
  ctx.device_free_async(d_row_len);
  ctx.device_free_async(d_buf1);
  ctx.device_free_async(d_buf2);
  release_csc(ctx, csc);
  cusparseDestroySpMat(matA);
  cusparseDestroySpMat(matB);
  cusparseDestroySpMat(matC);
  cusparseSpGEMM_destroyDescr(spgemm_desc);

  if (timing) {
    timing->wall_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
  }
}

} // namespace sporder
