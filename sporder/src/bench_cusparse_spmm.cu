#include "bench_cusparse_spmm.h"
#include "common/cuda_util.cuh"
#include "common/error_checking.h"
#include "common/timer.h"
#include <curand_kernel.h>
#include <cusparse_v2.h>

namespace sporder {

namespace {

__global__ void fill_random(int32_t count, float *__restrict__ vals,
                            uint64_t seed) {
  curandState_t state;
  int32_t tid = thread_in_grid();
  int32_t stride = total_threads_in_grid();
  curand_init(seed, tid, 0, &state);
  for (int32_t i = tid; i < count; i += stride) {
    vals[i] = curand_uniform(&state);
  }
}

} // namespace

float bench_cusparse_spmm(context &ctx, int32_t n, int32_t nnz,
                          const int32_t *d_ptr, const int32_t *d_ind,
                          int32_t cols, int32_t warmup, int32_t repeat,
                          uint64_t seed) {
  cudaStream_t stream = ctx.stream();
  auto handle = ctx.cusparse_handle();

  int32_t block_dim = 256;
  int32_t nnz_grid =
      ctx.grid_dim_by_block_dim(block_dim, 4, ceil_div(nnz, block_dim));
  int64_t b_count = static_cast<int64_t>(n) * cols;
  int32_t b_grid =
      ctx.grid_dim_by_block_dim(block_dim, 4, ceil_div(b_count, block_dim));

  auto d_a_val = ctx.device_alloc_async<float>(nnz > 0 ? nnz : 1);
  fill_random<<<nnz_grid, block_dim, 0, stream>>>(nnz, d_a_val, seed ^ 0xA);
  ctx.check_last_cuda_error();

  auto d_b = ctx.device_alloc_async<float>(b_count > 0 ? b_count : 1);
  fill_random<<<b_grid, block_dim, 0, stream>>>(b_count, d_b, seed ^ 0xB);
  ctx.check_last_cuda_error();

  int64_t c_count = b_count;
  auto d_c = ctx.device_alloc_async<float>(c_count > 0 ? c_count : 1);

  cusparseSpMatDescr_t matA;
  ensure(cusparseCreateCsr(&matA, n, n, nnz, const_cast<int32_t *>(d_ptr),
                           const_cast<int32_t *>(d_ind), d_a_val,
                           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  cusparseDnMatDescr_t matB;
  ensure(cusparseCreateDnMat(&matB, n, cols, n, d_b, CUDA_R_32F,
                             CUSPARSE_ORDER_COL));

  cusparseDnMatDescr_t matC;
  ensure(cusparseCreateDnMat(&matC, n, cols, n, d_c, CUDA_R_32F,
                             CUSPARSE_ORDER_COL));

  float alpha = 1.0f, beta = 0.0f;

  size_t buf_size = 0;
  ensure(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                                 matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_CSR_ALG1, &buf_size));
  auto d_buf = ctx.device_alloc_async<char>(buf_size > 0 ? buf_size : 1);

  for (int32_t i = 0; i < warmup; ++i) {
    ensure(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                        &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1,
                        d_buf));
  }
  ctx.wait();

  cuda_timer t(stream);
  t.begin();
  for (int32_t i = 0; i < repeat; ++i) {
    ensure(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB,
                        &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1,
                        d_buf));
  }
  t.end();
  ctx.wait();

  float avg_ms = repeat > 0 ? t.elapsed_ms() / repeat : 0.0f;

  ctx.device_free_async(d_a_val);
  ctx.device_free_async(d_b);
  ctx.device_free_async(d_c);
  ctx.device_free_async(d_buf);
  ensure(cusparseDestroySpMat(matA));
  ensure(cusparseDestroyDnMat(matB));
  ensure(cusparseDestroyDnMat(matC));

  return avg_ms;
}

} // namespace sporder
