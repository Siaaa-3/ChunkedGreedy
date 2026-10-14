/**
 * @file mma_tf32.cu
 * @author Haisha Zhao
 * @date 2025-04-02
 * 
 * @copyright MIT License (c) 2025 Haisha Zhao
*/

#include "mmio.hpp"
#include "tf32_comp.hpp"

__host__
void tf32_spmm(
    METCFBit<MAT_VAL_TYPE>& metcf_bit,
    vint numNodes, vint numEdges,
    float& elapsed_time,
    const vint feature_dim,
    GpuTimer& timer,
    TF32Compute& tf32_compute
) {
    vint rowWindowOffsetSize    =       metcf_bit.rowWindowOffset.size();
    vint tcOffsetSize           =       metcf_bit.tcOffset.size();
    vint sparseA2BSize          =       metcf_bit.sparseA2B.size();
    vint tcLocalBitSize         =       metcf_bit.tcLocalBit.size();
    vint dataSize               =       metcf_bit.data.size();

    vint numBlocks              =       rowWindowOffsetSize - 1;
    vint denseC_size            =       numBlocks * ROW_WINDOW * feature_dim;
    
    /*---------------- CPU Malloc ------------------*/
    vint* ptr_rowWindowOffset           =   (vint*)malloc(sizeof(vint) * rowWindowOffsetSize);
    vint* ptr_sparseA2B                 =   (vint*)malloc(sizeof(vint) * sparseA2BSize);
    vint* ptr_tcOffset                  =   (vint*)malloc(sizeof(vint) * tcOffsetSize);
    TCLOCAL_TYPE* ptr_tcLocalBit        =   (TCLOCAL_TYPE*)malloc(sizeof(TCLOCAL_TYPE) * tcLocalBitSize);
    MAT_VAL_TYPE* ptr_data              =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * dataSize);
    MAT_VAL_TYPE* DenseMatB             =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * numNodes * feature_dim);
    MAT_VAL_TYPE* DenseMatC             =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * denseC_size);

    std::copy(metcf_bit.rowWindowOffset.begin(), metcf_bit.rowWindowOffset.end(), ptr_rowWindowOffset);
    std::copy(metcf_bit.sparseA2B.begin(), metcf_bit.sparseA2B.end(), ptr_sparseA2B);
    std::copy(metcf_bit.tcOffset.begin(), metcf_bit.tcOffset.end(), ptr_tcOffset);
    std::copy(metcf_bit.tcLocalBit.begin(), metcf_bit.tcLocalBit.end(), ptr_tcLocalBit);
    std::copy(metcf_bit.data.begin(), metcf_bit.data.end(), ptr_data);
    init_vec1(numNodes * feature_dim, DenseMatB, 1.0);
    init_vec1(denseC_size, DenseMatC, 0.0);

    /*---------------- GPU Malloc ------------------*/
    vint* d_rowWindowOffset, *d_sparseA2B, *d_tcOffset;
    MAT_VAL_TYPE* d_data;
    TCLOCAL_TYPE* d_tcLocalBit;
    MAT_VAL_TYPE*   d_DenseMatB, *d_DenseMatC;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_rowWindowOffset, sizeof(vint) * rowWindowOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sparseA2B, sizeof(vint) * sparseA2BSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tcOffset, sizeof(vint) * tcOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tcLocalBit, sizeof(TCLOCAL_TYPE) * tcLocalBitSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, sizeof(MAT_VAL_TYPE) * dataSize));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_DenseMatB, sizeof(MAT_VAL_TYPE) * numNodes * feature_dim));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size));
    /*---------------- CUDA Memcpy -----------------*/
    CHECK_CUDA_ERROR(cudaMemcpy(d_rowWindowOffset, ptr_rowWindowOffset, sizeof(vint) * rowWindowOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sparseA2B, ptr_sparseA2B, sizeof(vint) * sparseA2BSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tcOffset, ptr_tcOffset, sizeof(vint) * tcOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tcLocalBit, ptr_tcLocalBit, sizeof(TCLOCAL_TYPE) * tcLocalBitSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, ptr_data, sizeof(MAT_VAL_TYPE) * dataSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_DenseMatB, DenseMatB, sizeof(MAT_VAL_TYPE) * numNodes * feature_dim, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_DenseMatC, DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyHostToDevice));

    elapsed_time = tf32_compute.compute(
        d_tcLocalBit, d_sparseA2B, d_data, d_DenseMatB, d_DenseMatC, 
        d_rowWindowOffset, d_tcOffset,
        numNodes, numBlocks, feature_dim, timer
    );

    cudaMemcpy(DenseMatC, d_DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyDeviceToHost);

    free(ptr_rowWindowOffset);
    free(ptr_sparseA2B);
    free(ptr_tcOffset);
    free(ptr_tcLocalBit);
    free(ptr_data);
    free(DenseMatB);
    free(DenseMatC);

    cudaFree(d_rowWindowOffset);
    cudaFree(d_sparseA2B);
    cudaFree(d_tcOffset);
    cudaFree(d_tcLocalBit);
    cudaFree(d_data);
    cudaFree(d_DenseMatB);
    cudaFree(d_DenseMatC);
}

__host__
void adp_tf32_spmm(
    AdpBME<MAT_VAL_TYPE>& adpbme,
    vint numNodes, vint numEdges,
    float& elapsed_time,
    const vint feature_dim,
    GpuTimer& timer,
    TF32Compute& tf32_compute
) {
    vint groupOffsetSize    =   adpbme.groupOffset.size();
    vint tcOffsetSize       =   adpbme.tcOffset.size();
    vint rowIndicesSize     =   adpbme.rowIndices.size();
    vint sparseA2BSize      =   adpbme.sparseA2B.size();
    vint tcLocalBitSize     =   adpbme.tcLocalBit.size();
    vint dataSize           =   adpbme.data.size();

    vint numBlocks          =   groupOffsetSize - 1;
    vint denseC_size        =   std::max(numBlocks * ROW_WINDOW * feature_dim, (adpbme.rowIndices.back() + 8) * feature_dim);

    /*---------------- CPU Malloc ------------------*/
    vint* ptr_groupOffset           =   (vint*)malloc(sizeof(vint) * groupOffsetSize);
    vint* ptr_tcOffset              =   (vint*)malloc(sizeof(vint) * tcOffsetSize);
    vint* ptr_rowIndices            =   (vint*)malloc(sizeof(vint) * rowIndicesSize);
    vint* ptr_sparseA2B             =   (vint*)malloc(sizeof(vint) * sparseA2BSize);
    TCLOCAL_TYPE* ptr_tcLocalBit    =   (TCLOCAL_TYPE*)malloc(sizeof(TCLOCAL_TYPE) * tcLocalBitSize);
    MAT_VAL_TYPE* ptr_data          =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * dataSize);
    MAT_VAL_TYPE* DenseMatB         =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * numNodes * feature_dim);
    MAT_VAL_TYPE* DenseMatC         =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * denseC_size);

    std::copy(adpbme.groupOffset.begin(), adpbme.groupOffset.end(), ptr_groupOffset);
    std::copy(adpbme.tcOffset.begin(), adpbme.tcOffset.end(), ptr_tcOffset);
    std::copy(adpbme.rowIndices.begin(), adpbme.rowIndices.end(), ptr_rowIndices);
    std::copy(adpbme.sparseA2B.begin(), adpbme.sparseA2B.end(), ptr_sparseA2B);
    std::copy(adpbme.tcLocalBit.begin(), adpbme.tcLocalBit.end(), ptr_tcLocalBit);
    std::copy(adpbme.data.begin(), adpbme.data.end(), ptr_data);

    init_vec1(numNodes * feature_dim, DenseMatB, 1.0);
    init_vec1(denseC_size, DenseMatC, 0.0);

    /*---------------- GPU Malloc ------------------*/
    vint* d_groupOffset, *d_tcOffset, *d_rowIndices, *d_sparseA2B;
    TCLOCAL_TYPE* d_tcLocalBit;
    MAT_VAL_TYPE* d_data, *d_DenseMatB, *d_DenseMatC;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_groupOffset, sizeof(vint) * groupOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tcOffset, sizeof(vint) * tcOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_rowIndices, sizeof(vint) * rowIndicesSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sparseA2B, sizeof(vint) * sparseA2BSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tcLocalBit, sizeof(TCLOCAL_TYPE) * tcLocalBitSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, sizeof(MAT_VAL_TYPE) * dataSize));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_DenseMatB, sizeof(MAT_VAL_TYPE) * numNodes * feature_dim));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size));
    /*---------------- CUDA Memcpy -----------------*/
    CHECK_CUDA_ERROR(cudaMemcpy(d_groupOffset, ptr_groupOffset, sizeof(vint) * groupOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tcOffset, ptr_tcOffset, sizeof(vint) * tcOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_rowIndices, ptr_rowIndices, sizeof(vint) * rowIndicesSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sparseA2B, ptr_sparseA2B, sizeof(vint) * sparseA2BSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tcLocalBit, ptr_tcLocalBit, sizeof(TCLOCAL_TYPE) * tcLocalBitSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, ptr_data, sizeof(MAT_VAL_TYPE) * dataSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_DenseMatB, DenseMatB, sizeof(MAT_VAL_TYPE) * numNodes * feature_dim, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_DenseMatC, DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyHostToDevice));

    elapsed_time = tf32_compute.adpBalanceCompute(
        d_groupOffset, d_tcOffset, d_rowIndices, d_tcLocalBit, 
        d_sparseA2B, d_data, d_DenseMatB, d_DenseMatC,
        numBlocks, numNodes, 
        feature_dim, timer
    );
    cudaMemcpy(DenseMatC, d_DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyDeviceToHost);

    free(ptr_groupOffset);
    free(ptr_tcOffset);
    free(ptr_rowIndices);
    free(ptr_sparseA2B);
    free(ptr_tcLocalBit);
    free(ptr_data);
    free(DenseMatB);
    free(DenseMatC);

    cudaFree(d_groupOffset);
    cudaFree(d_tcOffset);
    cudaFree(d_rowIndices);
    cudaFree(d_sparseA2B);
    cudaFree(d_tcLocalBit);
    cudaFree(d_data);
    cudaFree(d_DenseMatB);
    cudaFree(d_DenseMatC);
}

struct AccSpMMHandle {
    bool load_balance = false;
    vint num_nodes = 0;
    vint num_edges = 0;
    vint num_blocks = 0;
    vint denseC_row_cap = 0;

    vint cached_feature_dim = 0;
    MAT_VAL_TYPE* d_DenseMatC = nullptr;
    vint denseC_size = 0;

    vint* d_rowWindowOffset = nullptr;
    vint* d_sparseA2B = nullptr;
    vint* d_tcOffset = nullptr;
    TCLOCAL_TYPE* d_tcLocalBit = nullptr;
    MAT_VAL_TYPE* d_data = nullptr;

    vint* d_groupOffset = nullptr;
    vint* d_tcOffset_bal = nullptr;
    vint* d_rowIndices = nullptr;
    vint* d_sparseA2B_bal = nullptr;
    TCLOCAL_TYPE* d_tcLocalBit_bal = nullptr;
    MAT_VAL_TYPE* d_data_bal = nullptr;
};

__host__
static void acc_upload_metcf_sparse(
    METCFBit<MAT_VAL_TYPE>& metcf_bit, AccSpMMHandle* handle) {
    vint rowWindowOffsetSize = metcf_bit.rowWindowOffset.size();
    vint tcOffsetSize = metcf_bit.tcOffset.size();
    vint sparseA2BSize = metcf_bit.sparseA2B.size();
    vint tcLocalBitSize = metcf_bit.tcLocalBit.size();
    vint dataSize = metcf_bit.data.size();

    handle->num_blocks = rowWindowOffsetSize - 1;

    vint* ptr_rowWindowOffset = (vint*)malloc(sizeof(vint) * rowWindowOffsetSize);
    vint* ptr_sparseA2B = (vint*)malloc(sizeof(vint) * sparseA2BSize);
    vint* ptr_tcOffset = (vint*)malloc(sizeof(vint) * tcOffsetSize);
    TCLOCAL_TYPE* ptr_tcLocalBit =
        (TCLOCAL_TYPE*)malloc(sizeof(TCLOCAL_TYPE) * tcLocalBitSize);
    MAT_VAL_TYPE* ptr_data = (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * dataSize);

    std::copy(metcf_bit.rowWindowOffset.begin(), metcf_bit.rowWindowOffset.end(),
              ptr_rowWindowOffset);
    std::copy(metcf_bit.sparseA2B.begin(), metcf_bit.sparseA2B.end(), ptr_sparseA2B);
    std::copy(metcf_bit.tcOffset.begin(), metcf_bit.tcOffset.end(), ptr_tcOffset);
    std::copy(metcf_bit.tcLocalBit.begin(), metcf_bit.tcLocalBit.end(), ptr_tcLocalBit);
    std::copy(metcf_bit.data.begin(), metcf_bit.data.end(), ptr_data);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&handle->d_rowWindowOffset,
                                sizeof(vint) * rowWindowOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&handle->d_sparseA2B,
                                sizeof(vint) * sparseA2BSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&handle->d_tcOffset, sizeof(vint) * tcOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&handle->d_tcLocalBit,
                                sizeof(TCLOCAL_TYPE) * tcLocalBitSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&handle->d_data, sizeof(MAT_VAL_TYPE) * dataSize));

    CHECK_CUDA_ERROR(cudaMemcpy(handle->d_rowWindowOffset, ptr_rowWindowOffset,
                                sizeof(vint) * rowWindowOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(handle->d_sparseA2B, ptr_sparseA2B,
                                sizeof(vint) * sparseA2BSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(handle->d_tcOffset, ptr_tcOffset,
                                sizeof(vint) * tcOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(handle->d_tcLocalBit, ptr_tcLocalBit,
                                sizeof(TCLOCAL_TYPE) * tcLocalBitSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(handle->d_data, ptr_data,
                                sizeof(MAT_VAL_TYPE) * dataSize, cudaMemcpyHostToDevice));

    free(ptr_rowWindowOffset);
    free(ptr_sparseA2B);
    free(ptr_tcOffset);
    free(ptr_tcLocalBit);
    free(ptr_data);
}

__host__
static void acc_upload_adpbme_sparse(
    AdpBME<MAT_VAL_TYPE>& adpbme, AccSpMMHandle* handle) {
    vint groupOffsetSize = adpbme.groupOffset.size();
    vint tcOffsetSize = adpbme.tcOffset.size();
    vint rowIndicesSize = adpbme.rowIndices.size();
    vint sparseA2BSize = adpbme.sparseA2B.size();
    vint tcLocalBitSize = adpbme.tcLocalBit.size();
    vint dataSize = adpbme.data.size();

    handle->num_blocks = groupOffsetSize - 1;
    handle->denseC_row_cap = adpbme.rowIndices.back() + 8;

    vint* ptr_groupOffset = (vint*)malloc(sizeof(vint) * groupOffsetSize);
    vint* ptr_tcOffset = (vint*)malloc(sizeof(vint) * tcOffsetSize);
    vint* ptr_rowIndices = (vint*)malloc(sizeof(vint) * rowIndicesSize);
    vint* ptr_sparseA2B = (vint*)malloc(sizeof(vint) * sparseA2BSize);
    TCLOCAL_TYPE* ptr_tcLocalBit =
        (TCLOCAL_TYPE*)malloc(sizeof(TCLOCAL_TYPE) * tcLocalBitSize);
    MAT_VAL_TYPE* ptr_data = (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * dataSize);

    std::copy(adpbme.groupOffset.begin(), adpbme.groupOffset.end(), ptr_groupOffset);
    std::copy(adpbme.tcOffset.begin(), adpbme.tcOffset.end(), ptr_tcOffset);
    std::copy(adpbme.rowIndices.begin(), adpbme.rowIndices.end(), ptr_rowIndices);
    std::copy(adpbme.sparseA2B.begin(), adpbme.sparseA2B.end(), ptr_sparseA2B);
    std::copy(adpbme.tcLocalBit.begin(), adpbme.tcLocalBit.end(), ptr_tcLocalBit);
    std::copy(adpbme.data.begin(), adpbme.data.end(), ptr_data);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&handle->d_groupOffset, sizeof(vint) * groupOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&handle->d_tcOffset_bal, sizeof(vint) * tcOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&handle->d_rowIndices, sizeof(vint) * rowIndicesSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&handle->d_sparseA2B_bal, sizeof(vint) * sparseA2BSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&handle->d_tcLocalBit_bal,
                                sizeof(TCLOCAL_TYPE) * tcLocalBitSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&handle->d_data_bal, sizeof(MAT_VAL_TYPE) * dataSize));

    CHECK_CUDA_ERROR(cudaMemcpy(handle->d_groupOffset, ptr_groupOffset,
                                sizeof(vint) * groupOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(handle->d_tcOffset_bal, ptr_tcOffset,
                                sizeof(vint) * tcOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(handle->d_rowIndices, ptr_rowIndices,
                                sizeof(vint) * rowIndicesSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(handle->d_sparseA2B_bal, ptr_sparseA2B,
                                sizeof(vint) * sparseA2BSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(handle->d_tcLocalBit_bal, ptr_tcLocalBit,
                                sizeof(TCLOCAL_TYPE) * tcLocalBitSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(handle->d_data_bal, ptr_data,
                                sizeof(MAT_VAL_TYPE) * dataSize, cudaMemcpyHostToDevice));

    free(ptr_groupOffset);
    free(ptr_tcOffset);
    free(ptr_rowIndices);
    free(ptr_sparseA2B);
    free(ptr_tcLocalBit);
    free(ptr_data);
}

__host__
static void acc_ensure_dense_buffers(AccSpMMHandle* handle, vint feature_dim) {
    vint denseC_size = 0;
    if (handle->load_balance) {
        denseC_size = std::max(handle->num_blocks * ROW_WINDOW * feature_dim,
                               handle->denseC_row_cap * feature_dim);
    } else {
        denseC_size = handle->num_blocks * ROW_WINDOW * feature_dim;
    }

    if (handle->cached_feature_dim == feature_dim && handle->d_DenseMatC != nullptr) {
        handle->denseC_size = denseC_size;
        return;
    }

    if (handle->d_DenseMatC) cudaFree(handle->d_DenseMatC);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&handle->d_DenseMatC,
                                sizeof(MAT_VAL_TYPE) * static_cast<size_t>(denseC_size)));

    handle->cached_feature_dim = feature_dim;
    handle->denseC_size = denseC_size;
}

__host__
AccSpMMHandle* acc_spmm_prepare(const int* row_ptr, const int* col_idx,
                                const float* values, int num_nodes, int num_edges,
                                int force_balance) {
    COO<MAT_VAL_TYPE>* coo = (COO<MAT_VAL_TYPE>*)malloc(sizeof(COO<MAT_VAL_TYPE>));
    coo->rows = num_nodes;
    coo->cols = num_nodes;
    coo->nnz = num_edges;
    coo->row = (vint*)malloc(sizeof(vint) * num_edges);
    coo->col = (vint*)malloc(sizeof(vint) * num_edges);
    coo->data = (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * num_edges);

    for (int r = 0; r < num_nodes; ++r) {
        for (int idx = row_ptr[r]; idx < row_ptr[r + 1]; ++idx) {
            coo->row[idx] = r;
            coo->col[idx] = col_idx[idx];
            coo->data[idx] = values[idx];
        }
    }

    CSR<MAT_VAL_TYPE> csr(coo);
    METCFBit<MAT_VAL_TYPE> metcf_bit;
    metcf_bit.convertFromCSR(csr);

    bool load_balance = force_balance < 0
        ? balanceStrategy(
              metcf_bit, metcf_bit.tcOffset.size() - 1,
              metcf_bit.rowWindowOffset.size() - 1)
        : force_balance != 0;

    auto* handle = new AccSpMMHandle();
    handle->load_balance = load_balance;
    handle->num_nodes = num_nodes;
    handle->num_edges = num_edges;

    if (load_balance) {
        AdpBME<MAT_VAL_TYPE> adpbme;
        adpbme.CSR2AdpBME(csr, 3, 1, 2);
        acc_upload_adpbme_sparse(adpbme, handle);
    } else {
        acc_upload_metcf_sparse(metcf_bit, handle);
    }

    free(coo->row);
    free(coo->col);
    free(coo->data);
    free(coo);
    return handle;
}

__host__
float acc_spmm_run(AccSpMMHandle* handle, const float* d_B, int feature_dim) {
    acc_ensure_dense_buffers(handle, feature_dim);

    CHECK_CUDA_ERROR(cudaMemset(handle->d_DenseMatC, 0,
                                sizeof(MAT_VAL_TYPE) * static_cast<size_t>(handle->denseC_size)));

    GpuTimer timer;
    TF32Compute tf32_compute(0, 1, 512);
    float elapsed_time = 0.0f;

    if (handle->load_balance) {
        elapsed_time = tf32_compute.adpBalanceCompute(
            handle->d_groupOffset, handle->d_tcOffset_bal, handle->d_rowIndices,
            handle->d_tcLocalBit_bal, handle->d_sparseA2B_bal, handle->d_data_bal,
            const_cast<MAT_VAL_TYPE*>(d_B), handle->d_DenseMatC, handle->num_blocks,
            handle->num_nodes, feature_dim, timer);
    } else {
        elapsed_time = tf32_compute.compute(
            handle->d_tcLocalBit, handle->d_sparseA2B, handle->d_data,
            const_cast<MAT_VAL_TYPE*>(d_B), handle->d_DenseMatC, handle->d_rowWindowOffset,
            handle->d_tcOffset, handle->num_nodes, handle->num_blocks, feature_dim,
            timer);
    }
    return elapsed_time;
}

__host__
void acc_spmm_release(AccSpMMHandle* handle) {
    if (!handle) return;
    if (handle->d_DenseMatC) cudaFree(handle->d_DenseMatC);
    if (handle->d_rowWindowOffset) cudaFree(handle->d_rowWindowOffset);
    if (handle->d_sparseA2B) cudaFree(handle->d_sparseA2B);
    if (handle->d_tcOffset) cudaFree(handle->d_tcOffset);
    if (handle->d_tcLocalBit) cudaFree(handle->d_tcLocalBit);
    if (handle->d_data) cudaFree(handle->d_data);
    if (handle->d_groupOffset) cudaFree(handle->d_groupOffset);
    if (handle->d_tcOffset_bal) cudaFree(handle->d_tcOffset_bal);
    if (handle->d_rowIndices) cudaFree(handle->d_rowIndices);
    if (handle->d_sparseA2B_bal) cudaFree(handle->d_sparseA2B_bal);
    if (handle->d_tcLocalBit_bal) cudaFree(handle->d_tcLocalBit_bal);
    if (handle->d_data_bal) cudaFree(handle->d_data_bal);
    delete handle;
}

__host__
float acc_spmm_bench_run(int* row_ptr, int* col_idx, float* values,
                         int num_nodes, int num_edges, int feature_dim) {
    AccSpMMHandle* handle = acc_spmm_prepare(
        row_ptr, col_idx, values, num_nodes, num_edges, -1);
    vint dense_elems = num_nodes * feature_dim;
    MAT_VAL_TYPE* d_B = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B, sizeof(MAT_VAL_TYPE) * dense_elems));
    CHECK_CUDA_ERROR(cudaMemset(d_B, 0, sizeof(MAT_VAL_TYPE) * dense_elems));
    float elapsed = acc_spmm_run(handle, d_B, feature_dim);
    cudaFree(d_B);
    acc_spmm_release(handle);
    return elapsed;
}

__host__
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: ./mma_tf32 <matrix.mtx> <feature_dim>\n");
        return ERROR_ARGS;
    }
    
    const char* filename = argv[1];
    const int feature_dim = atoi(argv[2]);
        
    const std::string mtx_name = match_filename(std::string(filename));
    
    // Load original matrix
    COO<MAT_VAL_TYPE>* coo = (COO<MAT_VAL_TYPE>*)malloc(sizeof(COO<MAT_VAL_TYPE>));
    int read_status = read_from_mtx<MAT_VAL_TYPE>(filename, coo);
    if(read_status != SUCCESS) {
        return ERROR_READ_MTX;
    }
    
    CSR<MAT_VAL_TYPE> csr = CSR<MAT_VAL_TYPE>(coo);
    
    METCFBit<MAT_VAL_TYPE> metcf_bit;
    metcf_bit.convertFromCSR(csr);
    vint numNodes = coo->rows;
    vint numEdges = coo->nnz;
    
    bool load_balance = balanceStrategy(metcf_bit, 
                                        metcf_bit.tcOffset.size() - 1, 
                                        metcf_bit.rowWindowOffset.size() - 1);
    float elapsed_time = 0.0, spmm_throughput = 0.0;
    
    GpuTimer timer;
    TF32Compute tf32_compute;

    if(load_balance) {
        // Adaptive balance Tensor core operations
        AdpBME<MAT_VAL_TYPE> adpbme;
        adpbme.CSR2AdpBME(csr, 3, 1, 2);
        TF32Compute tf32_compute(10, 10, 128);
        adp_tf32_spmm(adpbme, numNodes, numEdges, elapsed_time, feature_dim, timer, tf32_compute);
            
    } else {
        // Tensor core operations
        TF32Compute tf32_compute;
        tf32_spmm(metcf_bit, numNodes, numEdges, elapsed_time, feature_dim, timer, tf32_compute);
    }
    spmm_throughput = (float(numEdges) * float(feature_dim) * 2.0 * 1000.) 
                    / (elapsed_time * 1000. * 1000. * 1000.);

    std::ofstream outFile("result.csv", std::ios::app);
    if (!outFile) {
        std::cerr << "Error Opening result.csv" << std::endl;
    }
    outFile << mtx_name << ","<< feature_dim << "," << elapsed_time << "," << spmm_throughput << "\n";
    outFile.close();  

    free(coo);
    return 0;
}
