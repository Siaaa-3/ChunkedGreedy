#pragma once

struct AccSpMMHandle;

AccSpMMHandle* acc_spmm_prepare(const int* row_ptr, const int* col_idx,
                                const float* values, int num_nodes, int num_edges,
                                int force_balance = -1);
float acc_spmm_run(AccSpMMHandle* handle, const float* d_B, int feature_dim);
void acc_spmm_release(AccSpMMHandle* handle);
