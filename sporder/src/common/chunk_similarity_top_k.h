#pragma once
#include "common/context.h"
#include "common/weighted_index.h"
#include <cstdint>

namespace sporder {

/*
 * multi_chunk_similarity_top_k -- all chunks in a single kernel launch.
 *
 * Partitions rows into ceil(n / chunk_size) chunks and computes the
 * per-chunk top-k similarity graph for all chunks concurrently in one
 * GPU kernel launch.  The Occupancy API determines block size, and
 * work is distributed across the GPU with ~2 waves.
 *
 * The output d_neighbors is populated for all rows.
 */
void multi_chunk_similarity_top_k(context &ctx, int32_t n, int32_t k,
                                  int32_t chunk_size, float oversampling,
                                  const int32_t *d_ptr, const int32_t *d_ind,
                                  const int32_t *d_col_ptr,
                                  const int32_t *d_row_ind,
                                  weighted_index *d_neighbors);

} // namespace sporder
