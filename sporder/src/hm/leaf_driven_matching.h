#pragma once
#include "common/context.h"
#include "common/weighted_index.h"

namespace sporder::hm {

/*
 * leaf_driven_matching -- one level of suitor matching between super-nodes.
 *
 * Every leaf proposes on behalf of its current super-node
 * (d_old2new[leaf] >> level) to the super-nodes of its top-k similarity
 * neighbors; mutual best proposals become partners.  On completion
 * d_partners[v] (length level_size) is v's partner, or -1 if unmatched.
 * Returns the number of suitor rounds executed.
 *
 * Synchronizes the context stream once per round for host-side loop
 * control, but the final d_partners write is stream-ordered: sync the
 * stream before reading it from the host.
 */
int32_t leaf_driven_matching(context &ctx, int32_t num_leaves,
                             int32_t level_size, int32_t level,
                             int32_t max_neighbors, int32_t max_rounds,
                             const weighted_index *d_neighbors,
                             const int32_t *d_old2new, int32_t *d_partners);

} // namespace sporder::hm
