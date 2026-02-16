#pragma once

#include "../core/aligned_matrix.h"
#include "../core/types.h"
#include "../math/distance.h"
#include <cstdint>

namespace tinyaiss {

// flat index: brute-force exact search
struct IndexFlat {
    AlignedMatrix vectors;    // (ntotal, dim_stride) contiguous storage
    int64_t* ids;             // [ntotal] external ids, or nullptr for sequential 0..n-1
    uint32_t dim;             // original dimensionality
    uint32_t ntotal;          // number of vectors stored
    MetricType metric;        // l2 or cosine
    DistanceKernel kernel;    // function pointers for distance computation

    IndexFlat(uint32_t dim, MetricType metric);
    ~IndexFlat();

    // non-copyable
    IndexFlat(const IndexFlat&) = delete;
    IndexFlat& operator=(const IndexFlat&) = delete;

    // add vectors to the index
    // data: row-major float array, n vectors of dimension dim
    // ids_in: optional external ids (nullptr for sequential assignment)
    void add(const float* data, const int64_t* ids_in, uint64_t n);

    // search for k nearest neighbors
    // queries: row-major float array, nq vectors of dimension dim
    // out_distances: [nq × k] output buffer
    // out_ids: [nq × k] output buffer
    void search(const float* queries, uint64_t nq, uint32_t k,
                float* out_distances, int64_t* out_ids) const;

    // get id for internal index
    int64_t get_id(uint32_t internal_idx) const {
        return ids ? ids[internal_idx] : static_cast<int64_t>(internal_idx);
    }
};

} // namespace tinyaiss
