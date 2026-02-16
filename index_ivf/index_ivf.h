#pragma once

#include "../core/aligned_matrix.h"
#include "../core/types.h"
#include "../math/distance.h"
#include "ivf_lists.h"
#include <cstdint>

namespace tinyaiss {

// ivf index: approximate nearest neighbor search using inverted files
struct IndexIVF {
    AlignedMatrix centroids;   // (nlist, dim_stride) centroid vectors
    InvertedLists lists;       // packed inverted list storage
    uint32_t dim;              // original dimensionality
    uint32_t nlist;            // number of centroids/lists
    uint32_t nprobe;           // number of lists to probe at search time
    uint32_t ntotal;           // total number of vectors stored
    MetricType metric;         // l2 or cosine
    bool is_trained;           // whether centroids have been computed
    DistanceKernel kernel;     // function pointers for distance computation

    IndexIVF(uint32_t dim, uint32_t nlist, MetricType metric);
    ~IndexIVF() = default;

    // non-copyable
    IndexIVF(const IndexIVF&) = delete;
    IndexIVF& operator=(const IndexIVF&) = delete;

    // train: compute centroids via k-means
    // data: row-major float array, n vectors of dimension dim
    void train(const float* data, uint64_t n);

    // add vectors to the index (requires is_trained == true)
    // data: row-major float array, n vectors of dimension dim
    // ids_in: optional external ids (nullptr for sequential assignment)
    void add(const float* data, const int64_t* ids_in, uint64_t n);

    // search for k nearest neighbors
    // queries: row-major float array, nq vectors of dimension dim
    // out_distances: [nq × k] output buffer
    // out_ids: [nq × k] output buffer
    void search(const float* queries, uint64_t nq, uint32_t k,
                float* out_distances, int64_t* out_ids) const;

    // set number of lists to probe
    void set_nprobe(uint32_t nprobe_);
};

} // namespace tinyaiss
