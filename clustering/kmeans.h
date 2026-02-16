#pragma once

#include "../core/aligned_matrix.h"
#include "../core/types.h"
#include <cstdint>

namespace tinyaiss {

struct KMeansConfig {
    uint32_t k;            // number of centroids
    uint32_t max_iter;     // default: 20
    float tolerance;       // convergence: stop if centroid movement < tolerance
    uint64_t seed;         // rng seed for deterministic initialization

    KMeansConfig()
        : k(0), max_iter(20), tolerance(1e-4f), seed(1234) {}
};

// train k-means clustering on data
// data: row-major float array, n vectors of dimension dim (with dim_stride)
// returns centroids as alignedmatrix (k rows, dim columns)
AlignedMatrix kmeans_train(const float* data, uint64_t n, uint32_t dim,
                           uint32_t dim_stride, const KMeansConfig& config,
                           MetricType metric);

} // namespace tinyaiss
