#include "kmeans.h"
#include "../math/distance.h"
#include <random>
#include <vector>
#include <algorithm>
#include <cstring>
#include <limits>
#include <cmath>

namespace tinyaiss {

// k-means++ initialization — uses batch distance for speed
static void kmeans_init(AlignedMatrix& centroids, const float* data, uint64_t n,
                       uint32_t dim, uint32_t dim_stride, uint32_t k,
                       uint64_t seed, MetricType metric,
                       DistanceBatchFunc batch_fn) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> uniform(0, n - 1);

    // select first centroid randomly
    uint64_t first_idx = uniform(rng);
    std::memcpy(centroids.row(0), data + first_idx * dim_stride, dim * sizeof(float));

    // track minimum distances
    std::vector<float> min_dists(n, std::numeric_limits<float>::max());
    std::vector<float> cur_dists(n);

    // select remaining centroids
    for (uint32_t c = 1; c < k; c++) {
        // batch-compute distances from all data points to the previous centroid
        const float* prev_centroid = centroids.row(c - 1);
        batch_fn(prev_centroid, data, static_cast<uint32_t>(n), dim, dim_stride,
                 cur_dists.data());

        for (uint64_t i = 0; i < n; i++) {
            if (cur_dists[i] < min_dists[i]) {
                min_dists[i] = cur_dists[i];
            }
        }

        // sample proportional to distance
        double total = 0.0;
        for (uint64_t i = 0; i < n; i++) {
            total += min_dists[i];
        }

        std::uniform_real_distribution<double> uniform_real(0.0, total);
        double threshold = uniform_real(rng);
        double cumulative = 0.0;

        for (uint64_t i = 0; i < n; i++) {
            cumulative += min_dists[i];
            if (cumulative >= threshold) {
                std::memcpy(centroids.row(c), data + i * dim_stride, dim * sizeof(float));
                break;
            }
        }
    }
}

AlignedMatrix kmeans_train(const float* data, uint64_t n, uint32_t dim,
                           uint32_t dim_stride, const KMeansConfig& config,
                           MetricType metric) {
    const uint32_t k = config.k;

    // select distance functions based on metric
    DistanceFunc distance_fn = (metric == MetricType::COSINE) ? distance_ip : distance_l2;
    DistanceBatchFunc batch_fn = (metric == MetricType::COSINE) ? distance_ip_batch
                                                                : distance_l2_batch;

    // allocate centroids
    AlignedMatrix centroids;
    centroids.allocate(k, dim);

    // initialize with k-means++
    kmeans_init(centroids, data, n, dim, dim_stride, k, config.seed, metric,
                batch_fn);

    // allocate working buffers
    std::vector<uint32_t> assignments(n);
    std::vector<uint32_t> counts(k);
    std::vector<float> dists(k);  // reusable per-vector distance buffer
    AlignedMatrix new_centroids;
    new_centroids.allocate(k, dim);

    // iteration loop
    for (uint32_t iter = 0; iter < config.max_iter; iter++) {
        // assignment step: find nearest centroid for each vector
        std::fill(counts.begin(), counts.end(), 0);
        for (uint32_t c = 0; c < k; c++) {
            std::memset(new_centroids.row(c), 0, centroids.col_stride * sizeof(float));
        }

        for (uint64_t i = 0; i < n; i++) {
            const float* vec = data + i * dim_stride;

            // batch-compute distance from this vector to all k centroids at once
            batch_fn(vec, centroids.data, k, dim, centroids.col_stride,
                     dists.data());

            // find nearest centroid from batch results
            uint32_t best_c = 0;
            float best_dist = dists[0];
            for (uint32_t c = 1; c < k; c++) {
                if (dists[c] < best_dist) {
                    best_dist = dists[c];
                    best_c = c;
                }
            }

            assignments[i] = best_c;
            counts[best_c]++;

            // accumulate into new_centroids
            float* centroid_accum = new_centroids.row(best_c);
            for (uint32_t d = 0; d < dim; d++) {
                centroid_accum[d] += vec[d];
            }
        }

        // update step: compute means
        float max_shift = 0.0f;
        std::mt19937_64 rng(config.seed + iter);
        std::uniform_int_distribution<uint64_t> uniform(0, n - 1);

        for (uint32_t c = 0; c < k; c++) {
            if (counts[c] == 0) {
                // dead centroid: reinitialize to random data point
                uint64_t random_idx = uniform(rng);
                std::memcpy(new_centroids.row(c), data + random_idx * dim_stride,
                           dim * sizeof(float));
            } else {
                // divide by count to get mean
                float* new_cent = new_centroids.row(c);
                for (uint32_t d = 0; d < dim; d++) {
                    new_cent[d] /= counts[c];
                }
            }

            // compute shift
            float shift = distance_fn(centroids.row(c), new_centroids.row(c), dim);
            if (shift > max_shift) {
                max_shift = shift;
            }
        }

        // swap centroids
        std::swap(centroids, new_centroids);

        // check convergence
        if (max_shift < config.tolerance) {
            break;
        }
    }

    return centroids;
}

} // namespace tinyaiss
