#include "index_flat.h"
#include "../search/topk.h"
#include <cstring>
#include <cmath>
#include <utility>
#include <limits>

namespace tinyaiss {

// select distance kernel based on metric
static DistanceKernel select_kernel(MetricType metric) {
    switch (metric) {
        case MetricType::L2:
            return { distance_l2, distance_l2_batch };
        case MetricType::COSINE:
            return { distance_ip, distance_ip_batch };
        default:
            return { distance_l2, distance_l2_batch };
    }
}

// normalize vector to unit length (for cosine similarity)
static void normalize_vector(float* vec, uint32_t dim) {
    float norm = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-10f) {
        for (uint32_t i = 0; i < dim; i++) {
            vec[i] /= norm;
        }
    }
}

IndexFlat::IndexFlat(uint32_t dim_, MetricType metric_)
    : ids(nullptr), dim(dim_), ntotal(0), metric(metric_),
      kernel(select_kernel(metric_)) {
}

IndexFlat::~IndexFlat() {
    delete[] ids;
}

void IndexFlat::add(const float* data, const int64_t* ids_in, uint64_t n) {
    uint32_t old_ntotal = ntotal;

    // validate that we won't overflow uint32_t
    if (n > UINT32_MAX || old_ntotal > UINT32_MAX - n) {
        // overflow would occur
        return;
    }

    uint32_t new_ntotal = old_ntotal + static_cast<uint32_t>(n);

    // reallocate vectors matrix
    AlignedMatrix new_vectors;
    new_vectors.allocate(new_ntotal, dim);

    // copy old vectors
    if (old_ntotal > 0) {
        std::memcpy(new_vectors.data, vectors.data, vectors.bytes());
    }

    // copy new vectors
    for (uint64_t i = 0; i < n; i++) {
        const float* src = data + i * dim;
        float* dst = new_vectors.row(old_ntotal + static_cast<uint32_t>(i));

        std::memcpy(dst, src, dim * sizeof(float));

        // normalize if cosine metric
        if (metric == MetricType::COSINE) {
            normalize_vector(dst, dim);
        }
    }

    vectors = std::move(new_vectors);

    // handle ids
    if (ids_in) {
        int64_t* new_ids = new int64_t[new_ntotal];
        if (old_ntotal > 0 && ids) {
            std::memcpy(new_ids, ids, old_ntotal * sizeof(int64_t));
        }
        std::memcpy(new_ids + old_ntotal, ids_in, n * sizeof(int64_t));
        delete[] ids;
        ids = new_ids;
    } else if (ids) {
        // was using external ids, now sequential - reallocate
        int64_t* new_ids = new int64_t[new_ntotal];
        std::memcpy(new_ids, ids, old_ntotal * sizeof(int64_t));
        for (uint64_t i = 0; i < n; i++) {
            new_ids[old_ntotal + i] = static_cast<int64_t>(old_ntotal + i);
        }
        delete[] ids;
        ids = new_ids;
    }

    ntotal = new_ntotal;
}

void IndexFlat::search(const float* queries, uint64_t nq, uint32_t k,
                       float* out_distances, int64_t* out_ids) const {
    if (ntotal == 0) {
        // initialize output with sentinel values
        for (uint64_t i = 0; i < nq * k; i++) {
            out_distances[i] = std::numeric_limits<float>::max();
            out_ids[i] = -1;
        }
        return;
    }

    // allocate normalized queries if cosine metric
    const float* query_data = queries;
    float* normalized_queries = nullptr;

    if (metric == MetricType::COSINE) {
        normalized_queries = new float[nq * dim];
        for (uint64_t i = 0; i < nq; i++) {
            std::memcpy(normalized_queries + i * dim, queries + i * dim, dim * sizeof(float));
            normalize_vector(normalized_queries + i * dim, dim);
        }
        query_data = normalized_queries;
    }

    TopKHeap heap(k);

    for (uint64_t i = 0; i < nq; i++) {
        const float* query = query_data + i * dim;
        heap.reset();

        for (uint32_t j = 0; j < ntotal; j++) {
            float d = kernel.single(query, vectors.row(j), dim);

            // for cosine similarity, negate distance (we want max, heap is min)
            if (metric == MetricType::COSINE) {
                d = -d;
            }

            heap.push_if_smaller(d, get_id(j));
        }

        heap.dump_sorted(out_distances + i * k, out_ids + i * k);
    }

    delete[] normalized_queries;
}

} // namespace tinyaiss
