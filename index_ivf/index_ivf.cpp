#include "index_ivf.h"
#include "../clustering/kmeans.h"
#include "../search/topk.h"
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <utility>

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

IndexIVF::IndexIVF(uint32_t dim_, uint32_t nlist_, MetricType metric_)
    : dim(dim_), nlist(nlist_), nprobe(1), ntotal(0),
      metric(metric_), is_trained(false), kernel(select_kernel(metric_)) {
}

void IndexIVF::train(const float* data, uint64_t n) {
    // prepare training data (normalize if cosine)
    float* train_data = nullptr;
    const float* train_ptr = data;

    if (metric == MetricType::COSINE) {
        train_data = new float[n * dim];
        for (uint64_t i = 0; i < n; i++) {
            std::memcpy(train_data + i * dim, data + i * dim, dim * sizeof(float));
            normalize_vector(train_data + i * dim, dim);
        }
        train_ptr = train_data;
    }

    // run k-means
    KMeansConfig config;
    config.k = nlist;
    config.max_iter = 20;
    config.tolerance = 1e-4f;
    config.seed = 1234;

    centroids = kmeans_train(train_ptr, n, dim, dim, config, metric);
    is_trained = true;

    delete[] train_data;
}

void IndexIVF::add(const float* data, const int64_t* ids_in, uint64_t n) {
    if (!is_trained) {
        return; // cannot add without training
    }

    // prepare data (normalize if cosine)
    float* add_data = nullptr;
    const float* add_ptr = data;

    if (metric == MetricType::COSINE) {
        add_data = new float[n * dim];
        for (uint64_t i = 0; i < n; i++) {
            std::memcpy(add_data + i * dim, data + i * dim, dim * sizeof(float));
            normalize_vector(add_data + i * dim, dim);
        }
        add_ptr = add_data;
    }

    // phase 1: assign each vector to nearest centroid (batch distance)
    std::vector<uint32_t> assignments(n);
    std::vector<float> centroid_dists(nlist);
    for (uint64_t i = 0; i < n; i++) {
        const float* vec = add_ptr + i * dim;

        // batch-compute distance from this vector to all centroids at once
        kernel.batch(vec, centroids.data, nlist, dim, centroids.col_stride,
                     centroid_dists.data());

        uint32_t best_list = 0;
        float best_dist = centroid_dists[0];
        for (uint32_t c = 1; c < nlist; c++) {
            if (centroid_dists[c] < best_dist) {
                best_dist = centroid_dists[c];
                best_list = c;
            }
        }
        assignments[i] = best_list;
    }

    // phase 2: count per-list sizes
    std::vector<uint32_t> counts(nlist, 0);
    for (uint64_t i = 0; i < n; i++) {
        counts[assignments[i]]++;
    }

    // phase 3: build packed inverted lists
    // validate that we won't overflow uint32_t
    if (n > UINT32_MAX || ntotal > UINT32_MAX - n) {
        delete[] add_data;
        return;
    }

    uint32_t total_vecs = ntotal + static_cast<uint32_t>(n);

    InvertedLists new_lists;
    new_lists.allocate(nlist, dim, total_vecs);

    // compute offsets
    uint32_t offset = 0;
    for (uint32_t c = 0; c < nlist; c++) {
        new_lists.list_offsets[c] = offset;
        uint32_t old_size = (ntotal > 0 && lists.list_sizes) ? lists.list_sizes[c] : 0;
        new_lists.list_sizes[c] = old_size + counts[c];
        offset += new_lists.list_sizes[c];
    }
    new_lists.list_offsets[nlist] = offset;

    // copy old data and insert new vectors
    std::vector<uint32_t> write_positions(nlist);
    for (uint32_t c = 0; c < nlist; c++) {
        uint32_t old_size = (ntotal > 0 && lists.list_sizes) ? lists.list_sizes[c] : 0;

        // copy old vectors for this list
        if (old_size > 0 && ntotal > 0) {
            const float* old_vecs = lists.list_vectors(c);
            const int64_t* old_ids = lists.list_ids(c);
            float* new_vecs = new_lists.list_vectors(c);
            int64_t* new_ids = new_lists.list_ids(c);

            std::memcpy(new_vecs, old_vecs,
                       old_size * lists.dim_stride * sizeof(float));
            std::memcpy(new_ids, old_ids, old_size * sizeof(int64_t));
        }

        write_positions[c] = old_size;
    }

    // insert new vectors
    for (uint64_t i = 0; i < n; i++) {
        uint32_t list_id = assignments[i];
        uint32_t pos = write_positions[list_id];

        float* dest_vec = new_lists.list_vectors(list_id) +
                         pos * new_lists.dim_stride;
        const float* src_vec = add_ptr + i * dim;

        std::memcpy(dest_vec, src_vec, dim * sizeof(float));

        int64_t* dest_id = new_lists.list_ids(list_id) + pos;
        *dest_id = ids_in ? ids_in[i] : static_cast<int64_t>(ntotal + i);

        write_positions[list_id]++;
    }

    lists = std::move(new_lists);
    ntotal = total_vecs;

    delete[] add_data;
}

void IndexIVF::search(const float* queries, uint64_t nq, uint32_t k,
                     float* out_distances, int64_t* out_ids) const {
    if (!is_trained || ntotal == 0) {
        // initialize output with sentinel values
        for (uint64_t i = 0; i < nq * k; i++) {
            out_distances[i] = std::numeric_limits<float>::max();
            out_ids[i] = -1;
        }
        return;
    }

    // prepare queries (normalize if cosine)
    float* query_data = nullptr;
    const float* query_ptr = queries;

    if (metric == MetricType::COSINE) {
        query_data = new float[nq * dim];
        for (uint64_t i = 0; i < nq; i++) {
            std::memcpy(query_data + i * dim, queries + i * dim, dim * sizeof(float));
            normalize_vector(query_data + i * dim, dim);
        }
        query_ptr = query_data;
    }

    TopKHeap heap(k);

    for (uint64_t i = 0; i < nq; i++) {
        const float* query = query_ptr + i * dim;

        // step 1: batch-compute distances to all centroids
        std::vector<float> centroid_dists(nlist);
        kernel.batch(query, centroids.data, nlist, dim, centroids.col_stride,
                     centroid_dists.data());

        // step 2: select nprobe nearest centroids
        std::vector<uint32_t> probe_indices(nlist);
        for (uint32_t c = 0; c < nlist; c++) {
            probe_indices[c] = c;
        }

        uint32_t actual_nprobe = std::min(nprobe, nlist);
        std::partial_sort(probe_indices.begin(),
                         probe_indices.begin() + actual_nprobe,
                         probe_indices.end(),
                         [&centroid_dists](uint32_t a, uint32_t b) {
                             return centroid_dists[a] < centroid_dists[b];
                         });

        // step 3: scan selected lists
        heap.reset();
        for (uint32_t p = 0; p < actual_nprobe; p++) {
            uint32_t list_id = probe_indices[p];
            uint32_t list_size = lists.list_sizes[list_id];

            if (list_size == 0) continue;

            const float* list_vecs = lists.list_vectors(list_id);
            const int64_t* list_ids_ptr = lists.list_ids(list_id);

            for (uint32_t j = 0; j < list_size; j++) {
                float d = kernel.single(query, list_vecs + j * lists.dim_stride, dim);

                // for cosine, negate (we want max, heap is min)
                if (metric == MetricType::COSINE) {
                    d = -d;
                }

                heap.push_if_smaller(d, list_ids_ptr[j]);
            }
        }

        // step 4: extract results
        heap.dump_sorted(out_distances + i * k, out_ids + i * k);
    }

    delete[] query_data;
}

void IndexIVF::set_nprobe(uint32_t nprobe_) {
    nprobe = nprobe_;
}

} // namespace tinyaiss
