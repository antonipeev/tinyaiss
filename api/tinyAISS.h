#pragma once

#include <cstdint>
#include "../core/types.h"

namespace tinyaiss {

struct SearchResult {
    float* distances;     // [nq × k], caller owns
    int64_t* ids;         // [nq × k], caller owns
};

struct IVFConfig {
    uint32_t nlist  = 256;
    uint32_t nprobe = 8;
};

// opaque handle
struct Index;

// --- lifecycle ---
// create new index object in memory and return pointer to it
Index* create_index(IndexType type, uint32_t dim, MetricType metric,
                    IVFConfig ivf_config = {});
void   destroy_index(Index* index);

// --- core operations ---
// train: required for ivf before add(). no-op for flat.
// data: row-major float array, n vectors of dimension dim.
void train(Index* index, const float* data, uint64_t n);

// add: insert vectors. ids may be nullptr for sequential assignment.
void add(Index* index, const float* data, const int64_t* ids, uint64_t n);

// search: find k nearest neighbors for nq query vectors.
// allocates result arrays internally; caller must free via free_result().
SearchResult search(const Index* index, const float* queries,
                    uint64_t nq, uint32_t k);

void free_result(SearchResult result);

// --- persistence ---
void save(const Index* index, const char* path);
Index* load(const char* path);

// --- accessors ---
uint64_t    ntotal(const Index* index);
uint32_t    dimension(const Index* index);
MetricType  metric(const Index* index);
IndexType   index_type(const Index* index);

// --- ivf-specific ---
void     set_nprobe(Index* index, uint32_t nprobe);
uint32_t get_nprobe(const Index* index);

} // namespace tinyaiss
