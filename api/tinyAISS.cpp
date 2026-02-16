#include "tinyAISS.h"
#include "../index_flat/index_flat.h"
#include "../index_ivf/index_ivf.h"
#include "../io/serialization.h"
#include <cstring>
#include <fstream>
#include <limits>

namespace tinyaiss {

// internal index wrapper
struct Index {
    IndexType type;
    void* impl;  // points to indexflat* or indexivf*

    Index(IndexType t, void* i) : type(t), impl(i) {}
};

Index* create_index(IndexType type, uint32_t dim, MetricType metric,
                    IVFConfig ivf_config) {
    void* impl = nullptr;

    switch (type) {
        case IndexType::FLAT:
            impl = new IndexFlat(dim, metric);
            break;
        case IndexType::IVF:
            impl = new IndexIVF(dim, ivf_config.nlist, metric);
            static_cast<IndexIVF*>(impl)->set_nprobe(ivf_config.nprobe);
            break;
    }

    return new Index(type, impl);
}

void destroy_index(Index* index) {
    if (!index) return;

    switch (index->type) {
        case IndexType::FLAT:
            delete static_cast<IndexFlat*>(index->impl);
            break;
        case IndexType::IVF:
            delete static_cast<IndexIVF*>(index->impl);
            break;
    }

    delete index;
}

void train(Index* index, const float* data, uint64_t n) {
    if (!index) return;

    switch (index->type) {
        case IndexType::FLAT:
            // no training needed for flat index
            break;
        case IndexType::IVF:
            static_cast<IndexIVF*>(index->impl)->train(data, n);
            break;
    }
}

void add(Index* index, const float* data, const int64_t* ids, uint64_t n) {
    if (!index) return;

    switch (index->type) {
        case IndexType::FLAT:
            static_cast<IndexFlat*>(index->impl)->add(data, ids, n);
            break;
        case IndexType::IVF:
            static_cast<IndexIVF*>(index->impl)->add(data, ids, n);
            break;
    }
}

SearchResult search(const Index* index, const float* queries,
                    uint64_t nq, uint32_t k) {
    SearchResult result;
    result.distances = new float[nq * k];
    result.ids = new int64_t[nq * k];

    // initialize output arrays with sentinel values
    for (uint64_t i = 0; i < nq * k; i++) {
        result.distances[i] = std::numeric_limits<float>::max();
        result.ids[i] = -1;
    }

    if (!index) {
        return result;
    }

    switch (index->type) {
        case IndexType::FLAT:
            static_cast<const IndexFlat*>(index->impl)->search(
                queries, nq, k, result.distances, result.ids);
            break;
        case IndexType::IVF:
            static_cast<const IndexIVF*>(index->impl)->search(
                queries, nq, k, result.distances, result.ids);
            break;
    }

    return result;
}

void free_result(SearchResult result) {
    delete[] result.distances;
    delete[] result.ids;
}

void save(const Index* index, const char* path) {
    if (!index) return;

    switch (index->type) {
        case IndexType::FLAT:
            save_index_flat(*static_cast<const IndexFlat*>(index->impl), path);
            break;
        case IndexType::IVF:
            save_index_ivf(*static_cast<const IndexIVF*>(index->impl), path);
            break;
    }
}

Index* load(const char* path) {
    // read header to determine index type
    std::ifstream file(path, std::ios::binary);
    if (!file) return nullptr;

    uint32_t magic, version, index_type;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    file.read(reinterpret_cast<char*>(&index_type), sizeof(index_type));
    file.close();

    if (magic != MAGIC_NUMBER || version != FORMAT_VERSION) {
        return nullptr;
    }

    void* impl = nullptr;
    IndexType type;

    switch (index_type) {
        case 0: // flat
            impl = load_index_flat(path);
            type = IndexType::FLAT;
            break;
        case 1: // ivf
            impl = load_index_ivf(path);
            type = IndexType::IVF;
            break;
        default:
            return nullptr;
    }

    if (!impl) return nullptr;
    return new Index(type, impl);
}

uint64_t ntotal(const Index* index) {
    if (!index) return 0;

    switch (index->type) {
        case IndexType::FLAT:
            return static_cast<const IndexFlat*>(index->impl)->ntotal;
        case IndexType::IVF:
            return static_cast<const IndexIVF*>(index->impl)->ntotal;
        default:
            return 0;
    }
}

uint32_t dimension(const Index* index) {
    if (!index) return 0;

    switch (index->type) {
        case IndexType::FLAT:
            return static_cast<const IndexFlat*>(index->impl)->dim;
        case IndexType::IVF:
            return static_cast<const IndexIVF*>(index->impl)->dim;
        default:
            return 0;
    }
}

MetricType metric(const Index* index) {
    if (!index) return MetricType::L2;

    switch (index->type) {
        case IndexType::FLAT:
            return static_cast<const IndexFlat*>(index->impl)->metric;
        case IndexType::IVF:
            return static_cast<const IndexIVF*>(index->impl)->metric;
        default:
            return MetricType::L2;
    }
}

IndexType index_type(const Index* index) {
    return index ? index->type : IndexType::FLAT;
}

void set_nprobe(Index* index, uint32_t nprobe) {
    if (!index || index->type != IndexType::IVF) return;
    static_cast<IndexIVF*>(index->impl)->set_nprobe(nprobe);
}

uint32_t get_nprobe(const Index* index) {
    if (!index || index->type != IndexType::IVF) return 0;
    return static_cast<const IndexIVF*>(index->impl)->nprobe;
}

} // namespace tinyaiss
