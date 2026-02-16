#include "ivf_lists.h"
#include "../core/memory.h"
#include "../core/aligned_matrix.h"
#include <cstring>

namespace tinyaiss {

InvertedLists::InvertedLists()
    : vectors(nullptr), ids(nullptr), list_offsets(nullptr),
      list_sizes(nullptr), nlist(0), dim(0), dim_stride(0), total_vectors(0) {}

InvertedLists::~InvertedLists() {
    free();
}

InvertedLists::InvertedLists(InvertedLists&& other) noexcept
    : vectors(other.vectors), ids(other.ids),
      list_offsets(other.list_offsets), list_sizes(other.list_sizes),
      nlist(other.nlist), dim(other.dim), dim_stride(other.dim_stride),
      total_vectors(other.total_vectors) {
    other.vectors = nullptr;
    other.ids = nullptr;
    other.list_offsets = nullptr;
    other.list_sizes = nullptr;
    other.nlist = 0;
    other.dim = 0;
    other.dim_stride = 0;
    other.total_vectors = 0;
}

InvertedLists& InvertedLists::operator=(InvertedLists&& other) noexcept {
    if (this != &other) {
        free();
        vectors = other.vectors;
        ids = other.ids;
        list_offsets = other.list_offsets;
        list_sizes = other.list_sizes;
        nlist = other.nlist;
        dim = other.dim;
        dim_stride = other.dim_stride;
        total_vectors = other.total_vectors;
        other.vectors = nullptr;
        other.ids = nullptr;
        other.list_offsets = nullptr;
        other.list_sizes = nullptr;
        other.nlist = 0;
        other.dim = 0;
        other.dim_stride = 0;
        other.total_vectors = 0;
    }
    return *this;
}

void InvertedLists::allocate(uint32_t nlist_, uint32_t dim_, uint32_t total_vectors_) {
    free();

    nlist = nlist_;
    dim = dim_;
    dim_stride = padded_dimension(dim_);
    total_vectors = total_vectors_;

    // allocate contiguous vector storage
    size_t vector_bytes = static_cast<size_t>(total_vectors) * dim_stride * sizeof(float);
    vectors = static_cast<float*>(aligned_alloc_wrapper(64, vector_bytes));
    std::memset(vectors, 0, vector_bytes);

    // allocate id storage
    ids = new int64_t[total_vectors];

    // allocate list metadata
    list_offsets = new uint32_t[nlist + 1];  // +1 for terminal offset
    list_sizes = new uint32_t[nlist];

    std::memset(list_offsets, 0, (nlist + 1) * sizeof(uint32_t));
    std::memset(list_sizes, 0, nlist * sizeof(uint32_t));
}

void InvertedLists::free() {
    if (vectors) {
        aligned_free_wrapper(vectors);
        vectors = nullptr;
    }
    delete[] ids;
    ids = nullptr;
    delete[] list_offsets;
    list_offsets = nullptr;
    delete[] list_sizes;
    list_sizes = nullptr;

    nlist = 0;
    dim = 0;
    dim_stride = 0;
    total_vectors = 0;
}

float* InvertedLists::list_vectors(uint32_t list_id) {
    return vectors + static_cast<size_t>(list_offsets[list_id]) * dim_stride;
}

const float* InvertedLists::list_vectors(uint32_t list_id) const {
    return vectors + static_cast<size_t>(list_offsets[list_id]) * dim_stride;
}

int64_t* InvertedLists::list_ids(uint32_t list_id) {
    return ids + list_offsets[list_id];
}

const int64_t* InvertedLists::list_ids(uint32_t list_id) const {
    return ids + list_offsets[list_id];
}

} // namespace tinyaiss
