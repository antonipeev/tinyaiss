#include "ivf_lists.h"
#include "../core/memory.h"
#include "../core/aligned_matrix.h"
#include <cstring>
#include <limits>
#include <new>

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

bool InvertedLists::allocate(uint32_t nlist_, uint32_t dim_, uint32_t total_vectors_) {
    free();

    nlist = nlist_;
    dim = dim_;
    dim_stride = padded_dimension(dim_);
    total_vectors = total_vectors_;

    // overflow-safe size computation for vector storage
    size_t vector_bytes = 0;
    if (total_vectors > 0 && dim_stride > 0) {
        if (total_vectors > std::numeric_limits<size_t>::max() / dim_stride) {
            free();
            return false;
        }
        size_t total_elements = static_cast<size_t>(total_vectors) * dim_stride;
        if (total_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
            free();
            return false;
        }
        vector_bytes = total_elements * sizeof(float);
    }

    // allocate contiguous vector storage
    if (vector_bytes > 0) {
        vectors = static_cast<float*>(aligned_alloc_wrapper(64, vector_bytes));
        if (!vectors) {
            free();
            return false;
        }
        std::memset(vectors, 0, vector_bytes);
    } else {
        vectors = nullptr;
    }

    // allocate id storage
    if (total_vectors > 0) {
        ids = new(std::nothrow) int64_t[total_vectors];
        if (!ids) {
            free();
            return false;
        }
    } else {
        ids = nullptr;
    }

    // allocate list metadata
    list_offsets = new(std::nothrow) uint32_t[nlist + 1];  // +1 for terminal offset
    list_sizes = new(std::nothrow) uint32_t[nlist];
    if (!list_offsets || !list_sizes) {
        free();
        return false;
    }

    std::memset(list_offsets, 0, (nlist + 1) * sizeof(uint32_t));
    std::memset(list_sizes, 0, nlist * sizeof(uint32_t));
    return true;
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
