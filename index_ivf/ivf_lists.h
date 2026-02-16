#pragma once

#include <cstdint>

namespace tinyaiss {

// inverted list storage (structure-of-arrays)
struct InvertedLists {
    float* vectors;         // single contiguous allocation for all list vectors
    int64_t* ids;           // parallel array of ids
    uint32_t* list_offsets; // list_offsets[i] = start index of list i in vectors/ids
    uint32_t* list_sizes;   // list_sizes[i]  = number of vectors in list i
    uint32_t nlist;         // total number of lists
    uint32_t dim;           // vector dimensionality (original)
    uint32_t dim_stride;    // padded dimensionality (for alignment)
    uint32_t total_vectors; // sum of all list_sizes

    InvertedLists();
    ~InvertedLists();

    // non-copyable
    InvertedLists(const InvertedLists&) = delete;
    InvertedLists& operator=(const InvertedLists&) = delete;

    // moveable
    InvertedLists(InvertedLists&& other) noexcept;
    InvertedLists& operator=(InvertedLists&& other) noexcept;

    // allocate storage for packed lists
    // returns false if allocation fails (overflow or out of memory)
    bool allocate(uint32_t nlist, uint32_t dim, uint32_t total_vectors);

    // free storage
    void free();

    // get pointer to list i's vectors
    float* list_vectors(uint32_t list_id);
    const float* list_vectors(uint32_t list_id) const;

    // get pointer to list i's ids
    int64_t* list_ids(uint32_t list_id);
    const int64_t* list_ids(uint32_t list_id) const;
};

} // namespace tinyaiss
