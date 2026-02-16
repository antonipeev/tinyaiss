#pragma once

#include <cstdint>

namespace tinyaiss {

// cache-line aligned matrix for vector storage
// memory layout: rows × col_stride contiguous floats, 64-byte aligned
struct AlignedMatrix {
    float* data;        // 64-byte aligned allocation
    uint32_t rows;      // number of vectors
    uint32_t cols;      // dimensionality (original)
    uint32_t col_stride; // actual stride per row in floats (padded to multiple of 16)

    AlignedMatrix();
    ~AlignedMatrix();

    // non-copyable
    AlignedMatrix(const AlignedMatrix&) = delete;
    AlignedMatrix& operator=(const AlignedMatrix&) = delete;

    // moveable
    AlignedMatrix(AlignedMatrix&& other) noexcept;
    AlignedMatrix& operator=(AlignedMatrix&& other) noexcept;

    // allocate storage for rows × cols vectors with padding
    void allocate(uint32_t rows, uint32_t cols);

    // free storage
    void free();

    // get pointer to row i
    float* row(uint32_t i);
    const float* row(uint32_t i) const;

    // total bytes allocated
    size_t bytes() const;
};

// compute padded dimension (next multiple of 16)
inline uint32_t padded_dimension(uint32_t dim) {
    return ((dim + 15) / 16) * 16;
}

} // namespace tinyaiss
