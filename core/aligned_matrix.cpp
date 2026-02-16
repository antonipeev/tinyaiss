#include "aligned_matrix.h"
#include "memory.h"
#include <cstring>

namespace tinyaiss {

AlignedMatrix::AlignedMatrix()
    : data(nullptr), rows(0), cols(0), col_stride(0) {}

AlignedMatrix::~AlignedMatrix() {
    free();
}

AlignedMatrix::AlignedMatrix(AlignedMatrix&& other) noexcept
    : data(other.data), rows(other.rows), cols(other.cols), col_stride(other.col_stride) {
    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
    other.col_stride = 0;
}

AlignedMatrix& AlignedMatrix::operator=(AlignedMatrix&& other) noexcept {
    if (this != &other) {
        free();
        data = other.data;
        rows = other.rows;
        cols = other.cols;
        col_stride = other.col_stride;
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
        other.col_stride = 0;
    }
    return *this;
}

void AlignedMatrix::allocate(uint32_t rows_, uint32_t cols_) {
    free();
    rows = rows_;
    cols = cols_;
    col_stride = padded_dimension(cols_);

    size_t total_bytes = static_cast<size_t>(rows) * col_stride * sizeof(float);
    if (total_bytes > 0) {
        data = static_cast<float*>(aligned_alloc_wrapper(64, total_bytes));
        // zero-initialize
        if (data) {
            std::memset(data, 0, total_bytes);
        }
    } else {
        data = nullptr;
    }
}

void AlignedMatrix::free() {
    if (data) {
        aligned_free_wrapper(data);
        data = nullptr;
    }
    rows = 0;
    cols = 0;
    col_stride = 0;
}

float* AlignedMatrix::row(uint32_t i) {
    return data + static_cast<size_t>(i) * col_stride;
}

const float* AlignedMatrix::row(uint32_t i) const {
    return data + static_cast<size_t>(i) * col_stride;
}

size_t AlignedMatrix::bytes() const {
    return static_cast<size_t>(rows) * col_stride * sizeof(float);
}

} // namespace tinyaiss
