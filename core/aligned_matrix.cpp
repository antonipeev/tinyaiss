#include "aligned_matrix.h"
#include "memory.h"
#include <cstring>
#include <limits>

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

bool AlignedMatrix::allocate(uint32_t rows_, uint32_t cols_) {
    free();
    rows = rows_;
    cols = cols_;
    col_stride = padded_dimension(cols_);

    // overflow-safe size computation: rows * col_stride * sizeof(float)
    size_t total_bytes = 0;
    if (rows > 0 && col_stride > 0) {
        // check: rows * col_stride overflows?
        if (rows > std::numeric_limits<size_t>::max() / col_stride) {
            free();
            return false;
        }
        size_t total_elements = static_cast<size_t>(rows) * col_stride;
        // check: total_elements * sizeof(float) overflows?
        if (total_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
            free();
            return false;
        }
        total_bytes = total_elements * sizeof(float);
    }

    if (total_bytes > 0) {
        data = static_cast<float*>(aligned_alloc_wrapper(64, total_bytes));
        if (!data) {
            free();
            return false;
        }
        std::memset(data, 0, total_bytes);
    } else {
        data = nullptr;
    }
    return true;
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
