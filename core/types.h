#pragma once

#include <cstdint>

namespace tinyaiss {

// metric types supported
enum class MetricType : uint32_t {
    L2 = 0,
    COSINE = 1,
};

// index types
enum class IndexType : uint32_t {
    FLAT = 0,
    IVF = 1,
};

} // namespace tinyaiss
