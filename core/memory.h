#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace tinyaiss {

// aligned allocation wrapper
inline void* aligned_alloc_wrapper(size_t alignment, size_t size) {
#if defined(_MSC_VER)
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

// aligned free wrapper
inline void aligned_free_wrapper(void* ptr) {
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

} // namespace tinyaiss
