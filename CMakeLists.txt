cmake_minimum_required(VERSION 3.16)
project(tinyAISS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SIMD flags (auto-detect or user-specified)
include(CheckCXXCompilerFlag)

if(MSVC)
    check_cxx_compiler_flag("/arch:AVX2" HAS_AVX2)
    if(HAS_AVX2)
        add_compile_options(/arch:AVX2)
    endif()
else()
    check_cxx_compiler_flag("-mavx2 -mfma" HAS_AVX2)
    if(HAS_AVX2)
        add_compile_options(-mavx2 -mfma)
    endif()
endif()

# Library sources
add_library(tinyaiss STATIC
    core/aligned_matrix.cpp
    math/distance_l2.cpp
    math/distance_ip.cpp
    clustering/kmeans.cpp
    index_flat/index_flat.cpp
    index_ivf/ivf_lists.cpp
    index_ivf/index_ivf.cpp
    search/topk.cpp
    io/serialization.cpp
    api/tinyAISS.cpp
)

target_include_directories(tinyaiss PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Optional: enable warnings
if(MSVC)
    target_compile_options(tinyaiss PRIVATE /W4)
else()
    target_compile_options(tinyaiss PRIVATE -Wall -Wextra -pedantic)
endif()

# Tests
enable_testing()

# Example executable
add_executable(example_usage
    tests/example_usage.cpp
)
target_link_libraries(example_usage tinyaiss)
