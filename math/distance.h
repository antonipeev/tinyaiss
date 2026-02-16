#pragma once

#include <cstdint>

namespace tinyaiss {

// function pointer types for distance computation
using DistanceFunc = float(*)(const float*, const float*, uint32_t);
using DistanceBatchFunc = void(*)(const float*, const float*, uint32_t,
                                   uint32_t, uint32_t, float*);

// distance kernel bundle
struct DistanceKernel {
    DistanceFunc single;
    DistanceBatchFunc batch;
};

// compute squared l2 distance between two vectors
float distance_l2(const float* a, const float* b, uint32_t dim);

// compute inner product of two vectors
float distance_ip(const float* a, const float* b, uint32_t dim);

// batch: compute distances from one query to n vectors
// out_distances must hold n floats
void distance_l2_batch(const float* query, const float* vectors,
                       uint32_t n, uint32_t dim, uint32_t dim_stride,
                       float* out_distances);

void distance_ip_batch(const float* query, const float* vectors,
                       uint32_t n, uint32_t dim, uint32_t dim_stride,
                       float* out_distances);

} // namespace tinyaiss
