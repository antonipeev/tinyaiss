#include "distance.h"
#include "platform.h"

namespace tinyaiss {

// scalar implementation
static float distance_ip_scalar(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

#if defined(TINYFAISS_USE_AVX2)

// horizontal sum helper (same as in distance_l2)
static inline float hsum_avx2(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}

static float distance_ip_avx2(const float* a, const float* b, uint32_t dim) {
    __m256 sum = _mm256_setzero_ps();
    uint32_t i = 0;

    // process 8 floats at a time
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);  // fma: sum += a*b
    }

    // horizontal reduction
    float result = hsum_avx2(sum);

    // scalar tail
    for (; i < dim; i++) {
        result += a[i] * b[i];
    }

    return result;
}

#elif defined(TINYFAISS_USE_NEON)

static float distance_ip_neon(const float* a, const float* b, uint32_t dim) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    uint32_t i = 0;

    // process 4 floats at a time
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);  // fma: sum += a*b
    }

    // horizontal reduction
    float result = vaddvq_f32(sum);

    // scalar tail
    for (; i < dim; i++) {
        result += a[i] * b[i];
    }

    return result;
}

#endif

// public function with simd dispatch
float distance_ip(const float* a, const float* b, uint32_t dim) {
#if defined(TINYFAISS_USE_AVX2)
    return distance_ip_avx2(a, b, dim);
#elif defined(TINYFAISS_USE_NEON)
    return distance_ip_neon(a, b, dim);
#else
    return distance_ip_scalar(a, b, dim);
#endif
}

// batch version
void distance_ip_batch(const float* query, const float* vectors,
                       uint32_t n, uint32_t dim, uint32_t dim_stride,
                       float* out_distances) {
    for (uint32_t i = 0; i < n; i++) {
        out_distances[i] = distance_ip(query, vectors + i * dim_stride, dim);
    }
}

} // namespace tinyaiss
