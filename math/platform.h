#pragma once

// simd platform detection
#if defined(__AVX2__) && defined(__FMA__)
    #define TINYFAISS_USE_AVX2 1
    #include <immintrin.h>
#elif defined(__ARM_NEON)
    #define TINYFAISS_USE_NEON 1
    #include <arm_neon.h>
#else
    #define TINYFAISS_USE_SCALAR 1
#endif
