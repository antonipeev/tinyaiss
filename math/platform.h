#pragma once

// simd platform detection
// MSVC with /arch:AVX2 defines __AVX2__ but NOT __FMA__,
// yet supports FMA intrinsics. GCC/Clang define both with -mavx2 -mfma.
#if defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
    #define TINYFAISS_USE_AVX2 1
    #include <immintrin.h>
#elif defined(__ARM_NEON)
    #define TINYFAISS_USE_NEON 1
    #include <arm_neon.h>
#else
    #define TINYFAISS_USE_SCALAR 1
#endif
