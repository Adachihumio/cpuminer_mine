#ifdef __AVX2__
#include "yescrypt-avx2.c"
#elif defined __SSE2__
#include "yescrypt-simd.c"
#else
#include "yescrypt-opt.c"
#endif
