/*-
 * Copyright 2009 Colin Percival
 * Copyright 2012-2014 Alexander Peslyak
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * This file was originally written by Colin Percival as part of the Tarsnap
 * online backup system.
 */

/*
 * On 64-bit, enabling SSE4.1 helps our pwxform code indirectly, via avoiding
 * gcc bug 54349 (fixed for gcc 4.9+).  On 32-bit, it's of direct help.  AVX
 * and XOP are of further help either way.
 */
#include <x86intrin.h>

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "sha256.h"
#include "sysendian.h"

#include "yescrypt.h"

#include "yescrypt-platform.c"

#if __STDC_VERSION__ >= 199901L
/* have restrict */
#elif defined(__GNUC__)
#define restrict __restrict
#else
#define restrict
#endif

#define PREFETCH(x, hint) _mm_prefetch((const char *)(x), (hint));
#define PREFETCH_OUT(x, hint) /* disabled */

#define STORE2(in, out0, out1) \
{ \
  __m256i Z = (in); \
  (out0) = _mm256_castsi256_si128(Z); \
  (out1) = _mm256_extracti128_si256(Z, 1); \
}

#define _mm256_setr_m128i(in0, in1) \
  _mm256_inserti128_si256(_mm256_castsi128_si256(in0), in1, 1)

#define ARX(out, in1, in2, s) \
	{ \
		__m128i T = _mm_add_epi32(in1, in2); \
		out = _mm_xor_si128(out, _mm_slli_epi32(T, s)); \
		out = _mm_xor_si128(out, _mm_srli_epi32(T, 32-s)); \
	}

#define ARX_VEC2(out, in1, in2, s) \
	{ \
		__m256i T = _mm256_add_epi32(in1, in2); \
		out = _mm256_xor_si256(out, _mm256_slli_epi32(T, s)); \
		out = _mm256_xor_si256(out, _mm256_srli_epi32(T, 32-s)); \
	}

#define SALSA20_2ROUNDS \
	/* Operate on "columns" */ \
	ARX(X1, X0, X3, 7) \
	ARX(X2, X1, X0, 9) \
	ARX(X3, X2, X1, 13) \
	ARX(X0, X3, X2, 18) \
\
	/* Rearrange data */ \
	X1 = _mm_shuffle_epi32(X1, 0x93); \
	X2 = _mm_shuffle_epi32(X2, 0x4E); \
	X3 = _mm_shuffle_epi32(X3, 0x39); \
\
	/* Operate on "rows" */ \
	ARX(X3, X0, X1, 7) \
	ARX(X2, X3, X0, 9) \
	ARX(X1, X2, X3, 13) \
	ARX(X0, X1, X2, 18) \
\
	/* Rearrange data */ \
	X1 = _mm_shuffle_epi32(X1, 0x39); \
	X2 = _mm_shuffle_epi32(X2, 0x4E); \
	X3 = _mm_shuffle_epi32(X3, 0x93);

#define SALSA20_2ROUNDS_VEC2 \
	/* Operate on "columns" */ \
	ARX_VEC2(X1, X0, X3, 7) \
	ARX_VEC2(X2, X1, X0, 9) \
	ARX_VEC2(X3, X2, X1, 13) \
	ARX_VEC2(X0, X3, X2, 18) \
\
	/* Rearrange data */ \
	X1 = _mm256_shuffle_epi32(X1, 0x93); \
	X2 = _mm256_shuffle_epi32(X2, 0x4E); \
	X3 = _mm256_shuffle_epi32(X3, 0x39); \
\
	/* Operate on "rows" */ \
	ARX_VEC2(X3, X0, X1, 7) \
	ARX_VEC2(X2, X3, X0, 9) \
	ARX_VEC2(X1, X2, X3, 13) \
	ARX_VEC2(X0, X1, X2, 18) \
\
	/* Rearrange data */ \
	X1 = _mm256_shuffle_epi32(X1, 0x39); \
	X2 = _mm256_shuffle_epi32(X2, 0x4E); \
	X3 = _mm256_shuffle_epi32(X3, 0x93);

/**
 * Apply the salsa20/8 core to the block provided in (X0 ... X3).
 */
#define SALSA20_8_BASE(maybe_decl, out) \
	{ \
		maybe_decl Y0 = X0; \
		maybe_decl Y1 = X1; \
		maybe_decl Y2 = X2; \
		maybe_decl Y3 = X3; \
		SALSA20_2ROUNDS \
		SALSA20_2ROUNDS \
		SALSA20_2ROUNDS \
		SALSA20_2ROUNDS \
		(out)[0] = X0 = _mm_add_epi32(X0, Y0); \
		(out)[1] = X1 = _mm_add_epi32(X1, Y1); \
		(out)[2] = X2 = _mm_add_epi32(X2, Y2); \
		(out)[3] = X3 = _mm_add_epi32(X3, Y3); \
	}

#define SALSA20_8_BASE_VEC2(maybe_decl, out0, out1) \
	{ \
		maybe_decl Y0 = X0; \
		maybe_decl Y1 = X1; \
		maybe_decl Y2 = X2; \
		maybe_decl Y3 = X3; \
		SALSA20_2ROUNDS_VEC2 \
		SALSA20_2ROUNDS_VEC2 \
		SALSA20_2ROUNDS_VEC2 \
		SALSA20_2ROUNDS_VEC2 \
		STORE2(X0 = _mm256_add_epi32(X0, Y0), (out0)[0], (out1)[0]);\
		STORE2(X1 = _mm256_add_epi32(X1, Y1), (out0)[1], (out1)[1]) \
		STORE2(X2 = _mm256_add_epi32(X2, Y2), (out0)[2], (out1)[2]) \
		STORE2(X3 = _mm256_add_epi32(X3, Y3), (out0)[3], (out1)[3]) \
	}

#define SALSA20_8(out) \
	SALSA20_8_BASE(__m128i, out)

#define SALSA20_8_VEC2(out0, out1) \
	SALSA20_8_BASE_VEC2(__m256i, out0, out1)

/**
 * Apply the salsa20/8 core to the block provided in (X0 ... X3) ^ (Z0 ... Z3).
 */
#define SALSA20_8_XOR_ANY(maybe_decl, Z0, Z1, Z2, Z3, out) \
	X0 = _mm_xor_si128(X0, Z0); \
	X1 = _mm_xor_si128(X1, Z1); \
	X2 = _mm_xor_si128(X2, Z2); \
	X3 = _mm_xor_si128(X3, Z3); \
	SALSA20_8_BASE(maybe_decl, out)

#define SALSA20_8_XOR_ANY_VEC2(maybe_decl, Z0, Z1, Z2, Z3, out0, out1) \
	X0 = _mm256_xor_si256(X0, Z0); \
	X1 = _mm256_xor_si256(X1, Z1); \
	X2 = _mm256_xor_si256(X2, Z2); \
	X3 = _mm256_xor_si256(X3, Z3); \
	SALSA20_8_BASE_VEC2(maybe_decl, out0, out1)

#define SALSA20_8_XOR_MEM(in, out) \
	SALSA20_8_XOR_ANY(__m128i, (in)[0], (in)[1], (in)[2], (in)[3], out)

#define SALSA20_8_XOR_MEM_VEC2(in0, in1, out0, out1) \
	SALSA20_8_XOR_ANY_VEC2(__m256i, \
      _mm256_setr_m128i((in0)[0], (in1)[0]), \
      _mm256_setr_m128i((in0)[1], (in1)[1]), \
      _mm256_setr_m128i((in0)[2], (in1)[2]), \
      _mm256_setr_m128i((in0)[3], (in1)[3]), \
      out0, out1)

#define SALSA20_8_XOR_REG(out) \
	SALSA20_8_XOR_ANY(/* empty */, Y0, Y1, Y2, Y3, out)

#define SALSA20_8_XOR_REG_VEC2(out0, out1) \
	SALSA20_8_XOR_ANY_VEC2(/* empty */, Y0, Y1, Y2, Y3, out0, out1)

typedef union {
	uint32_t w[16];
	__m128i q[4];
} salsa20_blk_t;

typedef struct {
  uint32_t x, y;
} uint32x2_t;

/**
 * blockmix_salsa8(Bin, Bout, r):
 * Compute Bout = BlockMix_{salsa20/8, r}(Bin).  The input Bin must be 128r
 * bytes in length; the output Bout must also be the same size.
 */
static inline void
blockmix_salsa8_vec2(const salsa20_blk_t *restrict Bin0, const salsa20_blk_t *restrict Bin1,
    salsa20_blk_t *restrict Bout0, salsa20_blk_t *restrict Bout1, size_t r)
{
	__m256i X0, X1, X2, X3;
	size_t i;

	r--;
	PREFETCH(&Bin0[r * 2 + 1], _MM_HINT_T0)
	PREFETCH(&Bin1[r * 2 + 1], _MM_HINT_T0)
	for (i = 0; i < r; i++) {
		PREFETCH(&Bin0[i * 2], _MM_HINT_T0)
		PREFETCH(&Bin1[i * 2], _MM_HINT_T0)
		PREFETCH_OUT(&Bout0[i], _MM_HINT_T0)
		PREFETCH_OUT(&Bout1[i], _MM_HINT_T0)
		PREFETCH(&Bin0[i * 2 + 1], _MM_HINT_T0)
		PREFETCH(&Bin1[i * 2 + 1], _MM_HINT_T0)
		PREFETCH_OUT(&Bout0[r + 1 + i], _MM_HINT_T0)
		PREFETCH_OUT(&Bout1[r + 1 + i], _MM_HINT_T0)
	}
	PREFETCH(&Bin0[r * 2], _MM_HINT_T0)
	PREFETCH(&Bin1[r * 2], _MM_HINT_T0)
	PREFETCH_OUT(&Bout0[r], _MM_HINT_T0)
	PREFETCH_OUT(&Bout1[r], _MM_HINT_T0)
	PREFETCH_OUT(&Bout0[r * 2 + 1], _MM_HINT_T0)
	PREFETCH_OUT(&Bout1[r * 2 + 1], _MM_HINT_T0)

	/* 1: X <-- B_{2r - 1} */
	X0 = _mm256_setr_m128i(Bin0[r * 2 + 1].q[0], Bin1[r * 2 + 1].q[0]);
	X1 = _mm256_setr_m128i(Bin0[r * 2 + 1].q[1], Bin1[r * 2 + 1].q[1]);
	X2 = _mm256_setr_m128i(Bin0[r * 2 + 1].q[2], Bin1[r * 2 + 1].q[2]);
	X3 = _mm256_setr_m128i(Bin0[r * 2 + 1].q[3], Bin1[r * 2 + 1].q[3]);

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	SALSA20_8_XOR_MEM_VEC2(Bin0[0].q, Bin1[0].q, Bout0[0].q, Bout1[0].q)

	/* 2: for i = 0 to 2r - 1 do */
	for (i = 0; i < r;) {
		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		SALSA20_8_XOR_MEM_VEC2(Bin0[i * 2 + 1].q, Bin1[i * 2 + 1].q, Bout0[r + 1 + i].q, Bout1[r + 1 + i].q)

		i++;

		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		SALSA20_8_XOR_MEM_VEC2(Bin0[i * 2].q, Bin1[i * 2].q, Bout0[i].q, Bout1[i].q)
	}

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	SALSA20_8_XOR_MEM_VEC2(Bin0[r * 2 + 1].q, Bin1[r * 2 + 1].q, Bout0[r * 2 + 1].q, Bout1[r * 2 + 1].q)
}

static inline void
blockmix_salsa8(const salsa20_blk_t *restrict Bin,
    salsa20_blk_t *restrict Bout, size_t r)
{
	__m128i X0, X1, X2, X3;
	size_t i;

	r--;
	PREFETCH(&Bin[r * 2 + 1], _MM_HINT_T0)
	for (i = 0; i < r; i++) {
		PREFETCH(&Bin[i * 2], _MM_HINT_T0)
		PREFETCH_OUT(&Bout[i], _MM_HINT_T0)
		PREFETCH(&Bin[i * 2 + 1], _MM_HINT_T0)
		PREFETCH_OUT(&Bout[r + 1 + i], _MM_HINT_T0)
	}
	PREFETCH(&Bin[r * 2], _MM_HINT_T0)
	PREFETCH_OUT(&Bout[r], _MM_HINT_T0)
	PREFETCH_OUT(&Bout[r * 2 + 1], _MM_HINT_T0)

	/* 1: X <-- B_{2r - 1} */
	X0 = Bin[r * 2 + 1].q[0];
	X1 = Bin[r * 2 + 1].q[1];
	X2 = Bin[r * 2 + 1].q[2];
	X3 = Bin[r * 2 + 1].q[3];

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	SALSA20_8_XOR_MEM(Bin[0].q, Bout[0].q)

	/* 2: for i = 0 to 2r - 1 do */
	for (i = 0; i < r;) {
		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		SALSA20_8_XOR_MEM(Bin[i * 2 + 1].q, Bout[r + 1 + i].q)

		i++;

		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		SALSA20_8_XOR_MEM(Bin[i * 2].q, Bout[i].q)
	}

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	SALSA20_8_XOR_MEM(Bin[r * 2 + 1].q, Bout[r * 2 + 1].q)
}

/*
 * (V)PSRLDQ and (V)PSHUFD have higher throughput than (V)PSRLQ on some CPUs
 * starting with Sandy Bridge.  Additionally, PSHUFD uses separate source and
 * destination registers, whereas the shifts would require an extra move
 * instruction for our code when building without AVX.  Unfortunately, PSHUFD
 * is much slower on Conroe (4 cycles latency vs. 1 cycle latency for PSRLQ)
 * and somewhat slower on some non-Intel CPUs (luckily not including AMD
 * Bulldozer and Piledriver).  Since for many other CPUs using (V)PSHUFD is a
 * win in terms of throughput or/and not needing a move instruction, we
 * currently use it despite of the higher latency on some older CPUs.  As an
 * alternative, the #if below may be patched to only enable use of (V)PSHUFD
 * when building with SSE4.1 or newer, which is not available on older CPUs
 * where this instruction has higher latency.
 */
#define HI32(X) \
	_mm_shuffle_epi32((X), _MM_SHUFFLE(2,3,0,1))

#define HI32_VEC2(X) \
	_mm256_shuffle_epi32((X), _MM_SHUFFLE(2,3,0,1))

#define EXTRACT64(X) _mm_cvtsi128_si64(X)
#define EXTRACT64_LO(X) _mm256_extract_epi64(X, 0)
#define EXTRACT64_HI(X) _mm256_extract_epi64(X, 2)

/* This is tunable */
#define S_BITS 8

/* Not tunable in this implementation, hard-coded in a few places */
#define S_SIMD 2
#define S_P 4

/* Number of S-boxes.  Not tunable by design, hard-coded in a few places. */
#define S_N 2

/* Derived values.  Not tunable except via S_BITS above. */
#define S_SIZE1 (1 << S_BITS)
#define S_MASK ((S_SIZE1 - 1) * S_SIMD * 8)
#define S_MASK2 (((uint64_t)S_MASK << 32) | S_MASK)
#define S_SIZE_ALL (S_N * S_SIZE1 * S_SIMD * 8)

/* 64-bit, or 32-bit without SSE4.1 */
#define PWXFORM_X_T uint64_t
#define PWXFORM_SIMD(X, x, s0, s1) \
	x = EXTRACT64(X) & S_MASK2; \
	s0 = *(const __m128i *)(S0 + (uint32_t)x); \
	s1 = *(const __m128i *)(S1 + (x >> 32)); \
	X = _mm_mul_epu32(HI32(X), X); \
	X = _mm_add_epi64(X, s0); \
	X = _mm_xor_si128(X, s1);

#define PWXFORM_SIMD_VEC2(X, x0, x1, s0, s1) \
	x0 = EXTRACT64_LO(X) & S_MASK2; \
	x1 = EXTRACT64_HI(X) & S_MASK2; \
	s0 = _mm256_setr_m128i(*(const __m128i *)(S00 + (uint32_t)x0), *(const __m128i *)(S01 + (uint32_t)x1)); \
	s1 = _mm256_setr_m128i(*(const __m128i *)(S10 + (x0 >> 32)), *(const __m128i *)(S11 + (x1 >> 32))); \
	X = _mm256_mul_epu32(HI32_VEC2(X), X); \
	X = _mm256_add_epi64(X, s0); \
	X = _mm256_xor_si256(X, s1);

#define PWXFORM_ROUND \
	PWXFORM_SIMD(X0, x0, s00, s01) \
	PWXFORM_SIMD(X1, x1, s10, s11) \
	PWXFORM_SIMD(X2, x2, s20, s21) \
	PWXFORM_SIMD(X3, x3, s30, s31)

#define PWXFORM_ROUND_VEC2 \
	PWXFORM_SIMD_VEC2(X0, x00, x01, s00, s01) \
	PWXFORM_SIMD_VEC2(X1, x10, x11, s10, s11) \
	PWXFORM_SIMD_VEC2(X2, x20, x21, s20, s21) \
	PWXFORM_SIMD_VEC2(X3, x30, x31, s30, s31)

#define PWXFORM \
	{ \
		PWXFORM_X_T x0, x1, x2, x3; \
		__m128i s00, s01, s10, s11, s20, s21, s30, s31; \
		__m128i X; \
		PWXFORM_ROUND PWXFORM_ROUND \
		PWXFORM_ROUND PWXFORM_ROUND \
		PWXFORM_ROUND PWXFORM_ROUND \
	}

#define PWXFORM_VEC2 \
	{ \
		PWXFORM_X_T x00, x10, x20, x30; \
		PWXFORM_X_T x01, x11, x21, x31; \
		__m256i s00, s01, s10, s11, s20, s21, s30, s31; \
		__m256i X; \
		PWXFORM_ROUND_VEC2 PWXFORM_ROUND_VEC2 \
		PWXFORM_ROUND_VEC2 PWXFORM_ROUND_VEC2 \
		PWXFORM_ROUND_VEC2 PWXFORM_ROUND_VEC2 \
	}

#define XOR4(in) \
	X0 = _mm_xor_si128(X0, (in)[0]); \
	X1 = _mm_xor_si128(X1, (in)[1]); \
	X2 = _mm_xor_si128(X2, (in)[2]); \
	X3 = _mm_xor_si128(X3, (in)[3]);

#define XOR4_VEC2(in0, in1) \
	X0 = _mm256_xor_si256(X0, _mm256_setr_m128i((in0)[0], (in1)[0])); \
	X1 = _mm256_xor_si256(X1, _mm256_setr_m128i((in0)[1], (in1)[1])); \
	X2 = _mm256_xor_si256(X2, _mm256_setr_m128i((in0)[2], (in1)[2])); \
	X3 = _mm256_xor_si256(X3, _mm256_setr_m128i((in0)[3], (in1)[3]));

#define OUT(out) \
	(out)[0] = X0; \
	(out)[1] = X1; \
	(out)[2] = X2; \
	(out)[3] = X3;

#define OUT_VEC2(out0, out1) \
	STORE2(X0, (out0)[0], (out1)[0]); \
	STORE2(X1, (out0)[1], (out1)[1]); \
	STORE2(X2, (out0)[2], (out1)[2]); \
	STORE2(X3, (out0)[3], (out1)[3]);

/**
 * blockmix_pwxform(Bin, Bout, r, S):
 * Compute Bout = BlockMix_pwxform{salsa20/8, r, S}(Bin).  The input Bin must
 * be 128r bytes in length; the output Bout must also be the same size.
 */
static void
blockmix_vec2(
    const salsa20_blk_t *restrict Bin0, const salsa20_blk_t *restrict Bin1,
    salsa20_blk_t *restrict Bout0, salsa20_blk_t *restrict Bout1,
    size_t r, const __m128i *restrict S0, const __m128i *restrict S1)
{
	const uint8_t * S00, * S10, * S01, * S11;
	__m256i X0, X1, X2, X3;
	size_t i;

	if (!S0 && !S1) {
		blockmix_salsa8_vec2(Bin0, Bin1, Bout0, Bout1, r);
		return;
	}

	S00 = (const uint8_t *)S0;
	S01 = (const uint8_t *)S1;
	S10 = (const uint8_t *)S0 + S_SIZE_ALL / 2;
	S11 = (const uint8_t *)S1 + S_SIZE_ALL / 2;

	/* Convert 128-byte blocks to 64-byte blocks */
	r *= 2;

	r--;
	PREFETCH(&Bin0[r], _MM_HINT_T0)
	PREFETCH(&Bin1[r], _MM_HINT_T0)
	for (i = 0; i < r; i++) {
		PREFETCH(&Bin0[i], _MM_HINT_T0)
		PREFETCH(&Bin1[i], _MM_HINT_T0)
		PREFETCH_OUT(&Bout0[i], _MM_HINT_T0)
		PREFETCH_OUT(&Bout1[i], _MM_HINT_T0)
	}
	PREFETCH_OUT(&Bout0[r], _MM_HINT_T0)
	PREFETCH_OUT(&Bout1[r], _MM_HINT_T0)

	/* X <-- B_{r1 - 1} */
	X0 = _mm256_setr_m128i(Bin0[r].q[0], Bin1[r].q[0]);
	X1 = _mm256_setr_m128i(Bin0[r].q[1], Bin1[r].q[1]);
	X2 = _mm256_setr_m128i(Bin0[r].q[2], Bin1[r].q[2]);
	X3 = _mm256_setr_m128i(Bin0[r].q[3], Bin1[r].q[3]);

	/* for i = 0 to r1 - 1 do */
	for (i = 0; i < r; i++) {
		/* X <-- H'(X \xor B_i) */
		XOR4_VEC2(Bin0[i].q, Bin1[i].q)
		PWXFORM_VEC2
		/* B'_i <-- X */
		OUT_VEC2(Bout0[i].q, Bout1[i].q)
	}

	/* Last iteration of the loop above */
	XOR4_VEC2(Bin0[i].q, Bin1[i].q)
	PWXFORM_VEC2

	/* B'_i <-- H(B'_i) */
	SALSA20_8_VEC2(Bout0[i].q, Bout1[i].q)
}

static void
blockmix(const salsa20_blk_t *restrict Bin, salsa20_blk_t *restrict Bout,
    size_t r, const __m128i *restrict S)
{
	const uint8_t * S0, * S1;
	__m128i X0, X1, X2, X3;
	size_t i;

	if (!S) {
		blockmix_salsa8(Bin, Bout, r);
		return;
	}

	S0 = (const uint8_t *)S;
	S1 = (const uint8_t *)S + S_SIZE_ALL / 2;

	/* Convert 128-byte blocks to 64-byte blocks */
	r *= 2;

	r--;
	PREFETCH(&Bin[r], _MM_HINT_T0)
	for (i = 0; i < r; i++) {
		PREFETCH(&Bin[i], _MM_HINT_T0)
		PREFETCH_OUT(&Bout[i], _MM_HINT_T0)
	}
	PREFETCH_OUT(&Bout[r], _MM_HINT_T0)

	/* X <-- B_{r1 - 1} */
	X0 = Bin[r].q[0];
	X1 = Bin[r].q[1];
	X2 = Bin[r].q[2];
	X3 = Bin[r].q[3];

	/* for i = 0 to r1 - 1 do */
	for (i = 0; i < r; i++) {
		/* X <-- H'(X \xor B_i) */
		XOR4(Bin[i].q)
		PWXFORM
		/* B'_i <-- X */
		OUT(Bout[i].q)
	}

	/* Last iteration of the loop above */
	XOR4(Bin[i].q)
	PWXFORM

	/* B'_i <-- H(B'_i) */
	SALSA20_8(Bout[i].q)
}

#define XOR4_2(in1, in2) \
	X0 = _mm_xor_si128((in1)[0], (in2)[0]); \
	X1 = _mm_xor_si128((in1)[1], (in2)[1]); \
	X2 = _mm_xor_si128((in1)[2], (in2)[2]); \
	X3 = _mm_xor_si128((in1)[3], (in2)[3]);

#define XOR4_2_VEC2(in10, in11, in20, in21) \
	X0 = _mm256_xor_si256(_mm256_setr_m128i((in10)[0], (in11)[0]), _mm256_setr_m128i((in20)[0], (in21)[0])); \
	X1 = _mm256_xor_si256(_mm256_setr_m128i((in10)[1], (in11)[1]), _mm256_setr_m128i((in20)[1], (in21)[1])); \
	X2 = _mm256_xor_si256(_mm256_setr_m128i((in10)[2], (in11)[2]), _mm256_setr_m128i((in20)[2], (in21)[2])); \
	X3 = _mm256_xor_si256(_mm256_setr_m128i((in10)[3], (in11)[3]), _mm256_setr_m128i((in20)[3], (in21)[3]));

static inline uint32x2_t
blockmix_salsa8_xor_vec2(const salsa20_blk_t *restrict Bin10, const salsa20_blk_t *restrict Bin11,
    const salsa20_blk_t *restrict Bin20, const salsa20_blk_t *restrict Bin21,
    salsa20_blk_t *restrict Bout0, salsa20_blk_t *restrict Bout1,
    size_t r, int Bin2_in_ROM)
{
	__m256i X0, X1, X2, X3;
	size_t i;

	r--;
	if (Bin2_in_ROM) {
		PREFETCH(&Bin20[r * 2 + 1], _MM_HINT_NTA)
		PREFETCH(&Bin21[r * 2 + 1], _MM_HINT_NTA)
		PREFETCH(&Bin10[r * 2 + 1], _MM_HINT_T0)
		PREFETCH(&Bin11[r * 2 + 1], _MM_HINT_T0)
		for (i = 0; i < r; i++) {
			PREFETCH(&Bin20[i * 2], _MM_HINT_NTA)
			PREFETCH(&Bin21[i * 2], _MM_HINT_NTA)
			PREFETCH(&Bin10[i * 2], _MM_HINT_T0)
			PREFETCH(&Bin11[i * 2], _MM_HINT_T0)
			PREFETCH(&Bin20[i * 2 + 1], _MM_HINT_NTA)
			PREFETCH(&Bin21[i * 2 + 1], _MM_HINT_NTA)
			PREFETCH(&Bin10[i * 2 + 1], _MM_HINT_T0)
			PREFETCH(&Bin11[i * 2 + 1], _MM_HINT_T0)
			PREFETCH_OUT(&Bout0[i], _MM_HINT_T0)
			PREFETCH_OUT(&Bout1[i], _MM_HINT_T0)
			PREFETCH_OUT(&Bout0[r + 1 + i], _MM_HINT_T0)
			PREFETCH_OUT(&Bout1[r + 1 + i], _MM_HINT_T0)
		}
		PREFETCH(&Bin20[r * 2], _MM_HINT_T0)
		PREFETCH(&Bin21[r * 2], _MM_HINT_T0)
	} else {
		PREFETCH(&Bin20[r * 2 + 1], _MM_HINT_T0)
		PREFETCH(&Bin21[r * 2 + 1], _MM_HINT_T0)
		PREFETCH(&Bin10[r * 2 + 1], _MM_HINT_T0)
		PREFETCH(&Bin11[r * 2 + 1], _MM_HINT_T0)
		for (i = 0; i < r; i++) {
			PREFETCH(&Bin20[i * 2], _MM_HINT_T0)
			PREFETCH(&Bin21[i * 2], _MM_HINT_T0)
			PREFETCH(&Bin10[i * 2], _MM_HINT_T0)
			PREFETCH(&Bin11[i * 2], _MM_HINT_T0)
			PREFETCH(&Bin20[i * 2 + 1], _MM_HINT_T0)
			PREFETCH(&Bin21[i * 2 + 1], _MM_HINT_T0)
			PREFETCH(&Bin10[i * 2 + 1], _MM_HINT_T0)
			PREFETCH(&Bin11[i * 2 + 1], _MM_HINT_T0)
			PREFETCH_OUT(&Bout0[i], _MM_HINT_T0)
			PREFETCH_OUT(&Bout1[i], _MM_HINT_T0)
			PREFETCH_OUT(&Bout0[r + 1 + i], _MM_HINT_T0)
			PREFETCH_OUT(&Bout1[r + 1 + i], _MM_HINT_T0)
		}
		PREFETCH(&Bin20[r * 2], _MM_HINT_T0)
		PREFETCH(&Bin21[r * 2], _MM_HINT_T0)
	}
	PREFETCH(&Bin10[r * 2], _MM_HINT_T0)
	PREFETCH(&Bin11[r * 2], _MM_HINT_T0)
	PREFETCH_OUT(&Bout0[r], _MM_HINT_T0)
	PREFETCH_OUT(&Bout1[r], _MM_HINT_T0)
	PREFETCH_OUT(&Bout0[r * 2 + 1], _MM_HINT_T0)
	PREFETCH_OUT(&Bout1[r * 2 + 1], _MM_HINT_T0)

	/* 1: X <-- B_{2r - 1} */
	XOR4_2_VEC2(Bin10[r * 2 + 1].q, Bin11[r * 2 + 1].q, Bin20[r * 2 + 1].q, Bin21[r * 2 + 1].q)

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	XOR4_VEC2(Bin10[0].q, Bin11[0].q)
	SALSA20_8_XOR_MEM_VEC2(Bin20[0].q, Bin21[0].q, Bout0[0].q, Bout1[0].q)

	/* 2: for i = 0 to 2r - 1 do */
	for (i = 0; i < r;) {
		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		XOR4_VEC2(Bin10[i * 2 + 1].q, Bin11[i * 2 + 1].q)
		SALSA20_8_XOR_MEM_VEC2(Bin20[i * 2 + 1].q, Bin21[i * 2 + 1].q, Bout0[r + 1 + i].q, Bout1[r + 1 + i].q)

		i++;

		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		XOR4_VEC2(Bin10[i * 2].q, Bin11[i * 2].q)
		SALSA20_8_XOR_MEM_VEC2(Bin20[i * 2].q, Bin21[i * 2].q, Bout0[i].q, Bout1[i].q)
	}

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	XOR4_VEC2(Bin10[r * 2 + 1].q, Bin11[r * 2 + 1].q)
	SALSA20_8_XOR_MEM_VEC2(Bin20[r * 2 + 1].q, Bin21[r * 2 + 1].q, Bout0[r * 2 + 1].q, Bout1[r * 2 + 1].q)

	return (uint32x2_t){_mm256_extract_epi32(X0, 0), _mm256_extract_epi32(X0, 4)};
}

static inline uint32_t
blockmix_salsa8_xor(const salsa20_blk_t *restrict Bin1,
    const salsa20_blk_t *restrict Bin2, salsa20_blk_t *restrict Bout,
    size_t r, int Bin2_in_ROM)
{
	__m128i X0, X1, X2, X3;
	size_t i;

	r--;
	if (Bin2_in_ROM) {
		PREFETCH(&Bin2[r * 2 + 1], _MM_HINT_NTA)
		PREFETCH(&Bin1[r * 2 + 1], _MM_HINT_T0)
		for (i = 0; i < r; i++) {
			PREFETCH(&Bin2[i * 2], _MM_HINT_NTA)
			PREFETCH(&Bin1[i * 2], _MM_HINT_T0)
			PREFETCH(&Bin2[i * 2 + 1], _MM_HINT_NTA)
			PREFETCH(&Bin1[i * 2 + 1], _MM_HINT_T0)
			PREFETCH_OUT(&Bout[i], _MM_HINT_T0)
			PREFETCH_OUT(&Bout[r + 1 + i], _MM_HINT_T0)
		}
		PREFETCH(&Bin2[r * 2], _MM_HINT_T0)
	} else {
		PREFETCH(&Bin2[r * 2 + 1], _MM_HINT_T0)
		PREFETCH(&Bin1[r * 2 + 1], _MM_HINT_T0)
		for (i = 0; i < r; i++) {
			PREFETCH(&Bin2[i * 2], _MM_HINT_T0)
			PREFETCH(&Bin1[i * 2], _MM_HINT_T0)
			PREFETCH(&Bin2[i * 2 + 1], _MM_HINT_T0)
			PREFETCH(&Bin1[i * 2 + 1], _MM_HINT_T0)
			PREFETCH_OUT(&Bout[i], _MM_HINT_T0)
			PREFETCH_OUT(&Bout[r + 1 + i], _MM_HINT_T0)
		}
		PREFETCH(&Bin2[r * 2], _MM_HINT_T0)
	}
	PREFETCH(&Bin1[r * 2], _MM_HINT_T0)
	PREFETCH_OUT(&Bout[r], _MM_HINT_T0)
	PREFETCH_OUT(&Bout[r * 2 + 1], _MM_HINT_T0)

	/* 1: X <-- B_{2r - 1} */
	XOR4_2(Bin1[r * 2 + 1].q, Bin2[r * 2 + 1].q)

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	XOR4(Bin1[0].q)
	SALSA20_8_XOR_MEM(Bin2[0].q, Bout[0].q)

	/* 2: for i = 0 to 2r - 1 do */
	for (i = 0; i < r;) {
		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		XOR4(Bin1[i * 2 + 1].q)
		SALSA20_8_XOR_MEM(Bin2[i * 2 + 1].q, Bout[r + 1 + i].q)

		i++;

		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		XOR4(Bin1[i * 2].q)
		SALSA20_8_XOR_MEM(Bin2[i * 2].q, Bout[i].q)
	}

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	XOR4(Bin1[r * 2 + 1].q)
	SALSA20_8_XOR_MEM(Bin2[r * 2 + 1].q, Bout[r * 2 + 1].q)

	return _mm_cvtsi128_si32(X0);
}

static uint32x2_t
blockmix_xor_vec2(
    const salsa20_blk_t *restrict Bin10, const salsa20_blk_t *restrict Bin11,
    const salsa20_blk_t *restrict Bin20, const salsa20_blk_t *restrict Bin21,
    salsa20_blk_t *restrict Bout0, salsa20_blk_t *restrict Bout1,
    size_t r, int Bin2_in_ROM, const __m128i *restrict S0, const __m128i *restrict S1)
{
	const uint8_t * S00, * S10, * S01, * S11;
	__m256i X0, X1, X2, X3;
	size_t i;

	if (!S0 && !S1)
		return blockmix_salsa8_xor_vec2(Bin10, Bin11, Bin20, Bin21, Bout0, Bout1, r, Bin2_in_ROM);

	S00 = (const uint8_t *)S0;
	S01 = (const uint8_t *)S1;
	S10 = (const uint8_t *)S0 + S_SIZE_ALL / 2;
	S11 = (const uint8_t *)S1 + S_SIZE_ALL / 2;

	/* Convert 128-byte blocks to 64-byte blocks */
	r *= 2;

	r--;
	if (Bin2_in_ROM) {
		PREFETCH(&Bin20[r], _MM_HINT_NTA)
		PREFETCH(&Bin21[r], _MM_HINT_NTA)
		PREFETCH(&Bin10[r], _MM_HINT_T0)
		PREFETCH(&Bin11[r], _MM_HINT_T0)
		for (i = 0; i < r; i++) {
			PREFETCH(&Bin20[i], _MM_HINT_NTA)
			PREFETCH(&Bin21[i], _MM_HINT_NTA)
			PREFETCH(&Bin10[i], _MM_HINT_T0)
			PREFETCH(&Bin11[i], _MM_HINT_T0)
			PREFETCH_OUT(&Bout0[i], _MM_HINT_T0)
			PREFETCH_OUT(&Bout1[i], _MM_HINT_T0)
		}
	} else {
		PREFETCH(&Bin20[r], _MM_HINT_T0)
		PREFETCH(&Bin21[r], _MM_HINT_T0)
		PREFETCH(&Bin10[r], _MM_HINT_T0)
		PREFETCH(&Bin11[r], _MM_HINT_T0)
		for (i = 0; i < r; i++) {
			PREFETCH(&Bin20[i], _MM_HINT_T0)
			PREFETCH(&Bin11[i], _MM_HINT_T0)
			PREFETCH_OUT(&Bout0[i], _MM_HINT_T0)
			PREFETCH_OUT(&Bout1[i], _MM_HINT_T0)
		}
	}
	PREFETCH_OUT(&Bout0[r], _MM_HINT_T0);
	PREFETCH_OUT(&Bout1[r], _MM_HINT_T0);

	/* X <-- B_{r1 - 1} */
	XOR4_2_VEC2(Bin10[r].q, Bin11[r].q, Bin20[r].q, Bin21[r].q)

	/* for i = 0 to r1 - 1 do */
	for (i = 0; i < r; i++) {
		/* X <-- H'(X \xor B_i) */
		XOR4_VEC2(Bin10[i].q, Bin11[i].q)
		XOR4_VEC2(Bin20[i].q, Bin21[i].q)
		PWXFORM_VEC2
		/* B'_i <-- X */
		OUT_VEC2(Bout0[i].q, Bout1[i].q)
	}

	/* Last iteration of the loop above */
	XOR4_VEC2(Bin10[i].q, Bin11[i].q)
	XOR4_VEC2(Bin20[i].q, Bin21[i].q)
	PWXFORM_VEC2

	/* B'_i <-- H(B'_i) */
	SALSA20_8_VEC2(Bout0[i].q, Bout1[i].q)

	return (uint32x2_t){_mm256_extract_epi32(X0, 0), _mm256_extract_epi32(X0, 4)};
}

static uint32_t
blockmix_xor(const salsa20_blk_t *restrict Bin1,
    const salsa20_blk_t *restrict Bin2, salsa20_blk_t *restrict Bout,
    size_t r, int Bin2_in_ROM, const __m128i *restrict S)
{
	const uint8_t * S0, * S1;
	__m128i X0, X1, X2, X3;
	size_t i;

	if (!S)
		return blockmix_salsa8_xor(Bin1, Bin2, Bout, r, Bin2_in_ROM);

	S0 = (const uint8_t *)S;
	S1 = (const uint8_t *)S + S_SIZE_ALL / 2;

	/* Convert 128-byte blocks to 64-byte blocks */
	r *= 2;

	r--;
	if (Bin2_in_ROM) {
		PREFETCH(&Bin2[r], _MM_HINT_NTA)
		PREFETCH(&Bin1[r], _MM_HINT_T0)
		for (i = 0; i < r; i++) {
			PREFETCH(&Bin2[i], _MM_HINT_NTA)
			PREFETCH(&Bin1[i], _MM_HINT_T0)
			PREFETCH_OUT(&Bout[i], _MM_HINT_T0)
		}
	} else {
		PREFETCH(&Bin2[r], _MM_HINT_T0)
		PREFETCH(&Bin1[r], _MM_HINT_T0)
		for (i = 0; i < r; i++) {
			PREFETCH(&Bin2[i], _MM_HINT_T0)
			PREFETCH(&Bin1[i], _MM_HINT_T0)
			PREFETCH_OUT(&Bout[i], _MM_HINT_T0)
		}
	}
	PREFETCH_OUT(&Bout[r], _MM_HINT_T0);

	/* X <-- B_{r1 - 1} */
	XOR4_2(Bin1[r].q, Bin2[r].q)

	/* for i = 0 to r1 - 1 do */
	for (i = 0; i < r; i++) {
		/* X <-- H'(X \xor B_i) */
		XOR4(Bin1[i].q)
		XOR4(Bin2[i].q)
		PWXFORM
		/* B'_i <-- X */
		OUT(Bout[i].q)
	}

	/* Last iteration of the loop above */
	XOR4(Bin1[i].q)
	XOR4(Bin2[i].q)
	PWXFORM

	/* B'_i <-- H(B'_i) */
	SALSA20_8(Bout[i].q)

	return _mm_cvtsi128_si32(X0);
}

#undef XOR4
#define XOR4(in, out) \
	(out)[0] = Y0 = _mm_xor_si128((in)[0], (out)[0]); \
	(out)[1] = Y1 = _mm_xor_si128((in)[1], (out)[1]); \
	(out)[2] = Y2 = _mm_xor_si128((in)[2], (out)[2]); \
	(out)[3] = Y3 = _mm_xor_si128((in)[3], (out)[3]);

#undef XOR4_VEC2
#define XOR4_VEC2(in0, in1, out0, out1) \
	STORE2(Y0 = _mm256_xor_si256(_mm256_setr_m128i((in0)[0], (in1)[0]), _mm256_setr_m128i((out0)[0], (out1)[0])), (out0)[0], (out1)[0]); \
	STORE2(Y1 = _mm256_xor_si256(_mm256_setr_m128i((in0)[1], (in1)[1]), _mm256_setr_m128i((out0)[1], (out1)[1])), (out0)[1], (out1)[1]); \
	STORE2(Y2 = _mm256_xor_si256(_mm256_setr_m128i((in0)[2], (in1)[2]), _mm256_setr_m128i((out0)[2], (out1)[2])), (out0)[2], (out1)[2]); \
	STORE2(Y3 = _mm256_xor_si256(_mm256_setr_m128i((in0)[3], (in1)[3]), _mm256_setr_m128i((out0)[3], (out1)[3])), (out0)[3], (out1)[3]);

static inline uint32x2_t
blockmix_salsa8_xor_save_vec2(
    const salsa20_blk_t *restrict Bin10, const salsa20_blk_t *restrict Bin11,
    salsa20_blk_t *restrict Bin20, salsa20_blk_t *restrict Bin21,
    salsa20_blk_t *restrict Bout0, salsa20_blk_t *restrict Bout1,
    size_t r)
{
	__m256i X0, X1, X2, X3, Y0, Y1, Y2, Y3;
	size_t i;

	r--;
	PREFETCH(&Bin20[r * 2 + 1], _MM_HINT_T0)
	PREFETCH(&Bin21[r * 2 + 1], _MM_HINT_T0)
	PREFETCH(&Bin10[r * 2 + 1], _MM_HINT_T0)
	PREFETCH(&Bin11[r * 2 + 1], _MM_HINT_T0)
	for (i = 0; i < r; i++) {
		PREFETCH(&Bin20[i * 2], _MM_HINT_T0)
		PREFETCH(&Bin21[i * 2], _MM_HINT_T0)
		PREFETCH(&Bin10[i * 2], _MM_HINT_T0)
		PREFETCH(&Bin11[i * 2], _MM_HINT_T0)
		PREFETCH(&Bin20[i * 2 + 1], _MM_HINT_T0)
		PREFETCH(&Bin21[i * 2 + 1], _MM_HINT_T0)
		PREFETCH(&Bin10[i * 2 + 1], _MM_HINT_T0)
		PREFETCH(&Bin11[i * 2 + 1], _MM_HINT_T0)
		PREFETCH_OUT(&Bout0[i], _MM_HINT_T0)
		PREFETCH_OUT(&Bout1[i], _MM_HINT_T0)
		PREFETCH_OUT(&Bout0[r + 1 + i], _MM_HINT_T0)
		PREFETCH_OUT(&Bout1[r + 1 + i], _MM_HINT_T0)
	}
	PREFETCH(&Bin20[r * 2], _MM_HINT_T0)
	PREFETCH(&Bin21[r * 2], _MM_HINT_T0)
	PREFETCH(&Bin10[r * 2], _MM_HINT_T0)
	PREFETCH(&Bin11[r * 2], _MM_HINT_T0)
	PREFETCH_OUT(&Bout0[r], _MM_HINT_T0)
	PREFETCH_OUT(&Bout1[r * 2 + 1], _MM_HINT_T0)

	/* 1: X <-- B_{2r - 1} */
	XOR4_2_VEC2(Bin10[r * 2 + 1].q, Bin11[r * 2 + 1].q, Bin20[r * 2 + 1].q, Bin21[r * 2 + 1].q)

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	XOR4_VEC2(Bin10[0].q, Bin11[0].q, Bin20[0].q, Bin21[0].q)
	SALSA20_8_XOR_REG_VEC2(Bout0[0].q, Bout1[0].q)

	/* 2: for i = 0 to 2r - 1 do */
	for (i = 0; i < r;) {
		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		XOR4_VEC2(Bin10[i * 2 + 1].q, Bin11[i * 2 + 1].q, Bin20[i * 2 + 1].q, Bin21[i * 2 + 1].q)
		SALSA20_8_XOR_REG_VEC2(Bout0[r + 1 + i].q, Bout1[r + 1 + i].q)

		i++;

		/* 3: X <-- H(X \xor B_i) */
		/* 4: Y_i <-- X */
		/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
		XOR4_VEC2(Bin10[i * 2].q, Bin11[i * 2].q, Bin20[i * 2].q, Bin21[i * 2].q)
		SALSA20_8_XOR_REG_VEC2(Bout0[i].q, Bout1[i].q)
	}

	/* 3: X <-- H(X \xor B_i) */
	/* 4: Y_i <-- X */
	/* 6: B' <-- (Y_0, Y_2 ... Y_{2r-2}, Y_1, Y_3 ... Y_{2r-1}) */
	XOR4_VEC2(Bin10[r * 2 + 1].q, Bin11[r * 2 + 1].q, Bin20[r * 2 + 1].q, Bin21[r * 2 + 1].q)
	SALSA20_8_XOR_REG_VEC2(Bout0[r * 2 + 1].q, Bout1[r * 2 + 1].q)

	return (uint32x2_t){_mm256_extract_epi32(X0, 0), _mm256_extract_epi32(X0, 4)};
}

#define XOR4_Y \
	X0 = _mm_xor_si128(X0, Y0); \
	X1 = _mm_xor_si128(X1, Y1); \
	X2 = _mm_xor_si128(X2, Y2); \
	X3 = _mm_xor_si128(X3, Y3);

#define XOR4_Y_VEC2 \
	X0 = _mm256_xor_si256(X0, Y0); \
	X1 = _mm256_xor_si256(X1, Y1); \
	X2 = _mm256_xor_si256(X2, Y2); \
	X3 = _mm256_xor_si256(X3, Y3);

static uint32x2_t
blockmix_xor_save_vec2(
    const salsa20_blk_t *restrict Bin10, const salsa20_blk_t *restrict Bin11,
    salsa20_blk_t *restrict Bin20, salsa20_blk_t *restrict Bin21,
    salsa20_blk_t *restrict Bout0, salsa20_blk_t *restrict Bout1,
    size_t r, const __m128i *restrict S0, const __m128i *restrict S1)
{
	const uint8_t * S00, * S10, * S01, * S11;
	__m256i X0, X1, X2, X3, Y0, Y1, Y2, Y3;
	size_t i;

	if (!S0 && !S1) {
    return blockmix_salsa8_xor_save_vec2(Bin10, Bin11, Bin20, Bin21, Bout0, Bout1, r);
  }

	S00 = (const uint8_t *)S0;
	S01 = (const uint8_t *)S1;
	S10 = (const uint8_t *)S0 + S_SIZE_ALL / 2;
	S11 = (const uint8_t *)S1 + S_SIZE_ALL / 2;

	/* Convert 128-byte blocks to 64-byte blocks */
	r *= 2;

	r--;
	PREFETCH(&Bin20[r], _MM_HINT_T0)
	PREFETCH(&Bin21[r], _MM_HINT_T0)
	PREFETCH(&Bin10[r], _MM_HINT_T0)
	PREFETCH(&Bin11[r], _MM_HINT_T0)
	for (i = 0; i < r; i++) {
		PREFETCH(&Bin20[i], _MM_HINT_T0)
		PREFETCH(&Bin21[i], _MM_HINT_T0)
		PREFETCH(&Bin10[i], _MM_HINT_T0)
		PREFETCH(&Bin11[i], _MM_HINT_T0)
		PREFETCH_OUT(&Bout0[i], _MM_HINT_T0)
		PREFETCH_OUT(&Bout1[i], _MM_HINT_T0)
	}
	PREFETCH_OUT(&Bout0[r], _MM_HINT_T0);
	PREFETCH_OUT(&Bout1[r], _MM_HINT_T0);

	/* X <-- B_{r1 - 1} */
	XOR4_2_VEC2(Bin10[r].q, Bin11[r].q, Bin20[r].q, Bin21[r].q)

	/* for i = 0 to r1 - 1 do */
	for (i = 0; i < r; i++) {
		XOR4_VEC2(Bin10[i].q, Bin11[i].q, Bin20[i].q, Bin21[i].q)
		/* X <-- H'(X \xor B_i) */
		XOR4_Y_VEC2
		PWXFORM_VEC2
		/* B'_i <-- X */
		OUT_VEC2(Bout0[i].q, Bout1[i].q)
	}

	/* Last iteration of the loop above */
	XOR4_VEC2(Bin10[i].q, Bin11[i].q, Bin20[i].q, Bin21[i].q)
	XOR4_Y_VEC2
	PWXFORM_VEC2

	/* B'_i <-- H(B'_i) */
	SALSA20_8_VEC2(Bout0[i].q, Bout1[i].q)

	return (uint32x2_t){_mm256_extract_epi32(X0, 0), _mm256_extract_epi32(X0, 4)};
}

#undef ARX
#undef SALSA20_2ROUNDS
#undef SALSA20_8
#undef SALSA20_8_XOR_ANY
#undef SALSA20_8_XOR_MEM
#undef SALSA20_8_XOR_REG
#undef PWXFORM_SIMD_1
#undef PWXFORM_SIMD_2
#undef PWXFORM_ROUND
#undef PWXFORM
#undef OUT
#undef XOR4
#undef XOR4_2
#undef XOR4_Y

/**
 * integerify(B, r):
 * Return the result of parsing B_{2r-1} as a little-endian integer.
 */
static inline uint32_t
integerify(const salsa20_blk_t * B, size_t r)
{
	return B[2 * r - 1].w[0];
}

/**
 * smix1(B, r, N, flags, V, NROM, shared, XY, S):
 * Compute first loop of B = SMix_r(B, N).  The input B must be 128r bytes in
 * length; the temporary storage V must be 128rN bytes in length; the temporary
 * storage XY must be 128r bytes in length.  The value N must be even and no
 * smaller than 2.  The array V must be aligned to a multiple of 64 bytes, and
 * arrays B and XY to a multiple of at least 16 bytes (aligning them to 64
 * bytes as well saves cache lines, but might result in cache bank conflicts).
 */
static void
smix1_vec2(uint8_t * B0, uint8_t * B1, size_t r, uint32_t N, yescrypt_flags_t flags,
    salsa20_blk_t * V0, salsa20_blk_t * V1, uint32_t NROM,
    const yescrypt_shared_t * shared0, const yescrypt_shared_t * shared1,
    salsa20_blk_t * XY0, salsa20_blk_t * XY1, void * S0, void * S1)
{
	const salsa20_blk_t * VROM0 = shared0->shared1.aligned;
	const salsa20_blk_t * VROM1 = shared1->shared1.aligned;
	uint32_t VROM_mask = shared0->mask1;
	size_t s = 2 * r;
	salsa20_blk_t * X0 = V0, * Y0;
	salsa20_blk_t * X1 = V1, * Y1;
	uint32_t i, j0, j1;
	size_t k;

	/* 1: X <-- B */
	/* 3: V_i <-- X */
	for (k = 0; k < 2 * r; k++) {
		for (i = 0; i < 16; i++) {
			X0[k].w[i] = le32dec(&B0[(k * 16 + (i * 5 % 16)) * 4]);
			X1[k].w[i] = le32dec(&B1[(k * 16 + (i * 5 % 16)) * 4]);
		}
	}

	if (NROM && (VROM_mask & 1)) {
		uint32_t n;
		salsa20_blk_t * V_n0;
		salsa20_blk_t * V_n1;
		const salsa20_blk_t * V_j0;
		const salsa20_blk_t * V_j1;

		/* 4: X <-- H(X) */
		/* 3: V_i <-- X */
		Y0 = &V0[s];
		Y1 = &V1[s];
    blockmix_vec2(X0, X1, Y0, Y1, r, S0, S1);

		X0 = &V0[2 * s];
		X1 = &V1[2 * s];
		if ((1 & VROM_mask) == 1) {
			/* j <-- Integerify(X) mod NROM */
			j0 = integerify(Y0, r) & (NROM - 1);
			j1 = integerify(Y1, r) & (NROM - 1);
			V_j0 = &VROM0[j0 * s];
			V_j1 = &VROM1[j1 * s];

			/* X <-- H(X \xor VROM_j) */
      uint32x2_t pj = blockmix_xor_vec2(Y0, Y1, V_j0, V_j1, X0, X1, r, 1, S0, S1);
			j0 = pj.x;
			j1 = pj.y;
		} else {
			/* X <-- H(X) */
      blockmix_vec2(Y0, Y1, X0, X1, r, S0, S1);
			j0 = integerify(X0, r);
			j1 = integerify(X1, r);
		}

		for (n = 2; n < N; n <<= 1) {
			uint32_t m = (n < N / 2) ? n : (N - 1 - n);

			V_n0 = &V0[n * s];
			V_n1 = &V1[n * s];

			/* 2: for i = 0 to N - 1 do */
			for (i = 1; i < m; i += 2) {
				/* j <-- Wrap(Integerify(X), i) */
				j0 &= n - 1;
				j1 &= n - 1;
				j0 += i - 1;
				j1 += i - 1;
				V_j0 = &V0[j0 * s];
				V_j1 = &V1[j1 * s];

				/* X <-- X \xor V_j */
				/* 4: X <-- H(X) */
				/* 3: V_i <-- X */
				Y0 = &V_n0[i * s];
				Y1 = &V_n1[i * s];
        uint32x2_t pj = blockmix_xor_vec2(X0, X1, V_j0, V_j1, Y0, Y1, r, 0, S0, S1);
				j0 = pj.x;
				j1 = pj.y;

				if (((n + i) & VROM_mask) == 1) {
					/* j <-- Integerify(X) mod NROM */
					j0 &= NROM - 1;
					j1 &= NROM - 1;
					V_j0 = &VROM0[j0 * s];
					V_j1 = &VROM1[j1 * s];
				} else {
					/* j <-- Wrap(Integerify(X), i) */
					j0 &= n - 1;
					j1 &= n - 1;
					j0 += i;
					j1 += i;
					V_j0 = &V0[j0 * s];
					V_j1 = &V1[j1 * s];
				}

				/* X <-- H(X \xor VROM_j) */
				X0 = &V_n0[(i + 1) * s];
				X1 = &V_n1[(i + 1) * s];
        pj = blockmix_xor_vec2(Y0, Y1, V_j0, V_j1, X0, X1, r, 1, S0, S1);
				j0 = pj.x;
				j1 = pj.y;
			}
		}

		n >>= 1;

		/* j <-- Wrap(Integerify(X), i) */
		j0 &= n - 1;
		j1 &= n - 1;
		j0 += N - 2 - n;
		j1 += N - 2 - n;
		V_j0 = &V0[j0 * s];
		V_j1 = &V1[j1 * s];

		/* X <-- X \xor V_j */
		/* 4: X <-- H(X) */
		/* 3: V_i <-- X */
		Y0 = &V0[(N - 1) * s];
		Y1 = &V1[(N - 1) * s];
    uint32x2_t pj = blockmix_xor_vec2(X0, X1, V_j0, V_j1, Y0, Y1, r, 0, S0, S1);
		j0 = pj.x;
		j1 = pj.y;

		if (((N - 1) & VROM_mask) == 1) {
			/* j <-- Integerify(X) mod NROM */
			j0 &= NROM - 1;
			j1 &= NROM - 1;
			V_j0 = &VROM0[j0 * s];
			V_j1 = &VROM1[j1 * s];
		} else {
			/* j <-- Wrap(Integerify(X), i) */
			j0 &= n - 1;
			j1 &= n - 1;
			j0 += N - 1 - n;
			j1 += N - 1 - n;
			V_j0 = &V0[j0 * s];
			V_j1 = &V1[j1 * s];
		}

		/* X <-- X \xor V_j */
		/* 4: X <-- H(X) */
		X0 = XY0;
		X1 = XY1;
    blockmix_xor_vec2(Y0, Y1, V_j0, V_j1, X0, X1, r, 1, S0, S1);
	} else if (flags & YESCRYPT_RW) {
		uint32_t n;
		salsa20_blk_t * V_n0, * V_n1, * V_j0, * V_j1;

		/* 4: X <-- H(X) */
		/* 3: V_i <-- X */
		Y0 = &V0[s];
		Y1 = &V1[s];
    blockmix_vec2(X0, X1, Y0, Y1, r, S0, S1);

		/* 4: X <-- H(X) */
		/* 3: V_i <-- X */
		X0 = &V0[2 * s];
		X1 = &V1[2 * s];
    blockmix_vec2(Y0, Y1, X0, X1, r, S0, S1);
		j0 = integerify(X0, r);
		j1 = integerify(X1, r);

		for (n = 2; n < N; n <<= 1) {
			uint32_t m = (n < N / 2) ? n : (N - 1 - n);

			V_n0 = &V0[n * s];
			V_n1 = &V1[n * s];

			/* 2: for i = 0 to N - 1 do */
			for (i = 1; i < m; i += 2) {
				Y0 = &V_n0[i * s];
				Y1 = &V_n1[i * s];

				/* j <-- Wrap(Integerify(X), i) */
				j0 &= n - 1;
				j1 &= n - 1;
				j0 += i - 1;
				j1 += i - 1;
				V_j0 = &V0[j0 * s];
				V_j1 = &V1[j1 * s];

				/* X <-- X \xor V_j */
				/* 4: X <-- H(X) */
				/* 3: V_i <-- X */
        uint32x2_t pj = blockmix_xor_vec2(X0, X1, V_j0, V_j1, Y0, Y1, r, 0, S0, S1);
				j0 = pj.x;
				j1 = pj.y;

				/* j <-- Wrap(Integerify(X), i) */
				j0 &= n - 1;
				j1 &= n - 1;
				j0 += i;
				j1 += i;
				V_j0 = &V0[j0 * s];
				V_j1 = &V1[j1 * s];

				/* X <-- X \xor V_j */
				/* 4: X <-- H(X) */
				/* 3: V_i <-- X */
				X0 = &V_n0[(i + 1) * s];
				X1 = &V_n1[(i + 1) * s];
        pj = blockmix_xor_vec2(Y0, Y1, V_j0, V_j1, X0, X1, r, 0, S0, S1);
				j0 = pj.x;
				j1 = pj.y;
			}
		}

		n >>= 1;

		/* j <-- Wrap(Integerify(X), i) */
		j0 &= n - 1;
		j1 &= n - 1;
		j0 += N - 2 - n;
		j1 += N - 2 - n;
		V_j0 = &V0[j0 * s];
		V_j1 = &V1[j1 * s];

		/* X <-- X \xor V_j */
		/* 4: X <-- H(X) */
		/* 3: V_i <-- X */
		Y0 = &V0[(N - 1) * s];
		Y1 = &V1[(N - 1) * s];
    uint32x2_t pj = blockmix_xor_vec2(X0, X1, V_j0, V_j1, Y0, Y1, r, 0, S0, S1);
		j0 = pj.x;
		j1 = pj.y;

		/* j <-- Wrap(Integerify(X), i) */
		j0 &= n - 1;
		j1 &= n - 1;
		j0 += N - 1 - n;
		j1 += N - 1 - n;
		V_j0 = &V0[j0 * s];
		V_j1 = &V1[j1 * s];

		/* X <-- X \xor V_j */
		/* 4: X <-- H(X) */
		X0 = XY0;
		X1 = XY1;
    blockmix_xor_vec2(Y0, Y1, V_j0, V_j1, X0, X1, r, 0, S0, S1);
	} else {
		/* 2: for i = 0 to N - 1 do */
		for (i = 1; i < N - 1; i += 2) {
			/* 4: X <-- H(X) */
			/* 3: V_i <-- X */
			Y0 = &V0[i * s];
			Y1 = &V1[i * s];
      blockmix_vec2(X0, X1, Y0, Y1, r, S0, S1);

			/* 4: X <-- H(X) */
			/* 3: V_i <-- X */
			X0 = &V0[(i + 1) * s];
			X1 = &V1[(i + 1) * s];
      blockmix_vec2(Y0, Y1, X0, X1, r, S0, S1);
		}

		/* 4: X <-- H(X) */
		/* 3: V_i <-- X */
		Y0 = &V0[i * s];
		Y1 = &V1[i * s];
    blockmix_vec2(X0, X1, Y0, Y1, r, S0, S1);

		/* 4: X <-- H(X) */
		X0 = XY0;
		X1 = XY1;
    blockmix_vec2(Y0, Y1, X0, X1, r, S0, S1);
	}

	/* B' <-- X */
	for (k = 0; k < 2 * r; k++) {
		for (i = 0; i < 16; i++) {
			le32enc(&B0[(k * 16 + (i * 5 % 16)) * 4], X0[k].w[i]);
			le32enc(&B1[(k * 16 + (i * 5 % 16)) * 4], X1[k].w[i]);
		}
	}
}

/**
 * smix2(B, r, N, Nloop, flags, V, NROM, shared, XY, S):
 * Compute second loop of B = SMix_r(B, N).  The input B must be 128r bytes in
 * length; the temporary storage V must be 128rN bytes in length; the temporary
 * storage XY must be 256r bytes in length.  The value N must be a power of 2
 * greater than 1.  The value Nloop must be even.  The array V must be aligned
 * to a multiple of 64 bytes, and arrays B and XY to a multiple of at least 16
 * bytes (aligning them to 64 bytes as well saves cache lines, but might result
 * in cache bank conflicts).
 */
static void
smix2_vec2(uint8_t * B0, uint8_t * B1, size_t r, uint32_t N, uint64_t Nloop,
    yescrypt_flags_t flags, salsa20_blk_t * V0, salsa20_blk_t * V1, uint32_t NROM,
    const yescrypt_shared_t * shared0, const yescrypt_shared_t * shared1,
    salsa20_blk_t * XY0, salsa20_blk_t * XY1, void * S0, void * S1)
{
	const salsa20_blk_t * VROM0 = shared0->shared1.aligned;
	const salsa20_blk_t * VROM1 = shared1->shared1.aligned;
	uint32_t VROM_mask = shared0->mask1;
	size_t s = 2 * r;
	salsa20_blk_t * X0 = XY0, * Y0 = &XY0[s];
	salsa20_blk_t * X1 = XY1, * Y1 = &XY1[s];
	uint64_t i;
	uint32_t j0, j1;
	size_t k;

	if (Nloop == 0)
		return;

	/* X <-- B' */
	/* 3: V_i <-- X */
	for (k = 0; k < 2 * r; k++) {
		for (i = 0; i < 16; i++) {
			X0[k].w[i] = le32dec(&B0[(k * 16 + (i * 5 % 16)) * 4]);
			X1[k].w[i] = le32dec(&B1[(k * 16 + (i * 5 % 16)) * 4]);
		}
	}

	i = Nloop / 2;

	/* 7: j <-- Integerify(X) mod N */
	j0 = integerify(X0, r) & (N - 1);
	j1 = integerify(X1, r) & (N - 1);

/*
 * Normally, NROM implies YESCRYPT_RW, but we check for these separately
 * because YESCRYPT_PARALLEL_SMIX resets YESCRYPT_RW for the smix2() calls
 * operating on the entire V.
 */
	if (NROM && (flags & YESCRYPT_RW)) {
		/* 6: for i = 0 to N - 1 do */
		for (i = 0; i < Nloop; i += 2) {
			salsa20_blk_t * V_j0 = &V0[j0 * s];
			salsa20_blk_t * V_j1 = &V1[j1 * s];

			/* 8: X <-- H(X \xor V_j) */
			/* V_j <-- Xprev \xor V_j */
			/* j <-- Integerify(X) mod NROM */
      uint32x2_t pj = blockmix_xor_save_vec2(X0, X1, V_j0, V_j1, Y0, Y1, r, S0, S1);
			j0 = pj.x;
			j1 = pj.y;

			if (((i + 1) & VROM_mask) == 1) {
				const salsa20_blk_t * VROM_j0;
				const salsa20_blk_t * VROM_j1;

				j0 &= NROM - 1;
				j1 &= NROM - 1;
				VROM_j0 = &VROM0[j0 * s];
				VROM_j1 = &VROM1[j1 * s];

				/* X <-- H(X \xor VROM_j) */
				/* 7: j <-- Integerify(X) mod N */
        uint32x2_t pj = blockmix_xor_vec2(Y0, Y1, VROM_j0, VROM_j1, X0, X1, r, 1, S0, S1);
				j0 = pj.x;
				j1 = pj.y;
			} else {
				j0 &= N - 1;
				j1 &= N - 1;
				V_j0 = &V0[j0 * s];
				V_j1 = &V1[j1 * s];

				/* 8: X <-- H(X \xor V_j) */
				/* V_j <-- Xprev \xor V_j */
				/* j <-- Integerify(X) mod NROM */
        uint32x2_t pj = blockmix_xor_save_vec2(Y0, Y1, V_j0, V_j1, X0, X1, r, S0, S1);
				j0 = pj.x;
				j1 = pj.y;
			}
			j0 &= N - 1;
			j1 &= N - 1;
			V_j0 = &V0[j0 * s];
			V_j1 = &V1[j1 * s];
		}
	} else if (NROM) {
		/* 6: for i = 0 to N - 1 do */
		for (i = 0; i < Nloop; i += 2) {
			const salsa20_blk_t * V_j0 = &V0[j0 * s];
			const salsa20_blk_t * V_j1 = &V1[j1 * s];

			/* 8: X <-- H(X \xor V_j) */
			/* V_j <-- Xprev \xor V_j */
			/* j <-- Integerify(X) mod NROM */
      uint32x2_t pj = blockmix_xor_vec2(X0, X1, V_j0, V_j1, Y0, Y1, r, 0, S0, S1);
			j0 = pj.x;
			j1 = pj.y;

			if (((i + 1) & VROM_mask) == 1) {
				j0 &= NROM - 1;
				j1 &= NROM - 1;
				V_j0 = &VROM0[j0 * s];
				V_j1 = &VROM1[j1 * s];
			} else {
				j0 &= N - 1;
				j1 &= N - 1;
				V_j0 = &V0[j0 * s];
				V_j1 = &V1[j1 * s];
			}

			/* X <-- H(X \xor VROM_j) */
			/* 7: j <-- Integerify(X) mod N */
      pj = blockmix_xor_vec2(Y0, Y1, V_j0, V_j1, X0, X1, r, 1, S0, S1);
			j0 = pj.x;
			j1 = pj.y;
			j0 &= N - 1;
			j1 &= N - 1;
			V_j0 = &V0[j0 * s];
			V_j1 = &V1[j1 * s];
		}
	} else if (flags & YESCRYPT_RW) {
		/* 6: for i = 0 to N - 1 do */
		do {
			salsa20_blk_t * V_j0 = &V0[j0 * s];
			salsa20_blk_t * V_j1 = &V1[j1 * s];

			/* 8: X <-- H(X \xor V_j) */
			/* V_j <-- Xprev \xor V_j */
			/* 7: j <-- Integerify(X) mod N */
      uint32x2_t pj = blockmix_xor_save_vec2(X0, X1, V_j0, V_j1, Y0, Y1, r, S0, S1);
			j0 = pj.x;
			j1 = pj.y;
			j0 &= N - 1;
			j1 &= N - 1;
			V_j0 = &V0[j0 * s];
			V_j1 = &V1[j1 * s];

			/* 8: X <-- H(X \xor V_j) */
			/* V_j <-- Xprev \xor V_j */
			/* 7: j <-- Integerify(X) mod N */
      pj = blockmix_xor_save_vec2(Y0, Y1, V_j0, V_j1, X0, X1, r, S0, S1);
			j0 = pj.x;
			j1 = pj.y;
			j0 &= N - 1;
			j1 &= N - 1;
		} while (--i);
	} else {
		/* 6: for i = 0 to N - 1 do */
		do {
			const salsa20_blk_t * V_j0 = &V0[j0 * s];
			const salsa20_blk_t * V_j1 = &V1[j1 * s];

			/* 8: X <-- H(X \xor V_j) */
			/* 7: j <-- Integerify(X) mod N */
      uint32x2_t pj = blockmix_xor_vec2(X0, X1, V_j0, V_j1, Y0, Y1, r, 0, S0, S1);
			j0 = pj.x;
			j1 = pj.y;
			j0 &= N - 1;
			j1 &= N - 1;
			V_j0 = &V0[j0 * s];
			V_j1 = &V1[j1 * s];

			/* 8: X <-- H(X \xor V_j) */
			/* 7: j <-- Integerify(X) mod N */
      pj = blockmix_xor_vec2(Y0, Y1, V_j0, V_j1, X0, X1, r, 0, S0, S1);
			j0 = pj.x;
			j1 = pj.y;
			j0 &= N - 1;
			j1 &= N - 1;
		} while (--i);
	}

	/* 10: B' <-- X */
	for (k = 0; k < 2 * r; k++) {
		for (i = 0; i < 16; i++) {
			le32enc(&B0[(k * 16 + (i * 5 % 16)) * 4], X0[k].w[i]);
			le32enc(&B1[(k * 16 + (i * 5 % 16)) * 4], X1[k].w[i]);
		}
	}
}

/**
 * p2floor(x):
 * Largest power of 2 not greater than argument.
 */
static uint64_t
p2floor(uint64_t x)
{
	uint64_t y;
	while ((y = x & (x - 1)))
		x = y;
	return x;
}

/**
 * smix(B, r, N, p, t, flags, V, NROM, shared, XY, S):
 * Compute B = SMix_r(B, N).  The input B must be 128rp bytes in length; the
 * temporary storage V must be 128rN bytes in length; the temporary storage XY
 * must be 256r or 256rp bytes in length (the larger size is required with
 * OpenMP-enabled builds).  The value N must be a power of 2 greater than 1.
 * The array V must be aligned to a multiple of 64 bytes, and arrays B and
 * XY to a multiple of at least 16 bytes (aligning them to 64 bytes as well
 * saves cache lines and helps avoid false sharing in OpenMP-enabled builds
 * when p > 1, but it might also result in cache bank conflicts).
 */
static void
smix_vec2(uint8_t * B0, uint8_t * B1, size_t r, uint32_t N, uint32_t p, uint32_t t,
    yescrypt_flags_t flags,
    salsa20_blk_t * V0, salsa20_blk_t * V1, uint32_t NROM,
    const yescrypt_shared_t * shared0, const yescrypt_shared_t * shared1,
    salsa20_blk_t * XY0, salsa20_blk_t * XY1, void * S0, void * S1)
{
	size_t s = 2 * r;
	uint32_t Nchunk = N / p;
	uint64_t Nloop_all, Nloop_rw;
	uint32_t i;

	Nloop_all = Nchunk;
	if (flags & YESCRYPT_RW) {
		if (t <= 1) {
			if (t)
				Nloop_all *= 2; /* 2/3 */
			Nloop_all = (Nloop_all + 2) / 3; /* 1/3, round up */
		} else {
			Nloop_all *= t - 1;
		}
	} else if (t) {
		if (t == 1)
			Nloop_all += (Nloop_all + 1) / 2; /* 1.5, round up */
		Nloop_all *= t;
	}

	Nloop_rw = 0;
	if (flags & __YESCRYPT_INIT_SHARED)
		Nloop_rw = Nloop_all;
	else if (flags & YESCRYPT_RW)
		Nloop_rw = Nloop_all / p;

	Nchunk &= ~(uint32_t)1; /* round down to even */
	Nloop_all++; Nloop_all &= ~(uint64_t)1; /* round up to even */
	Nloop_rw &= ~(uint64_t)1; /* round down to even */

#ifdef _OPENMP
#pragma omp parallel if (p > 1) default(none) private(i) shared(B, r, N, p, flags, V, NROM, shared, XY, S, s, Nchunk, Nloop_all, Nloop_rw)
	{
#pragma omp for
#endif
	for (i = 0; i < p; i++) {
		uint32_t Vchunk = i * Nchunk;
		uint8_t * Bp0 = &B0[128 * r * i];
		uint8_t * Bp1 = &B1[128 * r * i];
		salsa20_blk_t * Vp0 = &V0[Vchunk * s];
		salsa20_blk_t * Vp1 = &V1[Vchunk * s];
#ifdef _OPENMP
		salsa20_blk_t * XYp0 = &XY0[i * (2 * s)];
		salsa20_blk_t * XYp1 = &XY1[i * (2 * s)];
#else
		salsa20_blk_t * XYp0 = XY0;
		salsa20_blk_t * XYp1 = XY1;
#endif
		uint32_t Np = (i < p - 1) ? Nchunk : (N - Vchunk);
		void * Sp0 = S0 ? ((uint8_t *)S0 + i * S_SIZE_ALL) : S0;
		void * Sp1 = S1 ? ((uint8_t *)S1 + i * S_SIZE_ALL) : S1;
		if (Sp0 && Sp1) {
			//smix1(Bp0, 1, S_SIZE_ALL / 128,
			//    flags & ~YESCRYPT_PWXFORM,
			//    Sp0, NROM, shared0, XYp0, NULL);
			//smix1(Bp1, 1, S_SIZE_ALL / 128,
			//    flags & ~YESCRYPT_PWXFORM,
			//    Sp1, NROM, shared1, XYp1, NULL);
			smix1_vec2(Bp0, Bp1, 1, S_SIZE_ALL / 128,
			    flags & ~YESCRYPT_PWXFORM,
			    Sp0, Sp1, NROM, shared0, shared1, XYp0, XYp1, NULL, NULL);
    }
		if (!(flags & __YESCRYPT_INIT_SHARED_2)) {
			//smix1(Bp0, r, Np, flags, Vp0, NROM, shared0, XYp0, Sp0);
			//smix1(Bp1, r, Np, flags, Vp1, NROM, shared1, XYp1, Sp1);
			smix1_vec2(Bp0, Bp1, r, Np, flags, Vp0, Vp1, NROM, shared0, shared1, XYp0, XYp1, Sp0, Sp1);
    }
		smix2_vec2(Bp0, Bp1, r, p2floor(Np), Nloop_rw, flags, Vp0, Vp1,
		    NROM, shared0, shared1, XYp0, XYp1, Sp0, Sp1);
	}

	if (Nloop_all > Nloop_rw) {
#ifdef _OPENMP
#pragma omp for
#endif
		for (i = 0; i < p; i++) {
			uint8_t * Bp0 = &B0[128 * r * i];
			uint8_t * Bp1 = &B1[128 * r * i];
#ifdef _OPENMP
			salsa20_blk_t * XYp0 = &XY0[i * (2 * s)];
			salsa20_blk_t * XYp1 = &XY1[i * (2 * s)];
#else
			salsa20_blk_t * XYp0 = XY0;
			salsa20_blk_t * XYp1 = XY1;
#endif
			void * Sp0 = S0 ? ((uint8_t *)S0 + i * S_SIZE_ALL) : S0;
			void * Sp1 = S1 ? ((uint8_t *)S1 + i * S_SIZE_ALL) : S1;
			smix2_vec2(Bp0, Bp1, r, N, Nloop_all - Nloop_rw,
			    flags & ~YESCRYPT_RW, V0, V1, NROM, shared0, shared1, XYp0, XYp1, Sp0, Sp1);
		}
	}
#ifdef _OPENMP
	}
#endif
}

/**
 * yescrypt_kdf(shared, local, passwd, passwdlen, salt, saltlen,
 *     N, r, p, t, flags, buf, buflen):
 * Compute scrypt(passwd[0 .. passwdlen - 1], salt[0 .. saltlen - 1], N, r,
 * p, buflen), or a revision of scrypt as requested by flags and shared, and
 * write the result into buf.  The parameters r, p, and buflen must satisfy
 * r * p < 2^30 and buflen <= (2^32 - 1) * 32.  The parameter N must be a power
 * of 2 greater than 1.  (This optimized implementation currently additionally
 * limits N to the range from 8 to 2^31, but other implementation might not.)
 *
 * t controls computation time while not affecting peak memory usage.  shared
 * and flags may request special modes as described in yescrypt.h.  local is
 * the thread-local data structure, allowing to preserve and reuse a memory
 * allocation across calls, thereby reducing its overhead.
 *
 * Return 0 on success; or -1 on error.
 */
static int
yescrypt_kdf_vec2(const yescrypt_shared_t * shared0, const yescrypt_shared_t * shared1,
    yescrypt_local_t * local0, yescrypt_local_t * local1,
    const uint8_t * passwd0, const uint8_t * passwd1, size_t passwdlen,
    const uint8_t * salt0, const uint8_t * salt1, size_t saltlen,
    uint64_t N, uint32_t r, uint32_t p, uint32_t t, yescrypt_flags_t flags,
    uint8_t * buf0, uint8_t * buf1, size_t buflen)
{
	yescrypt_region_t tmp0, tmp1;
	uint64_t NROM;
	size_t B_size, V_size, XY_size, need;
	uint8_t * B0, * B1, * S0, * S1;
	salsa20_blk_t * V0, * V1, * XY0, * XY1;
	uint8_t sha256_0[32];
	uint8_t sha256_1[32];

	/*
	 * YESCRYPT_PARALLEL_SMIX is a no-op at p = 1 for its intended purpose,
	 * so don't let it have side-effects.  Without this adjustment, it'd
	 * enable the SHA-256 password pre-hashing and output post-hashing,
	 * because any deviation from classic scrypt implies those.
	 */
	if (p == 1)
		flags &= ~YESCRYPT_PARALLEL_SMIX;

	/* Sanity-check parameters */
	if (flags & ~YESCRYPT_KNOWN_FLAGS) {
		errno = EINVAL;
		return -1;
	}
#if SIZE_MAX > UINT32_MAX
	if (buflen > (((uint64_t)(1) << 32) - 1) * 32) {
		errno = EFBIG;
		return -1;
	}
#endif
	if ((uint64_t)(r) * (uint64_t)(p) >= (1 << 30)) {
		errno = EFBIG;
		return -1;
	}
	if (N > UINT32_MAX) {
		errno = EFBIG;
		return -1;
	}
	if (((N & (N - 1)) != 0) || (N <= 7) || (r < 1) || (p < 1)) {
		errno = EINVAL;
		return -1;
	}
	if ((flags & YESCRYPT_PARALLEL_SMIX) && (N / p <= 7)) {
		errno = EINVAL;
		return -1;
	}
	if ((r > SIZE_MAX / 256 / p) ||
	    (N > SIZE_MAX / 128 / r)) {
		errno = ENOMEM;
		return -1;
	}
#ifdef _OPENMP
	if (!(flags & YESCRYPT_PARALLEL_SMIX) &&
	    (N > SIZE_MAX / 128 / (r * p))) {
		errno = ENOMEM;
		return -1;
	}
#endif
	if ((flags & YESCRYPT_PWXFORM) &&
#ifndef _OPENMP
	    (flags & YESCRYPT_PARALLEL_SMIX) &&
#endif
	    p > SIZE_MAX / S_SIZE_ALL) {
		errno = ENOMEM;
		return -1;
	}

	NROM = 0;
	if (shared0->shared1.aligned) {
		NROM = shared0->shared1.aligned_size / ((size_t)128 * r);
		if (NROM > UINT32_MAX) {
			errno = EFBIG;
			return -1;
		}
		if (((NROM & (NROM - 1)) != 0) || (NROM <= 7) ||
		    !(flags & YESCRYPT_RW)) {
			errno = EINVAL;
			return -1;
		}
	}

	/* Allocate memory */
	V0 = NULL;
	V1 = NULL;
	V_size = (size_t)128 * r * N;
	need = V_size;
	if (flags & __YESCRYPT_INIT_SHARED) {
		if (local0->aligned_size < need) {
			if (local0->base || local0->aligned ||
			    local0->base_size || local0->aligned_size) {
				errno = EINVAL;
				return -1;
			}
			if (!alloc_region(local0, need))
				return -1;
			if (!alloc_region(local1, need))
				return -1;
		}
		V0 = (salsa20_blk_t *)local0->aligned;
		V1 = (salsa20_blk_t *)local1->aligned;
		need = 0;
	}
	B_size = (size_t)128 * r * p;
	need += B_size;
	if (need < B_size) {
		errno = ENOMEM;
		return -1;
	}
	XY_size = (size_t)256 * r;
	need += XY_size;
	if (need < XY_size) {
		errno = ENOMEM;
		return -1;
	}
	if (flags & YESCRYPT_PWXFORM) {
		size_t S_size = S_SIZE_ALL;
		need += S_size;
		if (need < S_size) {
			errno = ENOMEM;
			return -1;
		}
	}
	if (flags & __YESCRYPT_INIT_SHARED) {
		if (!alloc_region(&tmp0, need))
			return -1;
		if (!alloc_region(&tmp1, need))
			return -1;
		B0 = (uint8_t *)tmp0.aligned;
		B1 = (uint8_t *)tmp1.aligned;
		XY0 = (salsa20_blk_t *)((uint8_t *)B0 + B_size);
		XY1 = (salsa20_blk_t *)((uint8_t *)B1 + B_size);
	} else {
		init_region(&tmp0);
		init_region(&tmp1);
		if (local0->aligned_size < need) {
			if (free_region(local0))
				return -1;
			if (!alloc_region(local0, need))
				return -1;
		}
		if (local1->aligned_size < need) {
			if (free_region(local1))
				return -1;
			if (!alloc_region(local1, need))
				return -1;
		}
		B0 = (uint8_t *)local0->aligned;
		B1 = (uint8_t *)local1->aligned;
		V0 = (salsa20_blk_t *)((uint8_t *)B0 + B_size);
		V1 = (salsa20_blk_t *)((uint8_t *)B1 + B_size);
		XY0 = (salsa20_blk_t *)((uint8_t *)V0 + V_size);
		XY1 = (salsa20_blk_t *)((uint8_t *)V1 + V_size);
	}
	S0 = NULL;
	S1 = NULL;
	if (flags & YESCRYPT_PWXFORM) {
		S0 = (uint8_t *)XY0 + XY_size;
		S1 = (uint8_t *)XY1 + XY_size;
  }

	if (t || flags) {
		SHA256_CTX ctx_0;
		SHA256_Init(&ctx_0);
		SHA256_Update(&ctx_0, passwd0, passwdlen);
		SHA256_Final(sha256_0, &ctx_0);
		passwd0 = sha256_0;
		SHA256_CTX ctx_1;
		SHA256_Init(&ctx_1);
		SHA256_Update(&ctx_1, passwd1, passwdlen);
		SHA256_Final(sha256_1, &ctx_1);
		passwd1 = sha256_1;
		passwdlen = sizeof(sha256_0);
	}

	/* 1: (B_0 ... B_{p-1}) <-- PBKDF2(P, S, 1, p * MFLen) */
	PBKDF2_SHA256(passwd0, passwdlen, salt0, saltlen, 1, B0, B_size);
	PBKDF2_SHA256(passwd1, passwdlen, salt1, saltlen, 1, B1, B_size);

	if (t || flags) {
		memcpy(sha256_0, B0, sizeof(sha256_0));
		memcpy(sha256_1, B1, sizeof(sha256_1));
  }

	if (p == 1 || (flags & YESCRYPT_PARALLEL_SMIX)) { // TRUE
		smix_vec2(B0, B1, r, N, p, t, flags, V0, V1, NROM, shared0, shared1, XY0, XY1, S0, S1);
	} else {
//		uint32_t i;
//
//		/* 2: for i = 0 to p - 1 do */
//#ifdef _OPENMP
//#pragma omp parallel for default(none) private(i) shared(B, r, N, p, t, flags, V, NROM, shared, XY, S)
//#endif
//		for (i = 0; i < p; i++) {
//			/* 3: B_i <-- MF(B_i, N) */
//#ifdef _OPENMP
//			smix(&B[(size_t)128 * r * i], r, N, 1, t, flags,
//			    &V[(size_t)2 * r * i * N],
//			    NROM, shared,
//			    &XY[(size_t)4 * r * i],
//			    S ? &S[S_SIZE_ALL * i] : S);
//#else
//			smix(&B[(size_t)128 * r * i], r, N, 1, t, flags, V,
//			    NROM, shared, XY, S);
//#endif
//		}
	}

	/* 5: DK <-- PBKDF2(P, B, 1, dkLen) */
	PBKDF2_SHA256(passwd0, passwdlen, B0, B_size, 1, buf0, buflen);
	PBKDF2_SHA256(passwd1, passwdlen, B1, B_size, 1, buf1, buflen);

	/*
	 * Except when computing classic scrypt, allow all computation so far
	 * to be performed on the client.  The final steps below match those of
	 * SCRAM (RFC 5802), so that an extension of SCRAM (with the steps so
	 * far in place of SCRAM's use of PBKDF2 and with SHA-256 in place of
	 * SCRAM's use of SHA-1) would be usable with yescrypt hashes.
	 */
	if ((t || flags) && buflen == sizeof(sha256_0)) {
		/* Compute ClientKey */
		{
			HMAC_SHA256_CTX ctx_0;
			HMAC_SHA256_Init(&ctx_0, buf0, buflen);
			HMAC_SHA256_Update(&ctx_0, "Client Key", 10);
			HMAC_SHA256_Final(sha256_0, &ctx_0);
			HMAC_SHA256_CTX ctx_1;
			HMAC_SHA256_Init(&ctx_1, buf1, buflen);
			HMAC_SHA256_Update(&ctx_1, "Client Key", 10);
			HMAC_SHA256_Final(sha256_1, &ctx_1);
		}
		/* Compute StoredKey */
		{
			SHA256_CTX ctx_0;
			SHA256_Init(&ctx_0);
			SHA256_Update(&ctx_0, sha256_0, sizeof(sha256_0));
			SHA256_Final(buf0, &ctx_0);
			SHA256_CTX ctx_1;
			SHA256_Init(&ctx_1);
			SHA256_Update(&ctx_1, sha256_1, sizeof(sha256_1));
			SHA256_Final(buf1, &ctx_1);
		}
	}

	if (free_region(&tmp0))
		return -1;
	if (free_region(&tmp1))
		return -1;

	/* Success! */
	return 0;
}
