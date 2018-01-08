/*-
 * Copyright 2014 Alexander Peslyak
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
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
 */
#include "yescrypt.h"
#include "sha256.c"
#include "yescrypt-best.c"

#define YESCRYPT_N 2048
#define YESCRYPT_R 8
#define YESCRYPT_P 1
#define YESCRYPT_T 0
#define YESCRYPT_FLAGS (YESCRYPT_RW | YESCRYPT_PWXFORM)

static int yescrypt_bitzeny_vec2(const uint8_t *passwd0, const uint8_t *passwd1, size_t passwdlen,
                            const uint8_t *salt0, const uint8_t *salt1, size_t saltlen,
                            uint8_t *buf0, uint8_t *buf1, size_t buflen)
{
    static __thread int initialized = 0;
    static __thread yescrypt_shared_t shared0;
    static __thread yescrypt_shared_t shared1;
    static __thread yescrypt_local_t local0;
    static __thread yescrypt_local_t local1;
    int retval;
    if (!initialized) {
        /* "shared" could in fact be shared, but it's simpler to keep it private
         * along with "local".  It's dummy and tiny anyway. */
        if (yescrypt_init_shared(&shared0, NULL, 0,
                                 0, 0, 0, YESCRYPT_SHARED_DEFAULTS, 0, NULL, 0))
            return -1;
        if (yescrypt_init_shared(&shared1, NULL, 0,
                                 0, 0, 0, YESCRYPT_SHARED_DEFAULTS, 0, NULL, 0)) {
            yescrypt_free_shared(&shared0);
            return -1;
        }
        if (yescrypt_init_local(&local0)) {
            yescrypt_free_shared(&shared0);
            yescrypt_free_shared(&shared1);
            return -1;
        }
        if (yescrypt_init_local(&local1)) {
            yescrypt_free_local(&local0);
            yescrypt_free_shared(&shared0);
            yescrypt_free_shared(&shared1);
            return -1;
        }
        initialized = 1;
    }
    retval = yescrypt_kdf_vec2(&shared0, &shared1, &local0, &local1, passwd0, passwd1, passwdlen, salt0, salt1, saltlen,
                          YESCRYPT_N, YESCRYPT_R, YESCRYPT_P, YESCRYPT_T,
                          YESCRYPT_FLAGS, buf0, buf1, buflen);
#if 0
    if (yescrypt_free_local(&local)) {
        yescrypt_free_shared(&shared);
        return -1;
    }
    if (yescrypt_free_shared(&shared))
        return -1;
    initialized = 0;
#endif
    if (retval < 0) {
        yescrypt_free_local(&local0);
        yescrypt_free_local(&local1);
        yescrypt_free_shared(&shared0);
        yescrypt_free_shared(&shared1);
    }
    return retval;
}

static void yescrypt_hash_vec2(const char *input0, const char *input1,
    char *output0, char *output1)
{
    yescrypt_bitzeny_vec2((const uint8_t *) input0, (const uint8_t *)input1, 80,
                     (const uint8_t *) input0, (const uint8_t *)input1, 80,
                     (uint8_t *) output0, (uint8_t *)output1, 32);
}

#include <stdbool.h>
struct work_restart {
	volatile unsigned long	restart;
	char			padding[128 - sizeof(unsigned long)];
};

extern struct work_restart *work_restart;
extern bool fulltest(const uint32_t *hash, const uint32_t *target);

static int pretest(const uint32_t *hash, const uint32_t *target)
{
	return hash[7] < target[7];
}

int scanhash_yescrypt(int thr_id, uint32_t *pdata, const uint32_t *ptarget,
		      uint32_t max_nonce, unsigned long *hashes_done)
{
	uint32_t data0[20] __attribute__((aligned(128)));
	uint32_t data1[20] __attribute__((aligned(128)));
	uint32_t hash0[8] __attribute__((aligned(32)));
	uint32_t hash1[8] __attribute__((aligned(32)));
	uint32_t n = pdata[19] - 1;
	const uint32_t first_nonce = pdata[19];

	for (int i = 0; i < 20; i++) {
		be32enc(&data0[i], pdata[i]);
		be32enc(&data1[i], pdata[i]);
	}
	do {
		be32enc(&data0[19], ++n);
		be32enc(&data1[19], ++n);
		yescrypt_hash_vec2((char *)data0, (char *)data1, (char *)hash0, (char *)hash1);
		if (pretest(hash0, ptarget) && fulltest(hash0, ptarget)) {
			pdata[19] = n-1;
			*hashes_done = n - first_nonce + 1;
			return 1;
		}
		if (pretest(hash1, ptarget) && fulltest(hash1, ptarget)) {
			pdata[19] = n;
			*hashes_done = n - first_nonce + 1;
			return 1;
		}
	} while (n < max_nonce && !work_restart[thr_id].restart);
	
	*hashes_done = n - first_nonce + 1;
	pdata[19] = n;
	return 0;
}

