#ifndef __KERNEL_H__
#define __KERNEL_H__

#include <gmp.h>

#define MAX_PRIMES 20000000

typedef unsigned long ULong;

int gpu_factorize(mpz_t n, const unsigned int *primes_table, const unsigned primes_num, unsigned b_max,
                  unsigned b_start,
                  unsigned b_jump,
                  mpz_t *factor,
                  unsigned *b_found);

int cudaInitialize();

unsigned *allocate_primes(const unsigned prime_table[], const unsigned primes_num);

int free_primes(unsigned *dev_primes);

#endif /* __KERNEL_H__ */
