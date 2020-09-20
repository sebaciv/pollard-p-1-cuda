#ifndef __CPU_FACTOR_H__
#define __CPU_FACTOR_H__

#include <gmp.h>

int cpu_factorize(mpz_t n, const unsigned int primes[], const unsigned primes_num,
                  unsigned b_max,
                  unsigned b_start,
                  unsigned b_jump,
                  mpz_t *result,
                  unsigned *b_found);

#endif /* __CPU_FACTOR_H__ */
