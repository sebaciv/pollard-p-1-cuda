#include <cstdio>
#include <cmath>
#include <cassert>

#include <gmp.h>

void primes_power(mpz_t *e, const unsigned int *primes, const unsigned primes_num, unsigned B, mpz_t *tmp) {
    auto prime_i = (unsigned) primes[0];

    mpz_set_ui(*e, 1);

    for (unsigned i = 0; prime_i < B; i++) {
        unsigned power = (unsigned) (log((double) B) / log((double) prime_i));

        mpz_mul_ui(*tmp, *e, pow(prime_i, power));
        mpz_set(*e, *tmp);

        assert(i + 1 < primes_num);
        prime_i = primes[i + 1];
    }
}

int cpu_factorize(mpz_t n, const unsigned primes[], const unsigned primes_num, unsigned b_max,
               unsigned b_start,
               unsigned b_jump,
               mpz_t *result,
               unsigned *b_found) {
    unsigned B = b_start;
    const unsigned max_it = 1;
    unsigned iteration = 0;
    long long int globalIteration = 0;
    mpz_t a, d, e, b, tmp, one;

    mpz_init(a);
    mpz_init(d);
    mpz_init(e);
    mpz_init(b);
    mpz_init(tmp);
    mpz_init(one);
    mpz_init(*result);

    mpz_set_ui(a, 2);
    mpz_set_ui(one, 1);

    primes_power(&e, primes, primes_num, B, &tmp);
    for (iteration = 0; B < b_max; iteration++) {
        mpz_gcd(d, a, n);

        if (mpz_cmp(d, one) > 0) {
            mpz_set(*result, d);
            *b_found = B;
            printf("Found with B: %d\n", B);
            return 0;
        }

        mpz_powm(b, a, e, n); // b = (a ^ e) % n
        mpz_sub_ui(tmp, b, 1);
        mpz_set(b, tmp);

        mpz_gcd(d, b, n); // d = gcd(b, n)

        if (mpz_cmp(d, one) > 0 && mpz_cmp(d, n) < 0) {
            mpz_set(*result, d);
            *b_found = B;
            printf("B: %d\n", B);
            return 0;
        }

        mpz_set(b, a);
        mpz_add_ui(tmp, b, 1); // tmp = a+1

        if ((mpz_cmp(d, n) == 0 && mpz_cmp(tmp, n) < 0)  || (iteration > max_it)) {
            B += (globalIteration / 16 + 1) * b_jump;
            primes_power(&e, primes, primes_num, B, &tmp);
            iteration = 0;
        } else if (mpz_cmp(d, one) == 0) {
            mpz_add_ui(tmp, a, 1);
            mpz_set(a, tmp);
        } else {
            break;
        }
    }

    printf("Failed after %lld iterations!\n", globalIteration);
    return -1;
}
