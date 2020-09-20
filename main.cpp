#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>

#include "common/get_timestamp.h"
#include "common/prime_table.h"
#include "pollard/kernel.h"
#include "pollard/cpu_factor.h"

#include <gmp.h>
#include <vector>

#define B_MAX 33554432 // 2^25
#define B_JUMP 2048
#define B_START 2

class FactorAlgorithm {
public:
    virtual int factorize_single(mpz_t n,
                                 unsigned b_max,
                                 unsigned b_start,
                                 unsigned b_jump,
                                 mpz_t *result,
                                 unsigned *b_found) = 0;

    virtual int initialize(const unsigned primes[], const unsigned primes_num) = 0;

    virtual int clean() = 0;

    int factorize(mpz_t n, mpz_t max_factor, std::vector<mpz_ptr> &all_factors, std::vector<unsigned> &all_powers) {
        mpz_t new_n, q, mod, zero, one, two;

        mpz_init(two);
        mpz_init(q);
        mpz_init(new_n);
        mpz_init(zero);
        mpz_init(one);
        mpz_init(mod);
        mpz_set(new_n, n);
        mpz_set_ui(zero, 0);
        mpz_set_ui(one, 1);
        mpz_set_ui(two, 2);
        unsigned int power_two = 0;
        unsigned int b_start = B_START;
        unsigned int factor_count = 0;

        const long long t_start = get_timestamp();
        fflush(stdout);

        while (mpz_cdiv_q_ui(q, new_n, 2) == 0) {
            mpz_set(new_n, q);
            power_two++;
        }

        if (power_two > 0) {
            mpz_set(all_factors[factor_count], two);
            all_powers.push_back(power_two);
            factor_count++;
        }

        int b_jump = B_JUMP;

        printf("---------\n");

        while (true) {
            mpz_t factor;
            mpz_init(factor);

            char num_str[1024] = {'\0'};
            mpz_get_str(num_str, 16, new_n);

            if (mpz_probab_prime_p(new_n, 50) != 0) {
                printf("Input is prime!\n");
                mpz_set(all_factors[factor_count++], new_n);
                all_powers.push_back(1);
                return 0;
            }

            printf("Sub-factoring 0x%s\n", num_str);
            fflush(stdout);

            const long long start_single = get_timestamp();
            unsigned b_found = 0;
            int returnVal = factorize_single(new_n, B_MAX, b_start, b_jump, &factor, &b_found);
            if (returnVal != 0) {
                return -1;
            }
            const long long elapsed_us_single = get_timestamp() - start_single;

            if (mpz_probab_prime_p(factor, 50) == 0) {
                b_jump = b_jump / 2;
                if (b_jump < 2) {
                    return -1;
                }
                continue;
            } else {
                b_jump = B_JUMP;
            }

            char factor_str[1024] = {'\0'};
            mpz_get_str(factor_str, 16, factor);
            printf("Single factor computed in %ld.%06ld s: 0x%s", (long) (elapsed_us_single / 1000000),
                   (long) (elapsed_us_single % 1000000), factor_str);

            unsigned int power = 0;
            do {
                mpz_cdiv_qr(q, mod, new_n, factor);
                if (mpz_cmp(mod, zero) == 0) {
                    power++;
                    mpz_set(new_n, q);
                } else {
                    break;
                }
            } while (true);

            if (power > 0) {
                printf(" - correct\n");
            } else {
                printf(" - incorrect!\n");
                return -1;
            }

            mpz_set(all_factors[factor_count++], factor);
            all_powers.push_back(power);

            if (mpz_cmp(new_n, one) == 0) {
                printf("All factors found!\n");
                break;
            } else if (mpz_probab_prime_p(new_n, 50) != 0) { // new_n is prime
                printf("Quotient is prime!\n");
                mpz_set(all_factors[factor_count++], new_n);
                all_powers.push_back(1);
                break;
            } else if (mpz_cmp(factor, max_factor) >= 0) { // factor is greater than required
                printf("Found all factors of required size!\n");
                break;
            }
        }

        const long long elapsed_us = get_timestamp() - t_start;
        printf("---------\n");
        printf("Factorization computed in %ld.%06ld s: ", (long) (elapsed_us / 1000000), (long) (elapsed_us % 1000000));

        return 0;
    }
};

class CPUFactorAlgorithm : public FactorAlgorithm {
public:
    const unsigned *dev_primes = nullptr;
    unsigned int primes_num_p = 0;

    int factorize_single(mpz_t n,
                         unsigned b_max,
                         unsigned b_start,
                         unsigned b_jump,
                         mpz_t *result,
                         unsigned *b_found) override {
        return cpu_factorize(n, dev_primes, primes_num_p, b_max, b_start, b_jump, result, b_found);
    }

    int initialize(const unsigned int *primes, const unsigned int primes_num) override {
        dev_primes = primes;
        primes_num_p = primes_num;
        return 0;
    }

    int clean() override {
        return 0;
    }
};

class GPUFactorAlgorithm : public FactorAlgorithm {
public:
    unsigned *dev_primes = nullptr;
    unsigned int primes_num_p = 0;

    int factorize_single(mpz_t n,
                         unsigned b_max,
                         unsigned b_start,
                         unsigned b_jump,
                         mpz_t *result,
                         unsigned *b_found) override {
        return gpu_factorize(n, dev_primes, primes_num_p, b_max, b_start, b_jump, result, b_found);
    }

    int initialize(const unsigned int *primes, const unsigned int primes_num) override {
        if (cudaInitialize() != 0) {
            return -1;
        }

        primes_num_p = primes_num;
        dev_primes = allocate_primes(primes, primes_num_p);
        if (dev_primes == nullptr) {
            return -1;
        }

        return 0;
    }

    int clean() override {
        return free_primes(dev_primes);
    }
};

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        fprintf(stderr, "Usage: %s [-n-1] (subtracts 1 from input number) [-cpu] (list of hex numbers to factor)\n", argv[0]);
        return -1;
    }

    srand(time(NULL));

    FactorAlgorithm *alg;

    if ((strcmp(argv[1], "-cpu")) != 0) {
        alg = new GPUFactorAlgorithm;
    } else {
        alg = new CPUFactorAlgorithm;
    }

    bool minus_one = (strcmp(argv[1], "-n-1") == 0);
    int number_list_start = ((strcmp(argv[1], "-cpu") == 0 || (argc > 2 && strcmp(argv[2], "-cpu") == 0)) ? 2 : 1);
    if (minus_one) {
        number_list_start++;
    }

    unsigned primes_num = MAX_PRIMES;
    unsigned *prime_table = (unsigned *) calloc(primes_num, sizeof(unsigned));
    if (generate_prime_table(prime_table, primes_num) != 0) {
        return -1;
    }

    mpz_t n, mod, two, max_factor;
    mpz_t f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12;

    mpz_init(n);
    mpz_init(mod);
    mpz_init(two);
    mpz_init(max_factor);
    mpz_init(f1);
    mpz_init(f2);
    mpz_init(f3);
    mpz_init(f4);
    mpz_init(f5);
    mpz_init(f6);
    mpz_init(f7);
    mpz_init(f8);
    mpz_init(f9);
    mpz_init(f10);
    mpz_init(f11);
    mpz_init(f12);
    mpz_set_ui(mod, 1);
    mpz_set_ui(two, 2);

    if (alg->initialize(prime_table, primes_num) != 0) {
        free(prime_table);
        return -1;
    }

    unsigned factored_count = 0;
    unsigned num = number_list_start;
    for (; num < argc; num++) {
        printf("\n<----------------------------------->\n");
        print_timestamp();
        const char *num_as_str = argv[num];
        mpz_set_str(n, num_as_str, 16);

        if (minus_one) {
            mpz_sub_ui(max_factor, n, 1);
            mpz_set(n, max_factor);
        }

        mpz_pow_ui(max_factor, two, 63); //TODO generic, for now needs to be changed manually for bigger inputs

        std::vector<mpz_ptr> all_factors(12);
        {
            all_factors[0] = f1;
            all_factors[1] = f2;
            all_factors[2] = f3;
            all_factors[3] = f4;
            all_factors[4] = f5;
            all_factors[5] = f6;
            all_factors[6] = f7;
            all_factors[7] = f8;
            all_factors[8] = f9;
            all_factors[9] = f10;
            all_factors[10] = f11;
            all_factors[11] = f12;
        }

        std::vector<unsigned> all_powers;

        printf("Factoring 0x%s\n", num_as_str);
        int resCode = alg->factorize(n, max_factor, all_factors, all_powers);
        if (resCode != 0 && all_powers.empty()) {
            fprintf(stderr, "Failed to factorize %d\n", resCode);
            continue;
        } else if (resCode != 0) {
            printf("Only partial factorization found!\n");
        }

        for (int i = 0; i < all_powers.size(); ++i) {
            char factor_str[1024] = {'\0'};
            auto x = all_factors[i];
            mpz_set(mod, x);
            mpz_get_str(factor_str, 16, mod);
            printf("0x%s ^ %d, ", factor_str, all_powers[i]);
        }

        if (all_powers.empty()) {
            printf("Factors not found!\n");
        } else {
            factored_count++;
            printf("\n");
        }

        print_timestamp();
    }

    printf("\n<----------------------------------->\n");
    printf("Test run completed! Success rate %d/%d", factored_count, num - number_list_start);

    alg->clean();

    mpz_clear(n);
    mpz_clear(mod);

    free(prime_table);
    return 0;
}
