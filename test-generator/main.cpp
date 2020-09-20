#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

#include <gmp.h>

#define MAX_TRIES 10000

gmp_randstate_t random_state;

void random_num(size_t size, mpz_t* result) {
    mpz_rrandomb(*result, random_state, size);
}

int random_factor(mpz_t b, size_t b_size, size_t factor_size, mpz_t* result) {
    int i;
    mpz_t f, f1, num;
    mpz_init(f);
    mpz_init(f1);
    mpz_init(num);
    size_t num_size = factor_size - b_size;
    
    for (i = 0; i < MAX_TRIES; ++i) {
        random_num(num_size, &num);
        mpz_mul(f, num, b);
        mpz_add_ui(f1, f, 1);
        
        if (mpz_probab_prime_p(f1, 50) != 0) {
            mpz_set(*result, f1);
            break;
        }
    }

    mpz_clear(f);
    mpz_clear(f1);
    mpz_clear(num);
    if (i < MAX_TRIES) {
        return 0;
    } else {
        return -1;
    }
}

int main(int argc, char *argv[]) {
    if (argc <= 2) {
        fprintf(stderr, "Usage: %s size_of_n <list of b factors>\n", argv[0]);
        return -1;
    }

    srand(time(NULL));
    gmp_randinit_mt(random_state);

    size_t size_n = std::stoi(argv[1]);
    mpz_t b, p, q, n, n1, zero, rem;
    mpz_init(b);
    mpz_init(p);
    mpz_init(q);
    mpz_init(n);
    mpz_init(n1);
    mpz_init(zero);
    mpz_init(rem);
    mpz_set_ui(zero, 0);

    for (int num_to_factor = 2; num_to_factor < argc; num_to_factor++) {
        printf("<----------------------------------------------->\n");
        const char *num_str = argv[num_to_factor];
        mpz_set_str(b, num_str, 16);

        if (mpz_probab_prime_p(b, 50) == 0) {
            printf("Factor b: 0x%s is not a prime!\n", num_str);
            continue;
        }

        printf("Trying to find N for b: 0x%s\n", num_str);
        size_t b_size = mpz_sizeinbase(b, 2);
        size_t factor_size = size_n / 2;
        int diff = 3 + rand() % 4;
        size_t p_size = factor_size - diff;
        size_t q_size = factor_size + diff;

        int res_p = random_factor(b, b_size, p_size - 1, &p);
        if (res_p != 0) {
            printf("Could not compute p factor!\n");
            continue;
        }
        int res_q = random_factor(b, b_size, q_size, &q);
        if (res_q != 0) {
            printf("Could not compute q factor!\n");
            continue;
        }
        mpz_mul(n, p, q);

        char p_str[1024] = {'\0'};
        mpz_get_str(p_str, 16, p);
        printf("Found p: 0x%s\n", p_str);

        char q_str[1024] = {'\0'};
        mpz_get_str(q_str, 16, q);
        printf("Found q: 0x%s\n", q_str);

        char n_str[1024] = {'\0'};
        mpz_get_str(n_str, 16, n);
        printf("Found N: 0x%s\n", n_str);

        mpz_sub_ui(n1, n, 1);
        char n1_str[1024] = {'\0'};
        mpz_get_str(n1_str, 16, n1);
        printf("Found N - 1: 0x%s\n", n1_str);

        mpz_mod(rem, n1, b);
        if (mpz_cmp(rem, zero) != 0) {
            printf("Wrong!\n");
        }
    }

    mpz_clear(b);
    mpz_clear(p);
    mpz_clear(q);
    mpz_clear(n);
    mpz_clear(n1);
    return 0;
}
