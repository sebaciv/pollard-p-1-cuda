#define _CRT_SECURE_NO_WARNINGS

#include <cstdio>
#include "prime_table.h"
#include "../primegen/primegen.h"

#define LOAD_FROM_BINARY 1

static void get_prime_table(unsigned primes[], unsigned &n) {
    primegen pg;
    primegen_init(&pg);

    const unsigned primes_max = n;
    for (unsigned i = 0; i < primes_max; i++) {
        uint64 prime_number = primegen_next(&pg);
        if (prime_number > (uint64) 0xFFFFFFFFU) {
            n = i;
            break;
        };
        primes[i] = (unsigned) prime_number;
    }
}

static int check_prime_table(const unsigned primes[], const unsigned primes_num) {
    for (unsigned p = 0; p < primes_num; p++) {
        if (primes[p] <= 1 ||
            p > 0 && primes[p] < primes[p - 1]) {
            printf("Invalid prime number %lu (0x%08lu) at position %u\n", (unsigned long) primes[p],
                   (unsigned long) primes[p], p);
            return -1;
        }
    }
    return 0;
}

int generate_prime_table(unsigned primes[], unsigned &primes_num) {
#ifdef LOAD_FROM_BINARY
    const char primes_numbers_list_filename[] = "prime_numbers_list.bin";
#else
    const char primes_numbers_list_filename[] = "prime_numbers_list.txt";
#endif

    int list_loades_from_file = 0;
#ifdef LOAD_FROM_BINARY
    FILE *pFile = fopen(primes_numbers_list_filename, "rb");
#else
    FILE * pFile = fopen(primes_numbers_list_filename, "r");
#endif
    if (pFile != nullptr) {
#ifdef LOAD_FROM_BINARY
        auto p = (unsigned) fread(primes, sizeof(primes[0]), primes_num, pFile);
        if (p > 0) {
            list_loades_from_file = 1;
            primes_num = p;
        } else
            list_loades_from_file = 0;
#else
        setvbuf(pFile, NULL, _IOFBF, 16 * 1024u);
        unsigned p;
        list_loades_from_file = 1;
        for (p = 0; p < primes_num; p++)
        {
           unsigned long prime = 0;
           if (fscanf(pFile, "%u", &prime) != 1 || prime == 0)
           {
              fprintf(stderr, "Unable to read %u prime from file: %s\n", p, primes_numbers_list_filename);
              list_loades_from_file = 0;
              break;
           };
           primes[p] = prime;
        };
#endif
        fclose(pFile);
        printf("Loaded %u prime numbers from file: %s\n", primes_num, primes_numbers_list_filename);
    };

    if (!list_loades_from_file) {
        printf("Generating prime table...");
        fflush(stdout);
        get_prime_table(primes, primes_num);
        printf("Finished generating prime table!\n");
    };

    if (!list_loades_from_file) {
        //save to file
#ifdef LOAD_FROM_BINARY
        pFile = fopen(primes_numbers_list_filename, "wb");
#else
        pFile = fopen(primes_numbers_list_filename, "w");
#endif
        if (pFile != nullptr) {
#ifdef LOAD_FROM_BINARY
            auto p = (unsigned) fwrite(primes, sizeof(primes[0]), primes_num, pFile);
#else
            setvbuf(pFile, NULL, _IOFBF, 16 * 1024u);
            unsigned p;
            for (p = 0; p < primes_num; p++)
            {
               int res;
               if (p > 0)
               {
                  res = fprintf(pFile, p % 10 == 0 ? "\n" : "\t");
                  if (res <= 0)
                  {
                     fprintf(stderr, "Unable to write to file: %s\n", primes_numbers_list_filename);
                     break;
                  };
               };

               res = fprintf(pFile, "%u", primes[p]);
               if (res <= 0)
               {
                  fprintf(stderr, "Unable to write %u prime to file: %s\n", p, primes_numbers_list_filename);
                  break;
               };
            };

            fprintf(pFile, "\n");
#endif
            fclose(pFile);
            printf("Save %u/%u prime numbers to file: %s\n", p, primes_num, primes_numbers_list_filename);
        };
    };

#ifdef _DEBUG
    if (check_prime_table(primes, primes_num) != 0)
       return -1;
#endif

    printf("Last generated prime number: %u (0x%08x)\n", primes[primes_num - 1], primes[primes_num - 1]);
    return 0;
}
