#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cmath>

#include "kernel.h"

#include <gmp.h>
#include "cgbn/cgbn.h"

#define THREADS_PER_BLOCK 128

#define CGBN_CHECK(report) cgbn_check(report, __FILE__, __LINE__)

template<uint32_t tpi, uint32_t bits>
class pollard_params_t {
public:
    static const uint32_t TPI = tpi;                   // threads per instance
    static const uint32_t BITS = bits;                 // instance size
};

template<class params>
struct factor_result_t {
    cgbn_mem_t<params::BITS> factor;
    unsigned b;
};

void cgbn_check(cgbn_error_report_t *report, const char *file = nullptr, int32_t line = 0) {
    // check for cgbn errors

    if (cgbn_error_report_check(report)) {
        printf("\n");
        printf("CGBN error occurred: %s\n", cgbn_error_string(report));

        if (report->_instance != 0xFFFFFFFF) {
            printf("Error reported by instance %d", report->_instance);
            if (report->_blockIdx.x != 0xFFFFFFFF || report->_threadIdx.x != 0xFFFFFFFF)
                printf(", ");
            if (report->_blockIdx.x != 0xFFFFFFFF)
                printf("blockIdx=(%d, %d, %d) ", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
            if (report->_threadIdx.x != 0xFFFFFFFF)
                printf("threadIdx=(%d, %d, %d)", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
            printf("\n");
        } else {
            printf("Error reported by blockIdx=(%d %d %d)", report->_blockIdx.x, report->_blockIdx.y,
                   report->_blockIdx.z);
            printf("threadIdx=(%d %d %d)\n", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
        }
        if (file != nullptr)
            printf("file %s, line %d\n", file, line);
        exit(1);
    }
}

template<class params>
__global__
void parallel_factorize_kernel(cgbn_error_report_t *report,
                               cgbn_mem_t<params::BITS> n,
                               const unsigned *primes,
                               unsigned random_mul,
                               unsigned b_max,
                               unsigned b_start,
                               unsigned b_jump,
                               volatile bool *completed,
                               factor_result_t<params> *result) {
    typedef cgbn_context_t<params::TPI> context_t;
    typedef cgbn_env_t<context_t, params::BITS> env_t;
    typedef typename env_t::cgbn_t bn_t;

    if (*completed) return;

    const unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned instance = tid / params::TPI;
    const unsigned B = b_start + (b_jump * (blockIdx.x + 1)) * instance; // faster (other than linear) B growth ?
    if (B > b_max) return;

    const double log_b = log2((double) B);
    const unsigned log_b_ceil = ((unsigned) log_b) + 1;
    const unsigned prime_per_iter = params::BITS / log_b_ceil / 2;

    context_t bn_context(cgbn_report_monitor, report, instance);   // construct a context
    env_t bn_env(bn_context);                                  // construct an environment for big-int math

    unsigned primes_it;
    unsigned power;
    unsigned prime_i;

    bn_t N, a, d, e_sub, e, g, tmp;
    cgbn_load(bn_env, N, &n);
    cgbn_set_ui32(bn_env, a, (ULong) 2 + tid);
    cgbn_set_ui32(bn_env, d, 0);
    cgbn_set_ui32(bn_env, e, 0);
    cgbn_set_ui32(bn_env, e_sub, 0);
    cgbn_set_ui32(bn_env, g, 0);
    cgbn_set_ui32(bn_env, tmp, 0);

    // check NWD(a, N), if 0 then we have a factor else we can proceed with the algorithm
    cgbn_gcd(bn_env, d, a, N);
    if (cgbn_compare_ui32(bn_env, d, 1)) {
        *completed = true;
        cgbn_store(bn_env, &result->factor, d);
        result->b = B;
        return;
    }

    cgbn_set(bn_env, e, a); // e = a
    prime_i = (ULong) primes[0];
    for (primes_it = 0; prime_i <= B; primes_it += prime_per_iter) {
        if (*completed) return;
        cgbn_set_ui32(bn_env, e_sub, 1);

        for (unsigned i = 0; i < prime_per_iter && prime_i <= B; i++) {
            power = (unsigned) (log_b / log2((double) prime_i)); // p_pow_i = log(B) / log(p_i)
            cgbn_mul_ui32(bn_env, tmp, e_sub, (unsigned) pow((double) prime_i, power)); // e_sub *= p_i^p_pow_i
            cgbn_set(bn_env, e_sub, tmp);
            prime_i = primes[primes_it + i + 1];
        }
        cgbn_modular_power(bn_env, g, e, e_sub, N); // e = (e ** e_sub) % N - partial
        cgbn_set(bn_env, e, g);
    }

    if (!cgbn_equals_ui32(bn_env, e, 1)) {
        if (*completed) return;

        cgbn_sub_ui32(bn_env, g, e, 1); // g = e - 1
        cgbn_gcd(bn_env, d, g, N); // d = gcd(g, N)

        if (cgbn_compare_ui32(bn_env, d, 1) <= 0) {
            return;
        } else if (cgbn_compare(bn_env, d, N) >= 0) {
            return;
        } else { // factor found!
            *completed = true;
            cgbn_store(bn_env, &result->factor, d);
            result->b = B;
            return;
        }
    }
}

int cudaInitialize() {
    cudaError_t err;
    int num;
    if (cudaSuccess != (err = cudaGetDeviceCount(&num))) {
        fprintf(stderr, "Cannot get number of CUDA devices\nError [%d]%s\n", (int) err, cudaGetErrorString(err));
        return -1;
    };
    if (num < 1) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    };

    cudaDeviceProp prop;
    int MaxDevice = -1;
    int MaxGflops = -1;
    for (int dev = 0; dev < num; dev++) {
        if (cudaSuccess != (err = cudaGetDeviceProperties(&prop, dev))) {
            fprintf(stderr, "Error getting device %d properties\nError [%d]%s\n", dev, (int) err,
                    cudaGetErrorString(err));
            return -1;
        };
        int Gflops = prop.multiProcessorCount * prop.clockRate;
        printf("CUDA Device %d: %s Gflops %f Processors %d Threads/Block %d\n", dev, prop.name, 1e-6 * Gflops,
               prop.multiProcessorCount, prop.maxThreadsPerBlock);
        if (Gflops > MaxGflops) {
            MaxGflops = Gflops;
            MaxDevice = dev;
        };
    };
    printf("Fastest CUDA Device %d: %s\n", MaxDevice, prop.name);

    //  Print and set device
    if (cudaSuccess != (err = cudaGetDeviceProperties(&prop, MaxDevice))) {
        fprintf(stderr, "Error getting device %d properties\nError [%d]%s\n", MaxDevice, (int) err,
                cudaGetErrorString(err));
        return -1;
    };
    cudaSetDevice(MaxDevice);

    printf("TotalGlobalMem=%lu [MB]\n", (unsigned long) (prop.totalGlobalMem / 1024u / 1024u));
    printf("TotalConstMem=%lu [kB]\n", (unsigned long) (prop.totalConstMem / 1024u));
    printf("ClockRate=%d [MHz]\n", prop.clockRate / 1000);
    printf("MemoryClockRate=%d [MHz]\n", prop.memoryClockRate / 1000);

    printf("MaxTexture1D=%d\n", prop.maxTexture1D);
    printf("MaxTexture1DLinear=%u [KB]\n", prop.maxTexture1DLinear / 1024u);

    printf("MaxTexture2D=%d x %d\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
    printf("MaxTexture2DLinear=%d x %d\n", prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1]);

    printf("\n");
    return 0;
}

void to_mpz(mpz_t r, uint32_t *x, uint32_t count) {
    mpz_import(r, count, -1, sizeof(uint32_t), 0, 0, x);
}

void from_mpz(mpz_t s, uint32_t *x, uint32_t count) {
    size_t words;

    if (mpz_sizeinbase(s, 2) > count * 32) {
        fprintf(stderr, "from_mpz failed -- result does not fit\n");
        exit(1);
    }

    mpz_export(x, &words, -1, sizeof(uint32_t), 0, 0, s);
    while (words < count)
        x[words++] = 0;
}

unsigned *allocate_primes(const unsigned prime_table[], const unsigned primes_num) {
    cudaError_t err;

    unsigned *dev_primes;

    if (cudaSuccess != (err = cudaMalloc((void **) &dev_primes, primes_num * sizeof(prime_table[0])))) {
        fprintf(stderr, "Unable to allocate device prime table!\nError [%d]%s\n", (int) err, cudaGetErrorString(err));
        return nullptr;
    }

    if (cudaSuccess !=
        (err = cudaMemcpy(dev_primes, prime_table, primes_num * sizeof(prime_table[0]), cudaMemcpyHostToDevice))) {
        fprintf(stderr, "Unable to allocate device prime table!\nError [%d]%s\n", (int) err, cudaGetErrorString(err));
        return nullptr;
    }

    return dev_primes;
}

int free_primes(unsigned *dev_primes) {
    if (dev_primes != nullptr) cudaFree(dev_primes);
    return 0;
}

template<class params>
int parallel_factorize_param(mpz_t n,
                             const unsigned *gpu_primes_table,
                             const unsigned primes_num,
                             unsigned b_max,
                             unsigned b_start,
                             unsigned b_jump,
                             mpz_t *factor,
                             unsigned *b_found) {
    cudaError_t err;
    size_t result_size = sizeof(factor_result_t<params>);

    bool completed = false;
    unsigned start = 0;
    cgbn_mem_t<params::BITS> gpu_n;
    factor_result_t<params> *gpu_result = nullptr;
    factor_result_t<params> cpu_result;
    bool *gpu_completed = nullptr;
    unsigned *gpu_start = nullptr;
    cgbn_error_report_t *report;

    if (
            (cudaSuccess != (err = cudaMalloc((void **) &gpu_result, result_size))) ||
            (cudaSuccess != (err = cudaMalloc((void **) &gpu_completed, sizeof(bool)))) ||
            (cudaSuccess != (err = cudaMalloc((void **) &gpu_start, sizeof(unsigned)))) ||
            (cudaSuccess != (err = cudaMemset(gpu_result, 0L, result_size))) ||
            (cudaSuccess != (err = cudaMemset(gpu_completed, false, sizeof(bool)))) ||
            (cudaSuccess != (err = cudaMemset(gpu_start, 0L, sizeof(unsigned)))) ||
            (cudaSuccess != (err = cgbn_error_report_alloc(&report)))
            ) {
        fprintf(stderr, "Cannot allocate GPU memory!\nError [%d]%s\n", (int) err, cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    from_mpz(n, gpu_n._limbs, params::BITS / 32);
    unsigned randomMul = 4123457; //+ 1000 * rand() + rand();

    unsigned blocks_num = (b_max * params::TPI) / (b_jump * THREADS_PER_BLOCK);
    unsigned threads_per_block = THREADS_PER_BLOCK;
    parallel_factorize_kernel<params><<<blocks_num, threads_per_block>>>(report, gpu_n, gpu_primes_table, randomMul, b_max,
                                                                         b_start, b_jump,
                                                                         gpu_completed,
                                                                         gpu_result);

    if (cudaSuccess != (err = cudaDeviceSynchronize()))
        fprintf(stderr, "Unable to synchronize device!\nError [%d]%s\n", (int) err, cudaGetErrorString(err));

    CGBN_CHECK(report);

    if (cudaSuccess != (err = cudaMemcpy(&completed, gpu_completed, sizeof(bool), cudaMemcpyDeviceToHost))) {
        fprintf(stderr, "Unable to retrieve finished flag from host!\nError [%d]%s\n", (int) err,
                cudaGetErrorString(err));
        return -1;
    }

    if (cudaSuccess != (err = cudaMemcpy(&start, gpu_start, sizeof(start), cudaMemcpyDeviceToHost))) {
        fprintf(stderr, "Unable to retrieve work position from host!\nError [%d]%s\n", (int) err,
                cudaGetErrorString(err));
        return -1;
    }

    if (cudaSuccess != (err = cudaMemcpy(&cpu_result, gpu_result, result_size, cudaMemcpyDeviceToHost))) {
        fprintf(stderr, "Unable to retrieve result from host!\nError [%d]%s\n", (int) err, cudaGetErrorString(err));
        return -1;
    }

    to_mpz(*factor, cpu_result.factor._limbs, params::BITS / 32);
    printf("Found with B: %d\n", cpu_result.b);
    *b_found = cpu_result.b;

    if (gpu_result != nullptr) cudaFree(gpu_result);
    if (gpu_completed != nullptr) cudaFree(gpu_completed);
    if (gpu_completed != nullptr) cgbn_error_report_free(report);

    return 0;
}

int gpu_factorize(mpz_t n,
                  const unsigned int *primes_table,
                  const unsigned primes_num,
                  unsigned b_max,
                  unsigned b_start,
                  unsigned b_jump,
                  mpz_t *factor,
                  unsigned *b_found) {
    if (n->_mp_size < 2) {
        typedef pollard_params_t<4, 128> params;
        return parallel_factorize_param<params>(n, primes_table, primes_num, b_max, b_start, b_jump, factor, b_found);
    } else if (n->_mp_size < 4) {
        typedef pollard_params_t<8, 256> params;
        return parallel_factorize_param<params>(n, primes_table, primes_num, b_max, b_start, b_jump, factor, b_found);
    } else if (n->_mp_size < 8) {
        typedef pollard_params_t<16, 512> params;
        return parallel_factorize_param<params>(n, primes_table, primes_num, b_max, b_start, b_jump, factor, b_found);
    } else if (n->_mp_size < 16) {
        typedef pollard_params_t<32, 1024> params;
        return parallel_factorize_param<params>(n, primes_table, primes_num, b_max, b_start, b_jump, factor, b_found);
    } else {
        typedef pollard_params_t<32, 2048> params;
        return parallel_factorize_param<params>(n, primes_table, primes_num, b_max, b_start, b_jump, factor, b_found);
    }
}
