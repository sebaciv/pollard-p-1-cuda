cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.0/bin/nvcc" CACHE STRING "")
project(cuda_rsa LANGUAGES CUDA CXX)

include_directories(cgbn/include)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -gencode arch=compute_61,code=sm_61 -O2 -I$(GMP_HOME)/include -L$(GMP_HOME)/lib -Icgbn/include -lgmp
)

add_executable(cuda_rsa main.cpp
        common common/get_timestamp.cpp common/get_timestamp.h common/prime_table.cpp common/prime_table.h
        primegen primegen/int64.h primegen/primegen.cpp primegen/primegen.h primegen/primegen_impl.h primegen/primegen_init.cpp primegen/primegen_next.cpp primegen/primegen_skip.cpp primegen/uint32.h primegen/uint64.h
        pollard pollard/kernel.cu pollard/kernel.h pollard/cpu_factor.cpp pollard/cpu_factor.h
        )
target_link_libraries(cuda_rsa gmp)

set_target_properties(
        cuda_rsa
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_data test-generator/main.cpp)
target_link_libraries(test_data gmp)
