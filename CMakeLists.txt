cmake_minimum_required(VERSION 3.20)
project(CUDAProb3 VERSION 2.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES "86")

set(CMAKE_CXX_COMPILER /usr/bin/g++-15)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-15)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xcompiler=-Wall,-Wextra")

option(CUDA_PROB3_BUILD_TESTS "Build tests" ON)
option(CUDA_PROB3_BUILD_EXAMPLES "Build example programs" ON)
option(CUDA_PROB3_BUILD_BENCHMARKS "Build benchmarks" ON)

find_package(CUDAToolkit REQUIRED)

set(SOURCES
    src/core/matrix3x3.cu
    src/physics/mixing.cu
    src/physics/matter.cu
    src/physics/amplitudes.cu
    src/physics/layer_integrator.cu
    src/engine/device.cu
    src/engine/memory_pool.cu
    src/engine/stream_pool.cu
    src/propagator/single_gpu_propagator.cu
    src/propagator/multi_gpu_propagator.cu
    src/propagator/work_distribution.cu
    src/session/session.cu
    src/session/session_manager.cu
    src/session/resource_scheduler.cu
)

add_library(cudaprob3 ${SOURCES})

target_include_directories(cudaprob3
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudaprob3
    PUBLIC
        CUDA::cudart
        CUDA::cublas
    PRIVATE
        CUDA::curand
)

set_target_properties(cudaprob3 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

if(CUDA_PROB3_BUILD_BENCHMARKS)
    add_executable(bench_matrix_multiply tests/benchmarks/bench_matrix_multiply.cu)
    target_link_libraries(bench_matrix_multiply PRIVATE cudaprob3)
    add_test(NAME bench_matrix_multiply COMMAND bench_matrix_multiply)
endif()

if(CUDA_PROB3_BUILD_TESTS)
    enable_testing()
    add_executable(test_matrix3x3 tests/unit/test_matrix3x3.cu)
    target_link_libraries(test_matrix3x3 PRIVATE cudaprob3)
    add_test(NAME test_matrix3x3 COMMAND test_matrix3x3)

    add_executable(test_mixing tests/unit/test_mixing.cu)
    target_link_libraries(test_mixing PRIVATE cudaprob3)
    add_test(NAME test_mixing COMMAND test_mixing)

    add_executable(test_matter tests/unit/test_matter.cu)
    target_link_libraries(test_matter PRIVATE cudaprob3)
    add_test(NAME test_matter COMMAND test_matter)
endif()

if(CUDA_PROB3_BUILD_EXAMPLES)
    add_executable(example_benchmark examples/benchmark.cu)
    target_link_libraries(example_benchmark PRIVATE cudaprob3)

    add_executable(compare_new tests/comparison/compare_new.cu)
    target_link_libraries(compare_new PRIVATE cudaprob3)
endif()
