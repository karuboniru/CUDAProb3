cmake_minimum_required(VERSION 3.25)
project(cudaprob3 VERSION 2.0.0 LANGUAGES CXX CUDA)

# --------------------------------------------------------------------------
# Language standards
# --------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD  23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# Dependencies — Thrust and CUB are bundled with CUDA toolkit via CCCL
# --------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# CCCL (Thrust + CUB) are bundled with CUDA 12+ at:
#   ${CUDAToolkit_INCLUDE_DIRS}/../cccl
# Expose that include directory to the library.
# (find_package(Thrust) is not needed — we include via -I and nvcc already knows it)

# --------------------------------------------------------------------------
# Library target
# --------------------------------------------------------------------------
add_library(cudaprob3 STATIC
    src/kernels/oscillation_kernel.cu              # CUDA kernel (nvcc)
    src/calculators/single_gpu_calculator.cu       # CUDA resource init/launch (nvcc)
    src/calculators/single_gpu_calculator_api.cpp  # std::expected API (GCC C++23)
    src/calculators/multi_gpu_calculator.cpp       # Multi-GPU logic (GCC C++23)
    src/calculators/batch_calculator.cu            # Batch launch with Thrust (nvcc)
)

target_include_directories(cudaprob3
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cudaprob3
    PUBLIC  CUDA::cudart stdc++ m
    PRIVATE CUDA::cublas
)

set_target_properties(cudaprob3 PROPERTIES
    CUDA_SEPARABLE_COMPILATION    ON
    CUDA_ARCHITECTURES            "80;86;89;90"  # sm_70 dropped in CUDA 13.x
    POSITION_INDEPENDENT_CODE     ON
)

target_compile_options(cudaprob3 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -lineinfo
        --ptxas-options=-v
    >
)

# --------------------------------------------------------------------------
# Tests  (opt-in: cmake -DCUDAPROB3_BUILD_TESTS=ON)
# --------------------------------------------------------------------------
option(CUDAPROB3_BUILD_TESTS "Build test suite" OFF)
if(CUDAPROB3_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.7.1
        GIT_SHALLOW    ON)
    FetchContent_MakeAvailable(Catch2)
    include(CTest)
    add_subdirectory(tests)
endif()

# --------------------------------------------------------------------------
# Example  (opt-in: cmake -DCUDAPROB3_BUILD_EXAMPLES=ON)
# --------------------------------------------------------------------------
option(CUDAPROB3_BUILD_EXAMPLES "Build example programs" OFF)
if(CUDAPROB3_BUILD_EXAMPLES)
    add_executable(cudaprob3_example example/main.cpp)
    target_link_libraries(cudaprob3_example PRIVATE cudaprob3)
endif()

# --------------------------------------------------------------------------
# Benchmarks  (opt-in: cmake -DCUDAPROB3_BUILD_BENCHMARKS=ON)
# --------------------------------------------------------------------------
option(CUDAPROB3_BUILD_BENCHMARKS "Build benchmarks" OFF)
if(CUDAPROB3_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# --------------------------------------------------------------------------
# Install
# --------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS cudaprob3
    EXPORT  cudaprob3Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT cudaprob3Targets
    FILE      cudaprob3Config.cmake
    NAMESPACE cudaprob3::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cudaprob3)
