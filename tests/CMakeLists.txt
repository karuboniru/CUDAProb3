# Catch2 targets are provided by FetchContent in parent CMakeLists.txt
# Host-only tests (no GPU needed)
add_executable(test_math         test_math.cpp)
add_executable(test_physics_host test_physics_host.cpp)
add_executable(test_prem_model   test_prem_model.cpp)

foreach(t test_math test_physics_host test_prem_model)
    target_link_libraries(${t} PRIVATE Catch2::Catch2WithMain cudaprob3)
    target_compile_definitions(${t} PRIVATE CUDAPROB3_HOST_PHYSICS_TEST)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# GPU tests (require a GPU at test time)
# These files use only the high-level C++ API (no device code), compiled by GCC in C++23 mode.
foreach(t test_single_gpu test_multi_gpu test_batch test_cuda_graphs)
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t} PRIVATE Catch2::Catch2WithMain cudaprob3)
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# Path to reference data
target_compile_definitions(test_single_gpu PRIVATE
    REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/reference"
    MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../models")
target_compile_definitions(test_multi_gpu PRIVATE
    REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/reference"
    MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../models")
target_compile_definitions(test_batch PRIVATE
    REFERENCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/reference"
    MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../models")
target_compile_definitions(test_cuda_graphs PRIVATE
    MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../models")
