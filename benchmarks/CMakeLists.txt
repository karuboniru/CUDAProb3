add_executable(bench_single bench_single.cu)
target_link_libraries(bench_single PRIVATE cudaprob3)
set_target_properties(bench_single PROPERTIES CUDA_ARCHITECTURES "70;80;86;89;90")
target_compile_definitions(bench_single PRIVATE
    MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../example/models")

add_executable(bench_batch bench_batch.cu)
target_link_libraries(bench_batch PRIVATE cudaprob3)
set_target_properties(bench_batch PROPERTIES CUDA_ARCHITECTURES "70;80;86;89;90")
target_compile_definitions(bench_batch PRIVATE
    MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../example/models")
