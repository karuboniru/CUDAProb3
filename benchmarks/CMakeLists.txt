add_executable(bench bench.cpp)
target_link_libraries(bench PRIVATE cudaprob3)
target_compile_definitions(bench PRIVATE
    MODELS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../models")
