#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>

#include "cudaprob3/core/matrix3x3.hpp"
#include "cudaprob3/core/cuda_helpers.hpp"

static void print_header() {
    std::cout << std::left
              << std::setw(14) << "Batch Size"
              << std::setw(18) << "Register (ms)"
              << std::setw(14) << "Reg GFLOPS"
              << std::setw(18) << "cuBLAS (ms)"
              << std::setw(14) << "cuBLAS GFLOPS"
              << std::setw(12) << "Winner"
              << std::setw(10) << "Speedup"
              << std::endl;
    std::cout << std::string(100, '-') << std::endl;
}

static constexpr double flops_per_mult = 9.0 * (3.0 * 6.0 + 3.0 * 2.0);

static void print_row(size_t batch, const cudaprob3::MatrixMultiplyBenchmark& reg,
                       const cudaprob3::MatrixMultiplyBenchmark& blas) {
    std::string winner = (reg.elapsed_ms <= blas.elapsed_ms) ? "Register" : "cuBLAS";
    double speedup = (reg.elapsed_ms <= blas.elapsed_ms)
                     ? blas.elapsed_ms / reg.elapsed_ms
                     : reg.elapsed_ms / blas.elapsed_ms;

    std::cout << std::left
              << std::setw(14) << batch
              << std::fixed << std::setprecision(4)
              << std::setw(18) << reg.elapsed_ms
              << std::setw(14) << reg.gflops
              << std::setw(18) << blas.elapsed_ms
              << std::setw(14) << blas.gflops
              << std::setw(12) << winner
              << std::setprecision(2)
              << std::setw(10) << speedup << "x"
              << std::endl;
}

static size_t round_up(size_t n, size_t multiple) {
    return ((n + multiple - 1) / multiple) * multiple;
}

int main() {
    const int deviceId = 0;
    cuda_check_device(deviceId);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    std::cout << "GPU: " << prop.name
              << " (SM " << prop.major << "." << prop.minor << ")\n" << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Benchmark across a range of batch sizes
    // Small batches (1K-16K): typical for few (cos, E) pairs
    // Medium batches (32K-128K): typical for realistic grids (200x200=40K)
    // Large batches (256K-1M): stress test
    std::vector<size_t> batch_sizes = {
        512, 1024, 2048, 4096, 8192,
        16384, 32768, 65536, 131072, 262144, 524288
    };

    print_header();

    std::vector<std::pair<size_t, cudaprob3::MatrixMultiplyStrategy>> recommendations;

    for (auto batch : batch_sizes) {
        cudaprob3::MatrixMultiplyEngine reg(deviceId, cudaprob3::MatrixMultiplyStrategy::RegisterKernel);
        cudaprob3::MatrixMultiplyEngine blas(deviceId, cudaprob3::MatrixMultiplyStrategy::CuBlasBatched);

        auto bm_reg  = reg.warmup_and_benchmark(batch, 3, 10, stream);
        auto bm_blas = blas.warmup_and_benchmark(batch, 3, 10, stream);

        print_row(batch, bm_reg, bm_blas);

        recommendations.emplace_back(batch,
            (bm_reg.elapsed_ms <= bm_blas.elapsed_ms)
                ? cudaprob3::MatrixMultiplyStrategy::RegisterKernel
                : cudaprob3::MatrixMultiplyStrategy::CuBlasBatched);
    }

    std::cout << "\nRECOMMENDATION:" << std::endl;
    for (auto& [size, strat] : recommendations) {
        std::cout << "  batch=" << std::setw(8) << size
                  << " → " << cudaprob3::strategy_name(strat) << std::endl;
    }

    // Now do a targeted test at a typical grid size
    std::cout << "\n=== TYPICAL USE-CASE BENCHMARK ===" << std::endl;
    std::cout << "Grid sizes: 100×100=10K, 200×200=40K, 500×500=250K, 1000×100=100K\n" << std::endl;

    std::vector<size_t> typical_batches = {10000, 40000, 100000, 250000};
    for (auto batch : typical_batches) {
        cudaprob3::MatrixMultiplyEngine reg(deviceId, cudaprob3::MatrixMultiplyStrategy::RegisterKernel);
        cudaprob3::MatrixMultiplyEngine blas(deviceId, cudaprob3::MatrixMultiplyStrategy::CuBlasBatched);

        auto bm_reg  = reg.warmup_and_benchmark(batch, 2, 20, stream);
        auto bm_blas = blas.warmup_and_benchmark(batch, 2, 20, stream);

        std::cout << "Batch " << batch << ": "
                  << "Register=" << bm_reg.elapsed_ms << "ms "
                  << "cuBLAS=" << bm_blas.elapsed_ms << "ms → "
                  << ((bm_reg.elapsed_ms <= bm_blas.elapsed_ms) ? "REGISTER wins" : "cuBLAS wins")
                  << std::endl;
    }

    cudaStreamDestroy(stream);
    return 0;
}
