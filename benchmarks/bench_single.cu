// Benchmark: single-GPU calculate() throughput vs. grid size.
// Reports wall time (ms) and estimated GFLOPS.
#include "../include/cudaprob3/cudaprob3.hpp"
#include "../src/calculators/single_gpu_calculator.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace cudaprob3;

#ifndef MODELS_DIR
#define MODELS_DIR "."
#endif

static std::vector<double> linspace(double lo, double hi, int n) {
    std::vector<double> v(n);
    const double step = (hi - lo) / (n-1);
    for (int i=0;i<n-1;++i) v[i]=lo+i*step; v[n-1]=hi; return v;
}
static std::vector<double> logspace(double lo, double hi, int n) {
    std::vector<double> v(n);
    const double lolo=std::log(lo),lohi=std::log(hi),step=(lohi-lolo)/(n-1);
    v[0]=lo; v[n-1]=hi;
    for (int i=1;i<n-1;++i) v[i]=std::exp(lolo+i*step); return v;
}

static double benchmark(int nCos, int nE, int warmup=5, int iters=20) {
    auto model = PREMModel::fromFile(MODELS_DIR "/PREM_12layer.dat").value();
    auto grid  = std::make_shared<ArbitraryGrid>(
        linspace(-1.0, 0.0, nCos), logspace(1.0, 100.0, nE), 22.0);

    SingleGPUCalculator::Config cfg; cfg.deviceId = 0;
    auto calc = SingleGPUCalculator::create(cfg, grid,
        std::make_shared<PREMModel>(std::move(model))).value();

    OscillationParams params(0.5696, 0.1609, 0.7854, 0.0, 7.9e-5, 2.5e-3);

    // Warmup
    for (int i = 0; i < warmup; ++i)
        calc.calculate(params, NeutrinoType::Neutrino);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i)
        calc.calculate(params, NeutrinoType::Neutrino);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return static_cast<double>(ms) / iters;
}

int main() {
    std::printf("%-12s %-12s  %10s  %12s\n", "n_cosines", "n_energies",
                "mean_ms", "GFLOPS");
    for (int n : {50, 100, 200, 400}) {
        double ms = benchmark(n, n);
        // ~7000 flops per (cosine, energy) pair (rough estimate)
        double gflops = 7000.0 * n * n / (ms * 1e-3) / 1e9;
        std::printf("%-12d %-12d  %10.3f  %12.3f\n", n, n, ms, gflops);
    }
}
