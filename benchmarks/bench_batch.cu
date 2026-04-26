// Benchmark: BatchCalculator throughput vs. batch size B.
#include "../include/cudaprob3/cudaprob3.hpp"
#include "../src/calculators/batch_calculator.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <span>

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

int main() {
    static constexpr int kNCos = 200, kNE = 200;
    auto grid  = std::make_shared<ArbitraryGrid>(
        linspace(-1.0, 0.0, kNCos), logspace(1.0, 100.0, kNE), 22.0);
    auto model = std::make_shared<PREMModel>(
        PREMModel::fromFile(MODELS_DIR "/PREM_12layer.dat").value());

    OscillationParams refParams(0.5696, 0.1609, 0.7854, 0.0, 7.9e-5, 2.5e-3);

    std::printf("%-8s  %10s  %12s\n", "B", "total_ms", "ms_per_set");
    for (int B : {1, 4, 16, 64, 256}) {
        std::vector<OscillationParams> pVec(B, refParams);
        std::vector<const OscillationParams*> pPtrs;
        for (auto& p : pVec) pPtrs.push_back(&p);

        auto batch = BatchCalculator::create({0}, grid, model, std::min(B, 64)).value();

        // Warmup
        batch.calculate(std::span{pPtrs}, NeutrinoType::Neutrino);

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        constexpr int iters = 5;
        for (int it = 0; it < iters; ++it)
            batch.calculate(std::span{pPtrs}, NeutrinoType::Neutrino);
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);

        float ms = 0; cudaEventElapsedTime(&ms, t0, t1);
        const double total_ms = ms / iters;
        cudaEventDestroy(t0); cudaEventDestroy(t1);
        std::printf("%-8d  %10.2f  %12.4f\n", B, total_ms, total_ms / B);
    }
}
