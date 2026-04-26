#include <cudaprob3/cudaprob3.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

using namespace cudaprob3;
using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

#ifndef MODELS_DIR
#define MODELS_DIR "."
#endif

static std::vector<double> linspace(double lo, double hi, int n) {
    std::vector<double> v(n);
    const double step = (hi - lo) / (n - 1);
    for (int i = 0; i < n - 1; ++i) v[i] = lo + i * step;
    v[n - 1] = hi;
    return v;
}
static std::vector<double> logspace(double lo, double hi, int n) {
    std::vector<double> v(n);
    const double lolo = std::log(lo), lohi = std::log(hi);
    const double step = (lohi - lolo) / (n - 1);
    v[0] = lo; v[n - 1] = hi;
    for (int i = 1; i < n - 1; ++i) v[i] = std::exp(lolo + i * step);
    return v;
}

// Run fn() until at least target_s wall seconds have elapsed.
// Returns (iterations_run, total_wall_seconds).
template <typename Fn>
static std::pair<int, double> run_for(Fn&& fn, double target_s) {
    // calibration: time a single call to set the initial iteration count
    auto t0 = Clock::now();
    fn();
    double one_s = Sec(Clock::now() - t0).count();
    int batch = std::max(1, static_cast<int>(target_s / std::max(one_s, 1e-6)));
    batch = std::min(batch, 10000);

    int total = 0;
    double wall = 0.0;
    while (wall < target_s) {
        t0 = Clock::now();
        for (int i = 0; i < batch; ++i) fn();
        wall  += Sec(Clock::now() - t0).count();
        total += batch;
    }
    return {total, wall};
}

static void print_header() {
    std::printf("\n%-32s  %10s  %10s  %6s  %8s\n",
                "mode", "calls/s", "ms/call", "calls", "wall s");
    std::printf("%s\n", std::string(74, '-').c_str());
}

static void print_row(const char* label, int calls, double wall_s) {
    double cps    = calls / wall_s;
    double ms_per = wall_s / calls * 1e3;
    std::printf("%-32s  %10.1f  %10.3f  %6d  %8.3f\n",
                label, cps, ms_per, calls, wall_s);
}

// ─── Single GPU ───────────────────────────────────────────────────────────────
static void bench_single(int nCos, int nE,
                         std::shared_ptr<ArbitraryGrid> grid,
                         std::shared_ptr<PREMModel>     model) {
    OscillationParams params(0.5696, 0.1609, 0.7854, 0.0, 7.9e-5, 2.5e-3);

    // Without CUDA Graphs
    {
        SingleGPUCalculator::Config cfg; cfg.deviceId = 0; cfg.useCUDAGraphs = false;
        auto calc = SingleGPUCalculator::create(cfg, grid, model).value();
        calc.calculate(params, NeutrinoType::Neutrino); // warmup
        auto [calls, wall] = run_for(
            [&]{ calc.calculate(params, NeutrinoType::Neutrino); }, 1.0);
        char label[64];
        std::snprintf(label, sizeof(label), "single GPU %dx%d", nCos, nE);
        print_row(label, calls, wall);
    }

    // With CUDA Graphs
    {
        SingleGPUCalculator::Config cfg; cfg.deviceId = 0; cfg.useCUDAGraphs = true;
        auto calc = SingleGPUCalculator::create(cfg, grid, model).value();
        calc.calculate(params, NeutrinoType::Neutrino); // capture
        calc.calculate(params, NeutrinoType::Neutrino); // warmup replay
        auto [calls, wall] = run_for(
            [&]{ calc.calculate(params, NeutrinoType::Neutrino); }, 1.0);
        char label[64];
        std::snprintf(label, sizeof(label), "single GPU %dx%d (graphs)", nCos, nE);
        print_row(label, calls, wall);
    }
}

// ─── Batch ────────────────────────────────────────────────────────────────────
static void bench_batch(int nCos, int nE, int B,
                        std::shared_ptr<ArbitraryGrid> grid,
                        std::shared_ptr<PREMModel>     model) {
    OscillationParams ref(0.5696, 0.1609, 0.7854, 0.0, 7.9e-5, 2.5e-3);
    std::vector<OscillationParams> pVec(B, ref);
    std::vector<const OscillationParams*> ptrs;
    for (auto& p : pVec) ptrs.push_back(&p);

    auto batch = BatchCalculator::create({0}, grid, model, std::min(B, 64)).value();
    batch.calculate(std::span{ptrs}, NeutrinoType::Neutrino); // warmup

    auto [launches, wall] = run_for(
        [&]{ [[maybe_unused]] auto r = batch.calculate(std::span{ptrs}, NeutrinoType::Neutrino); }, 1.0);
    int calls = launches * B;

    char label[64];
    std::snprintf(label, sizeof(label), "batch B=%-4d %dx%d", B, nCos, nE);
    print_row(label, calls, wall);
}

int main() {
    const std::string modelPath = MODELS_DIR "/PREM_12layer.dat";
    auto model = std::make_shared<PREMModel>(
        PREMModel::fromFile(modelPath).value());

    std::printf("CUDAProb3 benchmark — %s\n", modelPath.c_str());
    std::printf("A 'call' = one full (nCos × nEnergy) grid for one PMNS set.\n");

    print_header();

    // Single GPU at several grid sizes
    for (int n : {100, 200, 400}) {
        auto grid = std::make_shared<ArbitraryGrid>(
            linspace(-1.0, 0.0, n), logspace(1.0, 100.0, n), 22.0);
        bench_single(n, n, grid, model);
    }

    std::printf("\n");

    // Batch at 200×200, varying batch size
    {
        auto grid = std::make_shared<ArbitraryGrid>(
            linspace(-1.0, 0.0, 200), logspace(1.0, 100.0, 200), 22.0);
        for (int B : {1, 4, 16, 64, 256})
            bench_batch(200, 200, B, grid, model);
    }

    std::printf("\n");
    return 0;
}
