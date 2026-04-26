#include <cudaprob3/cudaprob3.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace cudaprob3;

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

// ─── Example 1: single GPU, one PMNS set ─────────────────────────────────────
static void example_single_gpu(const std::string& modelPath) {
    std::cout << "=== Single GPU ===\n";

    auto model = PREMModel::fromFile(modelPath);
    if (!model) { std::cerr << model.error() << '\n'; return; }

    auto cosVec = linspace(-1.0, 0.0, 200);
    auto eVec   = logspace(1.0, 100.0, 200);
    auto grid   = std::make_shared<ArbitraryGrid>(cosVec, eVec, /*prodHeightKm=*/22.0);

    SingleGPUCalculator::Config cfg;
    cfg.deviceId      = 0;
    cfg.useCUDAGraphs = true;   // graph replay after first call
    auto calc = SingleGPUCalculator::create(cfg, grid,
                    std::make_shared<PREMModel>(std::move(*model)));
    if (!calc) { std::cerr << calc.error() << '\n'; return; }

    OscillationParams params(
        /*theta12=*/ 0.5695951908800630,
        /*theta13=*/ 0.1608752771983211,
        /*theta23=*/ 0.7853981633974483,
        /*dcp=*/     0.0,
        /*dm12sq=*/  7.9e-5,
        /*dm23sq=*/  2.5e-3);

    auto result = calc->calculate(params, NeutrinoType::Neutrino);
    if (!result) { std::cerr << result.error() << '\n'; return; }

    // Sample a few values
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "P(nu_mu -> nu_e)   icos=100 ie=100: "
              << result->probability(100, 100, ProbType::m_e) << '\n';
    std::cout << "P(nu_mu -> nu_mu)  icos=100 ie=100: "
              << result->probability(100, 100, ProbType::m_m) << '\n';
    std::cout << "P(nu_mu -> nu_tau) icos=100 ie=100: "
              << result->probability(100, 100, ProbType::m_t) << '\n';
}

// ─── Example 2: batch mode — sweep over 8 PMNS sets in one launch ────────────
static void example_batch(const std::string& modelPath) {
    std::cout << "\n=== Batch (8 PMNS sets) ===\n";

    auto model = PREMModel::fromFile(modelPath);
    if (!model) { std::cerr << model.error() << '\n'; return; }

    auto cosVec = linspace(-1.0, 0.0, 100);
    auto eVec   = logspace(1.0, 100.0, 100);
    auto grid   = std::make_shared<ArbitraryGrid>(cosVec, eVec, 22.0);

    auto batch = BatchCalculator::create({0}, grid,
                     std::make_shared<PREMModel>(std::move(*model)), /*chunkSize=*/8);
    if (!batch) { std::cerr << batch.error() << '\n'; return; }

    // Build 8 parameter sets varying theta23
    constexpr int B = 8;
    std::vector<OscillationParams> pVec;
    for (int i = 0; i < B; ++i) {
        const double th23 = 0.6 + i * 0.05;
        pVec.emplace_back(0.5696, 0.1609, th23, 0.0, 7.9e-5, 2.5e-3);
    }
    std::vector<const OscillationParams*> ptrs;
    for (auto& p : pVec) ptrs.push_back(&p);

    auto results = batch->calculate(std::span{ptrs}, NeutrinoType::Neutrino);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "P(nu_mu -> nu_mu) at icos=50 ie=50 for each theta23:\n";
    for (int b = 0; b < B; ++b) {
        std::cout << "  theta23=" << std::setprecision(3) << (0.6 + b * 0.05)
                  << "  P=" << std::setprecision(6)
                  << results[b].probability(50, 50, ProbType::m_m) << '\n';
    }
}

// ─── Example 3: multi-GPU ─────────────────────────────────────────────────────
static void example_multi_gpu(const std::string& modelPath) {
    std::cout << "\n=== Multi-GPU (GPUs {0}) ===\n";

    auto model = PREMModel::fromFile(modelPath);
    if (!model) { std::cerr << model.error() << '\n'; return; }

    auto cosVec = linspace(-1.0, 0.0, 200);
    auto eVec   = logspace(1.0, 100.0, 200);
    auto grid   = std::make_shared<ArbitraryGrid>(cosVec, eVec, 22.0);

    auto calc = MultiGPUCalculator::create(
        {0},   // add more device IDs for true multi-GPU, e.g. {0, 1}
        grid, std::make_shared<PREMModel>(std::move(*model)));
    if (!calc) { std::cerr << calc.error() << '\n'; return; }

    OscillationParams params(0.5696, 0.1609, 0.7854, 0.0, 7.9e-5, 2.5e-3);
    auto result = calc->calculate(params, NeutrinoType::Neutrino);
    if (!result) { std::cerr << result.error() << '\n'; return; }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "P(nu_e -> nu_e) icos=0 ie=0: "
              << result->probability(0, 0, ProbType::e_e) << '\n';
}

int main(int argc, char** argv) {
    const std::string modelPath = (argc > 1)
        ? std::string(argv[1]) + "/PREM_12layer.dat"
        : "../models/PREM_12layer.dat";

    example_single_gpu(modelPath);
    example_batch(modelPath);
    example_multi_gpu(modelPath);
    return 0;
}
