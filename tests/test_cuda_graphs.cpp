#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../include/cudaprob3/cudaprob3.hpp"
#include "../src/calculators/single_gpu_calculator.cuh"

#include <vector>
#include <cmath>

using namespace cudaprob3;

#ifndef MODELS_DIR
#define MODELS_DIR "."
#endif

static std::vector<double> linspace(double lo, double hi, int n) {
    std::vector<double> v(n);
    const double step = (hi - lo) / (n - 1);
    for (int i = 0; i < n - 1; ++i) v[i] = lo + i * step;
    v[n-1] = hi; return v;
}
static std::vector<double> logspace(double lo, double hi, int n) {
    std::vector<double> v(n);
    const double lolo = std::log(lo), lohi = std::log(hi);
    const double step = (lohi - lolo) / (n - 1);
    v[0] = lo; v[n-1] = hi;
    for (int i = 1; i < n-1; ++i) v[i] = std::exp(lolo + i*step);
    return v;
}

TEST_CASE("CUDA Graph replay produces bit-identical results to no-graph", "[gpu][graph]") {
    auto model1 = PREMModel::fromFile(std::string(MODELS_DIR) + "/PREM_12layer.dat");
    auto model2 = PREMModel::fromFile(std::string(MODELS_DIR) + "/PREM_12layer.dat");
    REQUIRE(model1.has_value()); REQUIRE(model2.has_value());

    const int nC = 100, nE = 100;
    auto cosVec = linspace(-1.0, 0.0, nC);
    auto eVec   = logspace(1.0, 100.0, nE);
    auto grid   = std::make_shared<ArbitraryGrid>(cosVec, eVec, 22.0);

    OscillationParams params(0.56959, 0.16088, 0.78540, 0.0, 7.9e-5, 2.5e-3);

    // No-graph baseline
    SingleGPUCalculator::Config cfgPlain;
    cfgPlain.deviceId = 0; cfgPlain.useCUDAGraphs = false;
    auto plain = SingleGPUCalculator::create(cfgPlain, grid,
        std::make_shared<PREMModel>(std::move(*model1)));
    REQUIRE(plain.has_value());
    auto baseline = plain->calculate(params, NeutrinoType::Neutrino);
    REQUIRE(baseline.has_value());

    // Graph-enabled calculator
    SingleGPUCalculator::Config cfgGraph;
    cfgGraph.deviceId = 0; cfgGraph.useCUDAGraphs = true;
    auto withGraph = SingleGPUCalculator::create(cfgGraph, grid,
        std::make_shared<PREMModel>(std::move(*model2)));
    REQUIRE(withGraph.has_value());

    // Run 3 times — first captures the graph, subsequent replay it
    for (int run = 0; run < 3; ++run) {
        auto res = withGraph->calculate(params, NeutrinoType::Neutrino);
        REQUIRE(res.has_value());

        // Results must match baseline exactly (same FP ops, same device)
        int failCount = 0;
        static const ProbType types[] = {
            ProbType::e_e, ProbType::e_m, ProbType::e_t,
            ProbType::m_e, ProbType::m_m, ProbType::m_t,
            ProbType::t_e, ProbType::t_m, ProbType::t_t
        };
        for (int fi = 0; fi < 9; ++fi)
            for (int ic = 0; ic < nC; ++ic)
                for (int ie = 0; ie < nE; ++ie) {
                    double got = res->probability(ic, ie, types[fi]);
                    double exp = baseline->probability(ic, ie, types[fi]);
                    if (got != exp) ++failCount;  // bit-identical
                }
        CAPTURE(run);
        CHECK(failCount == 0);
    }
}
