#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../include/cudaprob3/cudaprob3.hpp"
#include "../src/calculators/multi_gpu_calculator.cuh"

#include <vector>
#include <cmath>

using namespace cudaprob3;

static constexpr double kTheta12 = 0.5695951908800630;
static constexpr double kTheta13 = 0.1608752771983211;
static constexpr double kTheta23 = 0.7853981633974483;
static constexpr double kDCP     = 0.0;
static constexpr double kDm12sq  = 7.9e-5;
static constexpr double kDm23sq  = 2.5e-3;
static constexpr int    kNCos    = 200;
static constexpr int    kNE      = 200;

#ifndef MODELS_DIR
#define MODELS_DIR "."
#endif

static std::vector<double> linspace(double lo, double hi, int n) {
    std::vector<double> v(n);
    const double step = (hi - lo) / (n - 1);
    for (int i = 0; i < n - 1; ++i) v[i] = lo + i * step;
    v[n-1] = hi;
    return v;
}

static std::vector<double> logspace(double lo, double hi, int n) {
    std::vector<double> v(n);
    const double lolo = std::log(lo), lohi = std::log(hi);
    const double step = (lohi - lolo) / (n - 1);
    v[0] = lo; v[n-1] = hi;
    for (int i = 1; i < n - 1; ++i) v[i] = std::exp(lolo + i * step);
    return v;
}

TEST_CASE("MultiGPUCalculator matches single GPU reference", "[gpu][multi]") {
    int nDev = 0;
    cudaGetDeviceCount(&nDev);
    if (nDev < 1) { SKIP("No GPU available"); return; }

    auto modelSingle = PREMModel::fromFile(std::string(MODELS_DIR) + "/PREM_12layer.dat");
    REQUIRE(modelSingle.has_value());
    auto modelMulti  = PREMModel::fromFile(std::string(MODELS_DIR) + "/PREM_12layer.dat");

    auto cosVec = linspace(-1.0, 0.0, kNCos);
    auto eVec   = logspace(1.0, 100.0, kNE);
    auto grid   = std::make_shared<ArbitraryGrid>(cosVec, eVec, 22.0);

    OscillationParams params(kTheta12, kTheta13, kTheta23, kDCP, kDm12sq, kDm23sq);

    // Single GPU reference
    SingleGPUCalculator::Config cfg;
    cfg.deviceId = 0;
    auto single = SingleGPUCalculator::create(cfg, grid, std::make_shared<PREMModel>(std::move(*modelSingle)));
    REQUIRE(single.has_value());
    auto refResult = single->calculate(params, NeutrinoType::Neutrino);
    REQUIRE(refResult.has_value());

    // Multi-GPU (even if only 1 GPU, the MultiGPUCalculator code path is exercised)
    std::vector<int> devIds;
    for (int i = 0; i < std::min(nDev, 2); ++i) devIds.push_back(i);

    auto multi = MultiGPUCalculator::create(devIds, grid, std::make_shared<PREMModel>(std::move(*modelMulti)));
    REQUIRE(multi.has_value());
    auto multiResult = multi->calculate(params, NeutrinoType::Neutrino);
    REQUIRE(multiResult.has_value());

    static const ProbType kProbTypes[] = {
        ProbType::e_e, ProbType::e_m, ProbType::e_t,
        ProbType::m_e, ProbType::m_m, ProbType::m_t,
        ProbType::t_e, ProbType::t_m, ProbType::t_t
    };

    constexpr double kTol = 1e-12;
    int failCount = 0;
    for (int fi = 0; fi < 9; ++fi) {
        for (int ic = 0; ic < kNCos; ++ic) {
            for (int ie = 0; ie < kNE; ++ie) {
                const double got = multiResult->probability(ic, ie, kProbTypes[fi]);
                const double exp = refResult->probability(ic, ie, kProbTypes[fi]);
                if (std::fabs(got - exp) > kTol) {
                    ++failCount;
                    if (failCount <= 3)
                        WARN("f=" << fi << " cos[" << ic << "] E[" << ie
                                  << "]: diff=" << std::fabs(got - exp));
                }
            }
        }
    }
    CHECK(failCount == 0);
}
