#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../include/cudaprob3/cudaprob3.hpp"
#include "../src/calculators/batch_calculator.cuh"
#include "../src/calculators/single_gpu_calculator.cuh"

#include <array>
#include <vector>
#include <cmath>

using namespace cudaprob3;

static constexpr double kTheta12 = 0.5695951908800630;
static constexpr double kTheta13 = 0.1608752771983211;
static constexpr double kTheta23 = 0.7853981633974483;
static constexpr double kDCP     = 0.0;
static constexpr double kDm12sq  = 7.9e-5;
static constexpr double kDm23sq  = 2.5e-3;
static constexpr int    kNCos    = 50;   // smaller grid for speed
static constexpr int    kNE      = 50;
static constexpr int    kBatch   = 8;

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

TEST_CASE("BatchCalculator: B identical PMNS sets match single reference", "[gpu][batch]") {
    auto modelB = PREMModel::fromFile(std::string(MODELS_DIR) + "/PREM_12layer.dat");
    auto model1 = PREMModel::fromFile(std::string(MODELS_DIR) + "/PREM_12layer.dat");
    REQUIRE(modelB.has_value()); REQUIRE(model1.has_value());

    auto cosVec = linspace(-1.0, 0.0, kNCos);
    auto eVec   = logspace(1.0, 100.0, kNE);
    auto grid   = std::make_shared<ArbitraryGrid>(cosVec, eVec, 22.0);

    OscillationParams params(kTheta12, kTheta13, kTheta23, kDCP, kDm12sq, kDm23sq);

    // Single reference
    SingleGPUCalculator::Config cfg; cfg.deviceId = 0;
    auto single = SingleGPUCalculator::create(cfg, grid,
        std::make_shared<PREMModel>(std::move(*model1)));
    REQUIRE(single.has_value());
    auto refRes = single->calculate(params, NeutrinoType::Neutrino);
    REQUIRE(refRes.has_value());

    // Batch with B identical parameter sets
    auto batch = BatchCalculator::create({0}, grid,
        std::make_shared<PREMModel>(std::move(*modelB)), 4);
    REQUIRE(batch.has_value());

    std::vector<OscillationParams> pVec(kBatch, params);
    std::vector<const OscillationParams*> pPtrs;
    for (auto& p : pVec) pPtrs.push_back(&p);

    auto batchRes = batch->calculate(std::span{pPtrs}, NeutrinoType::Neutrino);
    REQUIRE(batchRes.size() == kBatch);

    static const ProbType kProbTypes[] = {
        ProbType::e_e, ProbType::e_m, ProbType::e_t,
        ProbType::m_e, ProbType::m_m, ProbType::m_t,
        ProbType::t_e, ProbType::t_m, ProbType::t_t
    };

    constexpr double kTol = 1e-10;
    for (int b = 0; b < kBatch; ++b) {
        const auto bv = batchRes[b];
        int failCount = 0;
        for (int fi = 0; fi < 9; ++fi)
            for (int ic = 0; ic < kNCos; ++ic)
                for (int ie = 0; ie < kNE; ++ie) {
                    const double got = bv.probability(ic, ie, kProbTypes[fi]);
                    const double exp = refRes->probability(ic, ie, kProbTypes[fi]);
                    if (std::fabs(got - exp) > kTol) ++failCount;
                }
        CHECK(failCount == 0);
    }
}
