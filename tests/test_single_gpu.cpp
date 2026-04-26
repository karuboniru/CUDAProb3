#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../include/cudaprob3/cudaprob3.hpp"

#include <array>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

using namespace cudaprob3;

// Reference parameters matching example/main.cpp
static constexpr double kTheta12 = 0.5695951908800630;
static constexpr double kTheta13 = 0.1608752771983211;
static constexpr double kTheta23 = 0.7853981633974483;
static constexpr double kDCP     = 0.0;
static constexpr double kDm12sq  = 7.9e-5;
static constexpr double kDm23sq  = 2.5e-3;
static constexpr int    kNCos    = 200;
static constexpr int    kNE      = 200;

#ifndef REFERENCE_DIR
#define REFERENCE_DIR "."
#endif
#ifndef MODELS_DIR
#define MODELS_DIR "."
#endif

static std::vector<double> loadReference(const std::string& filename) {
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Cannot open reference file: " + filename);
    int nc, ne;
    f >> nc >> ne;
    if (nc != kNCos || ne != kNE)
        throw std::runtime_error("Reference grid size mismatch: " + filename);
    std::vector<double> v;
    v.reserve(static_cast<std::size_t>(nc * ne));
    double x;
    while (f >> x) v.push_back(x);
    return v;
}

static std::array<std::vector<double>, 9> loadAllReference() {
    static const char* names[] = {
        "out_e_e.txt","out_e_m.txt","out_e_t.txt",
        "out_m_e.txt","out_m_m.txt","out_m_t.txt",
        "out_t_e.txt","out_t_m.txt","out_t_t.txt"
    };
    std::array<std::vector<double>, 9> refs;
    for (int i = 0; i < 9; ++i)
        refs[i] = loadReference(std::string(REFERENCE_DIR) + "/" + names[i]);
    return refs;
}

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

TEST_CASE("SingleGPUCalculator matches reference output", "[gpu][single]") {
    auto model = PREMModel::fromFile(std::string(MODELS_DIR) + "/PREM_12layer.dat");
    REQUIRE(model.has_value());

    auto cosVec = linspace(-1.0, 0.0, kNCos);
    auto eVec   = logspace(1.0, 100.0, kNE);

    auto grid = std::make_shared<ArbitraryGrid>(cosVec, eVec, 22.0);

    SingleGPUCalculator::Config cfg;
    cfg.deviceId = 0;
    auto calcOrErr = SingleGPUCalculator::create(
        cfg, grid, std::make_shared<PREMModel>(std::move(*model)));
    REQUIRE(calcOrErr.has_value());
    auto& calc = *calcOrErr;

    OscillationParams params(kTheta12, kTheta13, kTheta23, kDCP, kDm12sq, kDm23sq);
    auto resOrErr = calc.calculate(params, NeutrinoType::Neutrino);
    REQUIRE(resOrErr.has_value());
    const auto& result = *resOrErr;

    // Load reference and compare
    auto refs = loadAllReference();

    static const ProbType kProbTypes[] = {
        ProbType::e_e, ProbType::e_m, ProbType::e_t,
        ProbType::m_e, ProbType::m_m, ProbType::m_t,
        ProbType::t_e, ProbType::t_m, ProbType::t_t
    };

    constexpr double kTol = 1e-10;
    int failCount = 0;
    for (int fi = 0; fi < 9; ++fi) {
        const auto& ref = refs[fi];
        for (int ic = 0; ic < kNCos; ++ic) {
            for (int ie = 0; ie < kNE; ++ie) {
                const double got = result.probability(ic, ie, kProbTypes[fi]);
                const double exp = ref[ic * kNE + ie];
                if (std::fabs(got - exp) > kTol) {
                    ++failCount;
                    if (failCount <= 5) {
                        WARN("flavor " << fi << " cos[" << ic << "] E[" << ie
                             << "]: got=" << got << " exp=" << exp
                             << " diff=" << std::fabs(got - exp));
                    }
                }
            }
        }
    }
    CHECK(failCount == 0);
}
