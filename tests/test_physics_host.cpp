#define CUDAPROB3_HOST_PHYSICS_TEST
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "../src/physics/barger.cuh"
#include "../include/cudaprob3/oscillation_params.hpp"

using namespace cudaprob3;

static OscParamsPOD makeRefPOD() {
    // Parameters matching example/main.cpp reference run
    OscillationParams p(
        0.5695951908800630, 0.1608752771983211, 0.7853981633974483,
        0.0, 7.9e-5, 2.5e-3);
    return p.computePOD();
}

TEST_CASE("OscillationParams::computePOD mass ordering sane", "[physics]") {
    auto pod = makeRefPOD();
    // ORDER values must be a permutation of {0,1,2}
    bool seen[3] = {false, false, false};
    for (int i = 0; i < 3; ++i) {
        REQUIRE(pod.order[i] >= 0);
        REQUIRE(pod.order[i] < 3);
        seen[pod.order[i]] = true;
    }
    REQUIRE(seen[0]); REQUIRE(seen[1]); REQUIRE(seen[2]);
}

TEST_CASE("getMfast vacuum limit: fac=0 recovers vacuum masses", "[physics]") {
    auto pod = makeRefPOD();

    // rho=0 → fac=0 → matter masses equal vacuum masses
    double d_dmMatMat[3][3], d_dmMatVac[3][3];
    physics::getMfast(1.0, 0.0, NeutrinoType::Neutrino, pod, d_dmMatMat, d_dmMatVac);

    // d_dmMatVac[i][i] = mMat[i] - DM(i,0)
    // In vacuum mMat[i] = DM(i,0), so d_dmMatVac[i][i] ≈ 0
    for (int i = 0; i < 3; ++i)
        REQUIRE(std::fabs(d_dmMatVac[i][i]) < 1e-10);
}

TEST_CASE("get_transition_matrix vacuum: identity for L=0", "[physics]") {
    auto pod = makeRefPOD();
    math::Complex3x3<double> A{};
    physics::get_transition_matrix(NeutrinoType::Neutrino, 1.0, 0.0, 0.0, pod, A);

    // L=0 → argument of exp is 0 → identity matrix (modulo phase-factor details)
    // The diagonal elements should have magnitude 1, off-diagonal ~0
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double mag2 = A.re[i*3+j]*A.re[i*3+j] + A.im[i*3+j]*A.im[i*3+j];
            if (i == j)
                REQUIRE(mag2 == Catch::Approx(1.0).epsilon(1e-10));
            else
                REQUIRE(mag2 < 1e-10);
        }
}

TEST_CASE("get_transition_matrix: unitarity |A†A - I| < epsilon", "[physics]") {
    auto pod = makeRefPOD();
    math::Complex3x3<double> A{};
    // A typical segment: 500 km, 3 g/cm³, 5 GeV
    physics::get_transition_matrix(NeutrinoType::Neutrino, 5.0,
                                    3.0 * 0.5, 500.0, pod, A);

    // Check A†A ≈ I
    auto Adagger_A = A.operator*(A);  // not quite - we need A† * A
    // Compute A† manually: (A†)[i][j] = conj(A[j][i])
    math::Complex3x3<double> Ad{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            Ad.re[i*3+j] =  A.re[j*3+i];
            Ad.im[i*3+j] = -A.im[j*3+i];
        }
    auto AdA = Ad * A;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            if (i == j) {
                REQUIRE(AdA.re[i*3+j] == Catch::Approx(1.0).epsilon(1e-10));
            } else {
                REQUIRE(std::fabs(AdA.re[i*3+j]) < 1e-13);
            }
            REQUIRE(std::fabs(AdA.im[i*3+j]) < 1e-13);
        }
}

TEST_CASE("Probability sum = 1 for a simple path", "[physics]") {
    auto pod = makeRefPOD();
    math::Complex3x3<double> A{};
    physics::get_transition_matrix(NeutrinoType::Neutrino, 2.0,
                                    4.5 * 0.5, 1200.0, pod, A);

    // P(initial=mu → anything) should sum to 1
    for (int inflv = 0; inflv < 3; ++inflv) {
        double sum = 0;
        for (int outflv = 0; outflv < 3; ++outflv) {
            double r = A.re[outflv*3+inflv];
            double im = A.im[outflv*3+inflv];
            sum += r*r + im*im;
        }
        REQUIRE(sum == Catch::Approx(1.0).epsilon(1e-10));
    }
}
