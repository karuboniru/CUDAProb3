#define CUDAPROB3_HOST_PHYSICS_TEST
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cmath>
#include "../src/math/complex3x3.cuh"

using namespace cudaprob3::math;

TEST_CASE("Complex3x3 identity", "[math]") {
    auto I = Complex3x3<double>::identity();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            REQUIRE(I.re[i*3+j] == Catch::Approx(i==j ? 1.0 : 0.0));
            REQUIRE(I.im[i*3+j] == Catch::Approx(0.0));
        }
}

TEST_CASE("Identity is multiplicative identity", "[math]") {
    Complex3x3<double> A{};
    // Fill A with some complex values
    for (int k = 0; k < 9; ++k) {
        A.re[k] = static_cast<double>(k + 1);
        A.im[k] = static_cast<double>(k) * 0.1;
    }
    auto I = Complex3x3<double>::identity();
    auto AI = A * I;
    auto IA = I * A;
    for (int k = 0; k < 9; ++k) {
        REQUIRE(AI.re[k] == Catch::Approx(A.re[k]).epsilon(1e-14));
        REQUIRE(AI.im[k] == Catch::Approx(A.im[k]).epsilon(1e-14));
        REQUIRE(IA.re[k] == Catch::Approx(A.re[k]).epsilon(1e-14));
        REQUIRE(IA.im[k] == Catch::Approx(A.im[k]).epsilon(1e-14));
    }
}

TEST_CASE("Matrix multiply associativity (A*B)*C == A*(B*C)", "[math]") {
    Complex3x3<double> A{}, B{}, C{};
    for (int k = 0; k < 9; ++k) {
        A.re[k] = std::sin(static_cast<double>(k+1));
        A.im[k] = std::cos(static_cast<double>(k+2));
        B.re[k] = std::cos(static_cast<double>(k+3));
        B.im[k] = std::sin(static_cast<double>(k+4));
        C.re[k] = std::sin(static_cast<double>(k+5)) * 0.5;
        C.im[k] = std::cos(static_cast<double>(k+6)) * 0.5;
    }

    auto AB_C = (A * B) * C;
    auto A_BC = A * (B * C);

    for (int k = 0; k < 9; ++k) {
        REQUIRE(AB_C.re[k] == Catch::Approx(A_BC.re[k]).epsilon(1e-12));
        REQUIRE(AB_C.im[k] == Catch::Approx(A_BC.im[k]).epsilon(1e-12));
    }
}

TEST_CASE("ct_sqr and ct_cube", "[math]") {
    REQUIRE(ct_sqr(3.0)  == Catch::Approx(9.0));
    REQUIRE(ct_cube(2.0) == Catch::Approx(8.0));
}
