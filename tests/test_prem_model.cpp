#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "../include/cudaprob3/density_model.hpp"

using namespace cudaprob3;

TEST_CASE("PREMModel from ascending vectors", "[density]") {
    // PREM-like: 3 shells in ascending order (inner to outer)
    auto m = PREMModel::fromVectors({0.0, 3000.0, 6371.0}, {13.0, 4.5, 3.3});
    REQUIRE(m.has_value());
    // Should be reversed to descending (outermost first)
    REQUIRE(m->radii()[0] == Catch::Approx(6371.0));
    REQUIRE(m->radii()[2] == Catch::Approx(0.0));
}

TEST_CASE("PREMModel from descending vectors", "[density]") {
    auto m = PREMModel::fromVectors({6371.0, 3000.0, 0.0}, {3.3, 4.5, 13.0});
    REQUIRE(m.has_value());
    REQUIRE(m->radii()[0] == Catch::Approx(6371.0));
}

TEST_CASE("PREMModel size mismatch returns error", "[density]") {
    auto m = PREMModel::fromVectors({6371.0, 3000.0}, {3.3, 4.5, 13.0});
    REQUIRE(!m.has_value());
}

TEST_CASE("PREMModel non-monotonic returns error", "[density]") {
    auto m = PREMModel::fromVectors({6371.0, 5000.0, 6000.0}, {3.3, 4.5, 13.0});
    REQUIRE(!m.has_value());
}

TEST_CASE("maxLayersForCosine is 0 for downward neutrinos", "[density]") {
    auto m = PREMModel::fromVectors({6371.0, 3500.0, 1221.5, 0.0}, {3.3, 5.0, 10.0, 13.0});
    REQUIRE(m.has_value());
    // cosine = 0.5 → downward, doesn't enter Earth
    REQUIRE(m->maxLayersForCosine(0.5) == 0);
}

TEST_CASE("maxLayersForCosine increases for more vertical paths", "[density]") {
    auto m = PREMModel::fromVectors({6371.0, 3500.0, 1221.5, 0.0}, {3.3, 5.0, 10.0, 13.0});
    REQUIRE(m.has_value());
    const int layers_near_horiz = m->maxLayersForCosine(-0.1);
    const int layers_vertical   = m->maxLayersForCosine(-1.0);
    REQUIRE(layers_vertical >= layers_near_horiz);
}

#ifdef MODELS_DIR
TEST_CASE("PREMModel from file: PREM_12layer.dat", "[density][file]") {
    auto m = PREMModel::fromFile(MODELS_DIR "/PREM_12layer.dat");
    REQUIRE(m.has_value());
    REQUIRE(m->radii().size() == m->densities().size());
    REQUIRE(m->radii().size() > 0);
    // Outermost radius should be close to 6371 km
    REQUIRE(m->radii()[0] == Catch::Approx(6371.0).epsilon(0.01));
}
#endif
