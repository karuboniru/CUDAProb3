#include <iostream>
#include <cassert>
#include <cmath>
#include "cudaprob3/physics/mixing.hpp"
#include "cudaprob3/physics/matter.hpp"
#include "cudaprob3/core/constants.hpp"

using namespace cudaprob3;

int main() {
    MixingMatrix mixing(MixingParams(0.5839, 0.1484, 0.7385, 3.9095));
    MassOrdering massOrdering;
    massOrdering.setMassDifferences(7.42e-5, 2.517e-3);

    MatterEffects matter(mixing, massOrdering);

    double dmMatMat[3][3], dmMatVac[3][3];
    matter.computeEffectiveMasses(1.0, 3.0, NeutrinoType::Neutrino, dmMatMat, dmMatVac);

    // Verify basic properties: dmMatMat is antisymmetric
    for (int i = 0; i < 3; ++i) {
        assert(std::fabs(dmMatMat[i][i]) < 1e-10);
        for (int j = 0; j < 3; ++j) {
            assert(std::fabs(dmMatMat[i][j] + dmMatMat[j][i]) < 1e-10);
        }
    }

    std::cout << "test_matter: PASSED" << std::endl;
    return 0;
}
