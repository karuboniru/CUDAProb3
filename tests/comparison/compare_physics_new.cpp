// Standalone NEW physics comparison — host only
// Outputs intermediate physics values for a single trajectory point
#include <iostream>
#include <iomanip>
#include <cmath>

#include "cudaprob3/physics/mixing.hpp"
#include "cudaprob3/physics/matter.hpp"
#include "cudaprob3/physics/amplitudes.hpp"
#include "cudaprob3/core/matrix3x3.hpp"

int main() {
    const double theta12 = 0.5839, theta13 = 0.1484, theta23 = 0.7385, dCP = 3.9095;
    const double dm12sq = 7.42e-5, dm23sq = 2.517e-3;
    double Enu = 0.1, rho = 3.0, Len = 100.0;

    cudaprob3::MixingMatrix mix(cudaprob3::MixingParams(theta12, theta13, theta23, dCP));
    cudaprob3::MassOrdering massOrd;
    massOrd.setMassDifferences(dm12sq, dm23sq);
    cudaprob3::TransitionAmplitude amp(mix, massOrd);

    cudaprob3::Matrix3x3<double> A_new;
    amp.compute(Len, Enu, rho, cudaprob3::NeutrinoType::Neutrino, A_new, 0.0);

    std::cout.precision(15);
    std::cout << std::scientific;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            std::cout << A_new(i,j).re << " " << A_new(i,j).im << std::endl;
    return 0;
}
