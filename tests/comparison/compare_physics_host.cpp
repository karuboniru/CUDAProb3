// Host-only physics comparison — no GPU needed
// Compares old physics.hpp + math.hpp against new mixing/matter/amplitudes
// using exactly the same inputs

#include <iostream>
#include <iomanip>
#include <cmath>

// Include OLD headers (host-mode, no CUDA)
#define __CUDACC__  // needed for hpc_helpers
#include "../../constants.hpp"
#include "../../math.hpp"
#include "../../types.hpp"
#include "../../physics.hpp"

// Include NEW headers
#include "cudaprob3/physics/mixing.hpp"
#include "cudaprob3/physics/matter.hpp"
#include "cudaprob3/physics/amplitudes.hpp"
#include "cudaprob3/core/matrix3x3.hpp"
#include "cudaprob3/core/constants.hpp"

int main() {
    // Common parameters
    const double theta12 = 0.5839, theta13 = 0.1484, theta23 = 0.7385, dCP = 3.9095;
    const double dm12sq = 7.42e-5, dm23sq = 2.517e-3;
    const double Enu = 0.1, rho = 3.0, Len = 100.0;  // GeV, g/cm^3, km

    // ====== OLD (host path) ======
    cudaprob3::math::ComplexNumber<double> U_old[9];
    cudaprob3::math::ComplexNumber<double> A_old[3][3];
    double dm_old[9];

    // Build PMNS matrix same way as original
    {
        const double s12 = sin(theta12), s13 = sin(theta13), s23 = sin(theta23);
        const double c12 = cos(theta12), c13 = cos(theta13), c23 = cos(theta23);
        const double sd = sin(dCP), cd = cos(dCP);
        U_old[0].re = c12*c13; U_old[0].im = 0;
        U_old[1].re = s12*c13; U_old[1].im = 0;
        U_old[2].re = s13*cd;  U_old[2].im = -s13*sd;
        U_old[3].re = -s12*c23 - c12*s23*s13*cd; U_old[3].im = -c12*s23*s13*sd;
        U_old[4].re = c12*c23 - s12*s23*s13*cd;  U_old[4].im = -s12*s23*s13*sd;
        U_old[5].re = s23*c13; U_old[5].im = 0;
        U_old[6].re = s12*s23 - c12*c23*s13*cd;  U_old[6].im = -c12*c23*s13*sd;
        U_old[7].re = -c12*s23 - s12*c23*s13*cd; U_old[7].im = -s12*c23*s13*sd;
        U_old[8].re = c23*c13; U_old[8].im = 0;
    }

    // Build mass differences
    {
        double mVac[3] = {0.0, dm12sq, dm12sq + dm23sq};
        const double delta = 5.0e-9;
        if (dm12sq == 0.0) mVac[0] -= delta;
        if (dm23sq == 0.0) mVac[2] += delta;
        dm_old[0] = dm_old[4] = dm_old[8] = 0.0;
        dm_old[1] = mVac[0] - mVac[1]; dm_old[3] = -dm_old[1];
        dm_old[2] = mVac[0] - mVac[2]; dm_old[6] = -dm_old[2];
        dm_old[5] = mVac[1] - mVac[2]; dm_old[7] = -dm_old[5];
    }

    // Set old global state
    cudaprob3::physics::setMixMatrix_host<double>(U_old);
    cudaprob3::physics::setMassDifferences_host<double>(dm_old);

    // Compute old transition matrix
    cudaprob3::physics::get_transition_matrix(
        cudaprob3::Neutrino, Enu, rho, Len, A_old, 0.0);

    std::cout << "OLD get_transition_matrix:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << "  A[" << i << "][" << j << "] = ("
                      << A_old[i][j].re << ", " << A_old[i][j].im << ")";
            if (j < 2) std::cout << "  ";
        }
        std::cout << std::endl;
    }

    // ====== NEW ======
    cudaprob3::MixingMatrix mix_new(cudaprob3::MixingParams(theta12, theta13, theta23, dCP));
    cudaprob3::MassOrdering massOrd_new;
    massOrd_new.setMassDifferences(dm12sq, dm23sq);
    cudaprob3::TransitionAmplitude amp_new(mix_new, massOrd_new);

    cudaprob3::Matrix3x3<double> A_new_cap;
    amp_new.compute(Len, Enu, rho, cudaprob3::NeutrinoType::Neutrino, A_new_cap, 0.0);

    std::cout << "\nNEW TransitionAmplitude::compute:" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << "  A[" << i << "][" << j << "] = ("
                      << A_new_cap(i,j).re << ", " << A_new_cap(i,j).im << ")";
            if (j < 2) std::cout << "  ";
        }
        std::cout << std::endl;
    }

    // Compare
    std::cout << "\n=== DIFFERENCES ===" << std::endl;
    double max_abs = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double d_re = std::abs(A_old[i][j].re - A_new_cap(i,j).re);
            double d_im = std::abs(A_old[i][j].im - A_new_cap(i,j).im);
            max_abs = std::max(max_abs, std::max(d_re, d_im));
            if (d_re > 1e-12 || d_im > 1e-12) {
                std::cout << "  A[" << i << "][" << j << "]: Δre=" << d_re
                          << " Δim=" << d_im << std::endl;
            }
        }
    }
    std::cout << "Max absolute diff: " << max_abs << std::endl;

    if (max_abs < 1e-10) std::cout << "PHYSICS: MATCH" << std::endl;
    else                  std::cout << "PHYSICS: MISMATCH" << std::endl;

    return 0;
}
