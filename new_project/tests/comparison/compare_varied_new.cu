// Multi-trajectory GPU comparison — new code
// Tests varied cosines to isolate boundary-crossing behavior
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "cudaprob3/cudaprob3.hpp"

int main() {
    // Same cosines as old comparison
    std::vector<double> cosines = {
        0.0,      // no crossing (atmosphere only, zero density)
        -0.447,   // at boundary
        -0.448,   // just past boundary, crosses 1 layer
        -0.7,     // well within mantle, crosses 1-2 boundaries  
        -0.838,   // at outer core boundary
        -0.839,   // just past outer core, crosses 2-3 boundaries
        -0.95,    // deep, crosses 3 boundaries
        -0.99,    // very deep, crosses 4 boundaries
        -1.0      // vertical, crosses all layers
    };
    std::vector<int> expected_layers = {0, 1, 1, 1, 2, 2, 3, 4, 4};

    int n_cosines = cosines.size();

    std::vector<double> energies = {0.1, 1.0, 5.0, 10.0};
    int n_energies = energies.size();

    std::vector<double> radii = {0.0, 1220.0, 3480.0, 5701.0, 6371.0};
    std::vector<double> rhos  = {13.0, 13.0, 11.3, 5.0, 3.3};
    cudaprob3::EarthModel earthModel(radii, rhos);

    cudaprob3::PropagatorConfig cfg;
    cfg.mixing       = cudaprob3::MixingParams(0.5839, 0.1484, 0.7385, 3.9095);
    cfg.dm12sq       = 7.42e-5;
    cfg.dm23sq       = 2.517e-3;
    cfg.energies     = energies;
    cfg.cosines      = cosines;
    cfg.earthModel   = earthModel;
    cfg.productionHeightKm = 0.0;

    cudaprob3::SingleGPUPropagator<double> prop(0, n_cosines, n_energies);
    prop.configure(cfg);
    prop.calculate(cudaprob3::NeutrinoType::Neutrino);

    std::cout << std::scientific << std::setprecision(15);
    for (int ic = 0; ic < n_cosines; ++ic) {
        for (int ie = 0; ie < n_energies; ++ie) {
            std::cout << "C " << ic << " E " << ie << " cosZ=" << cosines[ic]
                      << " Enu=" << energies[ie] << " expL=" << expected_layers[ic];
            for (int t = 0; t < 9; ++t)
                std::cout << " " << prop.getProbability(ic, ie, static_cast<cudaprob3::ProbType>(t));
            std::cout << std::endl;
        }
    }
    return 0;
}
