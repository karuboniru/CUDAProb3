#include <iostream>
#include <iomanip>
#include <vector>
#include <string>

#include "cudaprob3/cudaprob3.hpp"

int main() {
    const int n_cosines  = 20;
    const int n_energies = 20;

    // Same parameters as old comparison
    std::vector<double> cosines(n_cosines);
    std::vector<double> energies(n_energies);
    for (int i = 0; i < n_cosines; ++i)
        cosines[i] = -1.0 + i * 1.0 / (n_cosines - 1);
    for (int i = 0; i < n_energies; ++i)
        energies[i] = 0.1 + i * 10.0 / (n_energies - 1);

    // Same 4-layer PREM
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

    // Output in same format
    std::cout << "# NEW n_cosines=" << n_cosines << " n_energies=" << n_energies << std::endl;
    std::cout << std::scientific << std::setprecision(15);

    for (int ic = 0; ic < n_cosines; ++ic) {
        for (int ie = 0; ie < n_energies; ++ie) {
            std::cout << ic << " " << ie;
            for (int t = 0; t < 9; ++t) {
                double p = prop.getProbability(ic, ie,
                    static_cast<cudaprob3::ProbType>(t));
                std::cout << " " << p;
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
