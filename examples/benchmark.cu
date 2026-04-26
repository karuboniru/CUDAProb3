#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <string>

#include "cudaprob3/cudaprob3.hpp"

using namespace cudaprob3;
using hires_clock = std::chrono::high_resolution_clock;

static void print_params(const PropagatorConfig& cfg) {
    std::cout << "Neutrino parameters:" << std::endl;
    std::cout << "  theta12=" << cfg.mixing.theta12
              << " theta13=" << cfg.mixing.theta13
              << " theta23=" << cfg.mixing.theta23
              << " dCP=" << cfg.mixing.dCP << std::endl;
    std::cout << "  dm12sq=" << cfg.dm12sq << " eV^2"
              << " dm23sq=" << cfg.dm23sq << " eV^2" << std::endl;
    std::cout << "Grid: " << cfg.cosines.size() << " cosines x "
              << cfg.energies.size() << " energies" << std::endl;
}

static EarthModel loadModel(const std::string& filename) {
    EarthModel em;
    em.loadFromFile(filename);
    std::cout << "Loaded " << em.nLayers() << " layers from " << filename << std::endl;
    return em;
}

int main(int argc, char** argv) {
    int nCosines  = argc > 1 ? std::stoi(argv[1]) : 100;
    int nEnergies = argc > 2 ? std::stoi(argv[2]) : 100;

    std::vector<int> gpuIds = {0};
    if (argc > 3) {
        gpuIds.clear();
        for (int i = 3; i < argc; ++i)
            gpuIds.push_back(std::stoi(argv[i]));
    }

    std::cout << "=== CUDAProb3 v2.0 — Multi-GPU Neutrino Oscillation Propagator ===" << std::endl;
    std::cout << "Grid: " << nCosines << " cosines x " << nEnergies << " energies" << std::endl;
    std::cout << "GPUs: ";
    for (int id : gpuIds) std::cout << id << " ";
    std::cout << std::endl;

    auto devices = DeviceInfo::enumerate();
    for (auto& d : devices) {
        std::cout << "  GPU " << d.id << ": " << d.name
                  << " (SM " << d.major << "." << d.minor << ")"
                  << std::endl;
    }

    std::vector<double> cosines(nCosines);
    std::vector<double> energies(nEnergies);
    for (int i = 0; i < nCosines; ++i)
        cosines[i] = -1.0 + i * 1.0 / (nCosines - 1);
    for (int i = 0; i < nEnergies; ++i)
        energies[i] = 0.1 + i * 10.0 / (nEnergies - 1);

    PropagatorConfig cfg;
    cfg.mixing = MixingParams(0.5839, 0.1484, 0.7385, 3.9095);
    cfg.dm12sq = 7.42e-5;
    cfg.dm23sq = 2.517e-3;
    cfg.cosines  = cosines;
    cfg.energies = energies;
    const std::string modelFile = "/run/media/data/yan/CUDAProb3-alt/example/models/PREM_4layer.dat";
    cfg.earthModel = loadModel(modelFile);
    cfg.productionHeightKm = 0.0;

    print_params(cfg);

    std::cout << "\n--- Single-GPU Run ---" << std::endl;
    {
        SingleGPUPropagator<double> prop(0, nCosines, nEnergies);
        prop.configure(cfg);

        auto t0 = hires_clock::now();
        prop.calculate(NeutrinoType::Neutrino);
        auto t1 = hires_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Time: " << ms << " ms" << std::endl;

        std::cout << "Sample results (cosine=0, energy=0):" << std::endl;
        std::cout << "  P(e→e)=" << prop.getProbability(0, 0, ProbType::e_e);
        std::cout << "  P(e→μ)=" << prop.getProbability(0, 0, ProbType::e_m);
        std::cout << "  P(e→τ)=" << prop.getProbability(0, 0, ProbType::e_t) << std::endl;
    }

    if (gpuIds.size() > 1) {
        std::cout << "\n--- Multi-GPU Run (" << gpuIds.size() << " GPUs) ---" << std::endl;
        {
            MultiGPUPropagator prop(gpuIds, nCosines, nEnergies,
                                     WorkDistributionStrategy::Cyclic);
            prop.configure(cfg);

            auto t0 = hires_clock::now();
            prop.calculate(NeutrinoType::Neutrino);
            auto t1 = hires_clock::now();

            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "Time: " << ms << " ms" << std::endl;

            std::cout << "Sample results (cosine=0, energy=0):" << std::endl;
            std::cout << "  P(e→e)=" << prop.getProbability(0, 0, ProbType::e_e);
            std::cout << "  P(e→μ)=" << prop.getProbability(0, 0, ProbType::e_m);
            std::cout << "  P(e→τ)=" << prop.getProbability(0, 0, ProbType::e_t) << std::endl;
        }
    }

    std::cout << "\n--- Parallel Sessions (2 parameter sets) ---" << std::endl;
    {
        PropagatorConfig cfg2 = cfg;
        cfg2.mixing.dCP = 4.5; // different dCP

        SessionConfig sCfg1{cfg, {0}, WorkDistributionStrategy::Block};
        SessionConfig sCfg2{cfg2, {0}, WorkDistributionStrategy::Block};

        auto& mgr = SessionManager::instance();

        auto t0 = hires_clock::now();

        auto session1 = mgr.createSession("fit_1", sCfg1);
        auto session2 = mgr.createSession("fit_2", sCfg2);

        auto f1 = session1->runAsync(NeutrinoType::Neutrino);
        auto f2 = session2->runAsync(NeutrinoType::Neutrino);

        auto r1 = f1.get();
        auto r2 = f2.get();

        auto t1 = hires_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::cout << "Both sessions completed in " << ms << " ms" << std::endl;
        std::cout << "  Session 1: P(e→e)[0,0]=" << r1.p(0, 0, ProbType::e_e) << std::endl;
        std::cout << "  Session 2: P(e→e)[0,0]=" << r2.p(0, 0, ProbType::e_e) << std::endl;
    }

    return 0;
}
