#pragma once

#include <cmath>
#include <span>
#include <vector>

namespace cudaprob3 {

// Grid of (cosine zenith, energy) bins. Production height is an attribute
// since it affects the atmospheric path length.

class UniformGrid {
public:
    // Cosines linearly spaced [cosMin, cosMax], energies log-spaced [eMin, eMax].
    UniformGrid(int nCos, double cosMin, double cosMax,
                int nE,   double eMin,   double eMax,
                double productionHeightKm)
        : prodHeight_(productionHeightKm) {
        cosines_.resize(nCos);
        energies_.resize(nE);
        const double cosStep = (cosMax - cosMin) / (nCos - 1);
        for (int i = 0; i < nCos - 1; ++i) cosines_[i] = cosMin + i * cosStep;
        cosines_[nCos - 1] = cosMax;

        const double logMin = std::log(eMin), logMax = std::log(eMax);
        const double logStep = (logMax - logMin) / (nE - 1);
        cosines_[0] = cosMin;
        energies_[0] = eMin;
        for (int i = 1; i < nE - 1; ++i) energies_[i] = std::exp(logMin + i * logStep);
        energies_[nE - 1] = eMax;
    }

    [[nodiscard]] std::span<const double> cosines()  const noexcept { return cosines_; }
    [[nodiscard]] std::span<const double> energies() const noexcept { return energies_; }
    [[nodiscard]] int nCosines()  const noexcept { return static_cast<int>(cosines_.size()); }
    [[nodiscard]] int nEnergies() const noexcept { return static_cast<int>(energies_.size()); }
    [[nodiscard]] double productionHeightKm() const noexcept { return prodHeight_; }

private:
    std::vector<double> cosines_, energies_;
    double prodHeight_;
};

class ArbitraryGrid {
public:
    ArbitraryGrid(std::vector<double> cosines,
                  std::vector<double> energies,
                  double productionHeightKm)
        : cosines_(std::move(cosines)), energies_(std::move(energies)),
          prodHeight_(productionHeightKm) {}

    [[nodiscard]] std::span<const double> cosines()  const noexcept { return cosines_; }
    [[nodiscard]] std::span<const double> energies() const noexcept { return energies_; }
    [[nodiscard]] int nCosines()  const noexcept { return static_cast<int>(cosines_.size()); }
    [[nodiscard]] int nEnergies() const noexcept { return static_cast<int>(energies_.size()); }
    [[nodiscard]] double productionHeightKm() const noexcept { return prodHeight_; }

private:
    std::vector<double> cosines_, energies_;
    double prodHeight_;
};

} // namespace cudaprob3
