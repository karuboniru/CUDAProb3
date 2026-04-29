#pragma once

#include "single_gpu_calculator.cuh"
#include "../../include/cudaprob3/types.hpp"

#ifndef __CUDACC__
#  include <expected>
#  include <memory>
#endif

#include <span>
#include <string>
#include <vector>

namespace cudaprob3 {

// Result of a batch calculation.
struct BatchResult {
    std::vector<std::vector<double, PinnedAllocator<double>>> h_data;
    int n_cosines  = 0;
    int n_energies = 0;

    ResultView<double> operator[](std::size_t i) const noexcept {
        return ResultView<double>{
            std::span<const double>{h_data[i].data(), h_data[i].size()},
            n_cosines, n_energies
        };
    }

    [[nodiscard]] std::size_t size() const noexcept { return h_data.size(); }
};

// Processes B PMNS parameter sets using a 3D kernel launch (z-dim = batch).
class BatchCalculator {
public:
#ifndef __CUDACC__
    template <typename Grid, typename Model>
    static std::expected<BatchCalculator, std::string>
    create(std::vector<int> deviceIds,
           std::shared_ptr<Grid> grid,
           std::shared_ptr<Model> model,
           int chunkSize = 64) {

        if (deviceIds.empty())
            return std::unexpected("device list is empty");
        if (chunkSize < 1)
            return std::unexpected("chunkSize must be >= 1");

        BatchCalculator b;
        b.nCos_      = grid->nCosines();
        b.nE_        = grid->nEnergies();
        b.chunkSize_ = chunkSize;
        b.deviceIds_ = std::move(deviceIds);

        b.grid_  = std::make_shared<ArbitraryGrid>(
            std::vector<double>(grid->cosines().begin(),  grid->cosines().end()),
            std::vector<double>(grid->energies().begin(), grid->energies().end()),
            grid->productionHeightKm());
        b.modelRadii_    = std::vector<double>(model->radii().begin(),     model->radii().end());
        b.modelDensities_= std::vector<double>(model->densities().begin(), model->densities().end());
        b.prodHeightCm_  = grid->productionHeightKm() * 1e5;
        b.maxlayers_     = model->buildMaxlayers(grid->cosines());

        return b;
    }
#endif // !__CUDACC__

    // Returns BatchResult; throws std::runtime_error on failure.
    [[nodiscard]] BatchResult
    calculate(std::span<const OscillationParams* const> params, NeutrinoType type);

private:
    BatchCalculator() = default;

    int nCos_ = 0, nE_ = 0, chunkSize_ = 64;
    std::vector<int> deviceIds_;
    double prodHeightCm_ = 0;
    std::vector<int>    maxlayers_;
    std::vector<double> modelRadii_, modelDensities_;
    std::shared_ptr<ArbitraryGrid> grid_;
};

} // namespace cudaprob3
