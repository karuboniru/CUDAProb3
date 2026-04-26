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

// Distributes cosine bins cyclically across N GPUs, launches kernels
// on all GPUs in parallel, then aggregates results into a merged pinned buffer.
class MultiGPUCalculator {
public:
#ifndef __CUDACC__
    template <typename Grid, typename Model>
    static std::expected<MultiGPUCalculator, std::string>
    create(std::vector<int> deviceIds,
           std::shared_ptr<Grid> grid,
           std::shared_ptr<Model> model,
           bool useCUDAGraphs = false) {

        if (deviceIds.empty())
            return std::unexpected("device list is empty");

        const int nGPU   = static_cast<int>(deviceIds.size());
        const int nCos   = grid->nCosines();
        const int nE     = grid->nEnergies();

        std::vector<SingleGPUCalculator> calcs;
        calcs.reserve(nGPU);

        for (int g = 0; g < nGPU; ++g) {
            std::vector<double> subCos;
            for (int i = g; i < nCos; i += nGPU)
                subCos.push_back(grid->cosines()[i]);

            auto subGrid = std::make_shared<ArbitraryGrid>(
                std::move(subCos),
                std::vector<double>(grid->energies().begin(), grid->energies().end()),
                grid->productionHeightKm());

            SingleGPUCalculator::Config cfg;
            cfg.deviceId      = deviceIds[g];
            cfg.useCUDAGraphs = useCUDAGraphs;

            auto calcOrErr = SingleGPUCalculator::create(cfg, subGrid, model);
            if (!calcOrErr) return std::unexpected(calcOrErr.error());

            calcs.push_back(std::move(*calcOrErr));
            calcs.back().cosineOffset = g;
        }

        MultiGPUCalculator m;
        m.calcs_         = std::move(calcs);
        m.deviceIds_     = std::move(deviceIds);
        m.nCos_          = nCos;
        m.nE_            = nE;
        m.nGPU_          = nGPU;
        m.h_results_.resize(9ULL * static_cast<std::size_t>(nCos) * static_cast<std::size_t>(nE));
        return m;
    }

    [[nodiscard]] std::expected<ResultView, std::string>
    calculate(const OscillationParams& params, NeutrinoType type);
#endif // !__CUDACC__

    [[nodiscard]] int nCosines()  const noexcept { return nCos_; }
    [[nodiscard]] int nEnergies() const noexcept { return nE_; }

private:
    MultiGPUCalculator() = default;

    void mergeResults();

    std::vector<SingleGPUCalculator> calcs_;
    std::vector<int> deviceIds_;
    int nCos_ = 0;
    int nE_   = 0;
    int nGPU_ = 0;

    std::vector<double, PinnedAllocator<double>> h_results_;
};

} // namespace cudaprob3
