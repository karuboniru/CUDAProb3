#pragma once

#include "density_model.hpp"
#include "grid.hpp"
#include "oscillation_params.hpp"
#include "types.hpp"
#include "../../src/calculators/single_gpu_calculator.cuh"
#include "../../src/calculators/multi_gpu_calculator.cuh"
#include "../../src/calculators/batch_calculator.cuh"

#include <expected>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace cudaprob3 {

// Default calculator variant uses double precision.
using AnyCalculator = std::variant<SingleGPUCalculator<double>, MultiGPUCalculator>;

// Fluent builder that validates configuration and constructs calculators (double precision).
class CalculatorBuilder {
public:
    CalculatorBuilder& withDevices(std::vector<int> ids) {
        deviceIds_ = std::move(ids); return *this;
    }
    CalculatorBuilder& withDevice(int id) {
        deviceIds_ = {id}; return *this;
    }
    CalculatorBuilder& withGrid(std::shared_ptr<ArbitraryGrid> g) {
        grid_ = std::move(g); return *this;
    }
    CalculatorBuilder& withGrid(std::shared_ptr<UniformGrid> g) {
        grid_ = std::make_shared<ArbitraryGrid>(
            std::vector<double>(g->cosines().begin(),  g->cosines().end()),
            std::vector<double>(g->energies().begin(), g->energies().end()),
            g->productionHeightKm());
        return *this;
    }
    CalculatorBuilder& withDensityModel(std::shared_ptr<PREMModel> m) {
        model_ = std::move(m); return *this;
    }
    CalculatorBuilder& withCUDAGraphs(bool v) { useCUDAGraphs_ = v; return *this; }
    CalculatorBuilder& withBatchChunkSize(int n) { chunkSize_ = n; return *this; }

    [[nodiscard]] std::expected<AnyCalculator, std::string> build() const {
        if (!grid_)  return std::unexpected("grid not set");
        if (!model_) return std::unexpected("density model not set");
        if (deviceIds_.empty()) return std::unexpected("no device IDs specified");

        if (deviceIds_.size() == 1) {
            SingleGPUCalculator<double>::Config cfg{ deviceIds_[0], useCUDAGraphs_ };
            auto c = SingleGPUCalculator<double>::create(cfg, grid_, model_);
            if (!c) return std::unexpected(c.error());
            return AnyCalculator{std::move(*c)};
        } else {
            auto c = MultiGPUCalculator::create(deviceIds_, grid_, model_, useCUDAGraphs_);
            if (!c) return std::unexpected(c.error());
            return AnyCalculator{std::move(*c)};
        }
    }

    [[nodiscard]] std::expected<BatchCalculator, std::string> buildBatch() const {
        if (!grid_)  return std::unexpected("grid not set");
        if (!model_) return std::unexpected("density model not set");
        if (deviceIds_.empty()) return std::unexpected("no device IDs specified");

        return BatchCalculator::create(deviceIds_, grid_, model_, chunkSize_);
    }

private:
    std::vector<int> deviceIds_ = {0};
    std::shared_ptr<ArbitraryGrid> grid_;
    std::shared_ptr<PREMModel> model_;
    bool useCUDAGraphs_ = false;
    int  chunkSize_     = 64;
};

// Helper: call calculate() on an AnyCalculator variant uniformly (double precision).
inline std::expected<ResultView<double>, std::string>
calculate(AnyCalculator& calc, const OscillationParams& params, NeutrinoType type) {
    return std::visit([&](auto& c) { return c.calculate(params, type); }, calc);
}

} // namespace cudaprob3
