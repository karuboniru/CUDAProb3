#ifndef CUDAPROB3_PROPAGATOR_MULTI_GPU_PROPAGATOR_HPP
#define CUDAPROB3_PROPAGATOR_MULTI_GPU_PROPAGATOR_HPP

#include <memory>
#include <vector>
#include <thread>

#include "cudaprob3/propagator/propagator.hpp"
#include "cudaprob3/propagator/single_gpu_propagator.hpp"
#include "cudaprob3/propagator/work_distribution.hpp"
#include "cudaprob3/core/cuda_helpers.hpp"

namespace cudaprob3 {

class MultiGPUPropagator : public Propagator {
public:
    MultiGPUPropagator(const std::vector<int>& deviceIds,
                       int nCosines, int nEnergies,
                       WorkDistributionStrategy distStrategy = WorkDistributionStrategy::Cyclic)
        : deviceIds_(deviceIds), nCosines_(nCosines), nEnergies_(nEnergies),
          distributor_(distStrategy, static_cast<int>(deviceIds.size()))
    {
        for (int id : deviceIds_)
            cuda_check_device(id);

        if (deviceIds_.empty())
            throw std::runtime_error("MultiGPUPropagator: no device IDs");
    }

    MultiGPUPropagator(int nCosines, int nEnergies,
                       WorkDistributionStrategy distStrategy = WorkDistributionStrategy::Cyclic)
        : MultiGPUPropagator(std::vector<int>{0}, nCosines, nEnergies, distStrategy) {}

    void configure(const PropagatorConfig& config) override {
        nCosines_ = static_cast<int>(config.cosines.size());
        nEnergies_ = static_cast<int>(config.energies.size());
        config_ = config;

        // Compute max layers for load-balanced distribution
        std::vector<int> maxLayers(config.cosines.size());
        const auto& coslimits = config.earthModel.coslimits();
        for (size_t i = 0; i < config.cosines.size(); ++i) {
            maxLayers[i] = std::count_if(coslimits.begin(), coslimits.end(),
                [&](double lim) { return config.cosines[i] < lim; });
        }
        distributor_.setMaxLayers(maxLayers);

        auto partitions = distributor_.partition(config.cosines);

        cosinelut_.resize(config.cosines.size());
        gpu_lut_.resize(config.cosines.size());

        subProps_.clear();
        for (auto& part : partitions) {
            if (part.localCosines.empty()) continue;

            PropagatorConfig subConfig = config;
            subConfig.cosines = part.localCosines;

            auto sub = std::make_unique<SingleGPUPropagator<double>>(
                deviceIds_[part.gpuId],
                static_cast<int>(part.localCosines.size()), nEnergies_);

            sub->configure(subConfig);
            subProps_.push_back(std::move(sub));

            for (size_t k = 0; k < part.globalCosineIndices.size(); ++k) {
                cosinelut_[part.globalCosineIndices[k]] = static_cast<int>(k);
                gpu_lut_[part.globalCosineIndices[k]]    = static_cast<int>(subProps_.size() - 1);
            }
        }
    }

    void calculate(NeutrinoType type) override {
        std::vector<std::thread> threads;
        for (auto& prop : subProps_) {
            threads.emplace_back([&prop, type]() {
                prop->calculate(type);
            });
        }
        for (auto& t : threads) t.join();
    }

    double getProbability(int index_cosine, int index_energy,
                          ProbType t) const override {
        int gpuIdx = gpu_lut_[index_cosine];
        int localCosineIdx = cosinelut_[index_cosine];
        return subProps_[gpuIdx]->getProbability(localCosineIdx, index_energy, t);
    }

    int nCosines() const override  { return nCosines_; }
    int nEnergies() const override { return nEnergies_; }
    int nGPUs() const { return static_cast<int>(deviceIds_.size()); }

    MultiGPUPropagator(const MultiGPUPropagator&) = delete;
    MultiGPUPropagator& operator=(const MultiGPUPropagator&) = delete;

private:
    std::vector<int> deviceIds_;
    int nCosines_, nEnergies_;
    WorkDistributor distributor_;
    std::vector<std::unique_ptr<SingleGPUPropagator<double>>> subProps_;
    std::vector<int> cosinelut_;
    std::vector<int> gpu_lut_;
    PropagatorConfig config_;
};

} // namespace cudaprob3

#endif
