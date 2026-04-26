#ifndef CUDAPROB3_PROPAGATOR_WORK_DISTRIBUTION_HPP
#define CUDAPROB3_PROPAGATOR_WORK_DISTRIBUTION_HPP

#include <string>
#include <vector>

namespace cudaprob3 {

enum class WorkDistributionStrategy {
    Cyclic,
    Block,
    LoadBalanced
};

struct CosinePartition {
    int gpuId;
    std::vector<int> globalCosineIndices;
    std::vector<double> localCosines;
};

class WorkDistributor {
public:
    WorkDistributor(WorkDistributionStrategy strategy, int nGPUs)
        : strategy_(strategy), nGPUs_(nGPUs) {}

    void setMaxLayers(const std::vector<int>& maxLayersPerCosine) {
        maxLayers_ = maxLayersPerCosine;
    }

    std::vector<CosinePartition> partition(const std::vector<double>& allCosines) {
        int nCosines = static_cast<int>(allCosines.size());
        std::vector<CosinePartition> partitions(nGPUs_);
        for (int g = 0; g < nGPUs_; ++g)
            partitions[g].gpuId = g;

        switch (strategy_) {
            case WorkDistributionStrategy::Cyclic:
                partitionCyclic(allCosines, partitions);
                break;
            case WorkDistributionStrategy::Block:
                partitionBlock(allCosines, partitions);
                break;
            case WorkDistributionStrategy::LoadBalanced:
                partitionLoadBalanced(allCosines, partitions);
                break;
        }

        return partitions;
    }

    static std::string name(WorkDistributionStrategy s) {
        switch (s) {
            case WorkDistributionStrategy::Cyclic:      return "Cyclic";
            case WorkDistributionStrategy::Block:       return "Block";
            case WorkDistributionStrategy::LoadBalanced: return "LoadBalanced";
        }
        return "Unknown";
    }

private:
    void partitionCyclic(const std::vector<double>& allCosines,
                         std::vector<CosinePartition>& partitions) {
        int n = static_cast<int>(allCosines.size());
        for (int ic = 0; ic < n; ++ic) {
            int gpu = ic % nGPUs_;
            partitions[gpu].globalCosineIndices.push_back(ic);
            partitions[gpu].localCosines.push_back(allCosines[ic]);
        }
    }

    void partitionBlock(const std::vector<double>& allCosines,
                        std::vector<CosinePartition>& partitions) {
        int n = static_cast<int>(allCosines.size());
        int base = 0;
        for (int g = 0; g < nGPUs_; ++g) {
            int cnt = n / nGPUs_;
            if (g < n % nGPUs_) ++cnt;
            for (int ic = base; ic < base + cnt && ic < n; ++ic) {
                partitions[g].globalCosineIndices.push_back(ic);
                partitions[g].localCosines.push_back(allCosines[ic]);
            }
            base += cnt;
        }
    }

    void partitionLoadBalanced(const std::vector<double>& allCosines,
                               std::vector<CosinePartition>& partitions) {
        if (maxLayers_.empty()) {
            partitionCyclic(allCosines, partitions);
            return;
        }
        // Greedy: assign each cosine to GPU with smallest current workload
        std::vector<int> load(nGPUs_, 0);
        int n = static_cast<int>(allCosines.size());
        for (int ic = 0; ic < n; ++ic) {
            int bestGpu = 0;
            int minLoad = load[0];
            for (int g = 1; g < nGPUs_; ++g) {
                if (load[g] < minLoad) { bestGpu = g; minLoad = load[g]; }
            }
            partitions[bestGpu].globalCosineIndices.push_back(ic);
            partitions[bestGpu].localCosines.push_back(allCosines[ic]);
            load[bestGpu] += maxLayers_[ic];
        }
    }

    WorkDistributionStrategy strategy_;
    int nGPUs_;
    std::vector<int> maxLayers_;
};

} // namespace cudaprob3

#endif
