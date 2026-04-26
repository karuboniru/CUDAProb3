#ifndef CUDAPROB3_SESSION_SESSION_HPP
#define CUDAPROB3_SESSION_SESSION_HPP

#include <memory>
#include <string>
#include <future>
#include <vector>

#include "cudaprob3/core/types.hpp"
#include "cudaprob3/propagator/propagator.hpp"
#include "cudaprob3/propagator/single_gpu_propagator.hpp"
#include "cudaprob3/propagator/multi_gpu_propagator.hpp"
#include "cudaprob3/propagator/work_distribution.hpp"

namespace cudaprob3 {

struct SessionConfig {
    PropagatorConfig params;
    std::vector<int> gpuIds = {0};
    WorkDistributionStrategy distStrategy = WorkDistributionStrategy::Block;
};

struct ResultSet {
    std::vector<double> probabilities; // [nCosines * nEnergies * 9]
    int nCosines, nEnergies;

    double p(int ic, int ie, ProbType t) const {
        size_t idx = static_cast<size_t>(ic) * nEnergies + ie;
        size_t off = static_cast<size_t>(static_cast<int>(t)) * nCosines * nEnergies;
        return probabilities[idx + off];
    }
};

class Session {
public:
    explicit Session(std::string name, const SessionConfig& config)
        : name_(std::move(name)), config_(config)
    {
        if (config_.gpuIds.size() > 1) {
            propagator_ = std::make_unique<MultiGPUPropagator>(
                config_.gpuIds, static_cast<int>(config_.params.cosines.size()),
                static_cast<int>(config_.params.energies.size()),
                config_.distStrategy);
        } else {
            propagator_ = std::make_unique<SingleGPUPropagator<double>>(
                config_.gpuIds[0],
                static_cast<int>(config_.params.cosines.size()),
                static_cast<int>(config_.params.energies.size()));
        }

        propagator_->configure(config_.params);
    }

    void runSync(NeutrinoType type = NeutrinoType::Neutrino) {
        propagator_->calculate(type);
        gatherResults();
    }

    std::future<ResultSet> runAsync(NeutrinoType type = NeutrinoType::Neutrino) {
        auto promise = std::make_shared<std::promise<ResultSet>>();
        auto future = promise->get_future();

        std::thread([this, type, promise]() {
            propagator_->calculate(type);
            promise->set_value(gatherResults());
        }).detach();

        return future;
    }

    ResultSet result() const { return results_; }
    const std::string& name() const { return name_; }

private:
    ResultSet gatherResults() {
        int nc = propagator_->nCosines();
        int ne = propagator_->nEnergies();
        ResultSet r;
        r.nCosines = nc;
        r.nEnergies = ne;
        r.probabilities.resize(static_cast<size_t>(nc) * ne * 9);

        for (int ic = 0; ic < nc; ++ic) {
            for (int ie = 0; ie < ne; ++ie) {
                for (int t = 0; t < 9; ++t) {
                    size_t idx = static_cast<size_t>(ic) * ne + ie;
                    size_t off = static_cast<size_t>(t) * nc * ne;
                    r.probabilities[idx + off] = propagator_->getProbability(
                        ic, ie, static_cast<ProbType>(t));
                }
            }
        }
        results_ = r;
        return r;
    }

    std::string name_;
    SessionConfig config_;
    std::unique_ptr<Propagator> propagator_;
    ResultSet results_;
};

} // namespace cudaprob3

#endif
