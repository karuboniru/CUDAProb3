#ifndef CUDAPROB3_SESSION_SESSION_MANAGER_HPP
#define CUDAPROB3_SESSION_SESSION_MANAGER_HPP

#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <future>
#include <algorithm>
#include <stdexcept>

#include "cudaprob3/session/session.hpp"
#include "cudaprob3/engine/device.hpp"

namespace cudaprob3 {

class ResourceScheduler {
public:
    ResourceScheduler() {
        auto devices = DeviceInfo::enumerate();
        for (auto& d : devices) {
            availableIds_.push_back(d.id);
        }
        if (availableIds_.empty())
            throw std::runtime_error("No CUDA GPUs available");
    }

    explicit ResourceScheduler(const std::vector<int>& preferredIds) {
        availableIds_ = preferredIds;
    }

    std::vector<int> allocate(int nGPUs) {
        if (nGPUs > static_cast<int>(availableIds_.size())) {
            // Round-robin: can't allocate more than available
            return availableIds_;
        }
        // Simple allocation: return first nGPUs
        return std::vector<int>(availableIds_.begin(), availableIds_.begin() + nGPUs);
    }

    void release(const std::vector<int>& ids) {
        // No-op for static allocation
    }

private:
    std::vector<int> availableIds_;
};

class SessionManager {
public:
    static SessionManager& instance() {
        static SessionManager mgr;
        return mgr;
    }

    std::unique_ptr<Session> createSession(const std::string& name,
                                            const SessionConfig& config) {
        auto gpuIds = scheduler_.allocate(config.gpuIds.size());
        SessionConfig cfg = config;
        cfg.gpuIds = gpuIds;
        return std::make_unique<Session>(name, cfg);
    }

    ResultSet computeSync(const std::string& name, const SessionConfig& config,
                          NeutrinoType type = NeutrinoType::Neutrino) {
        auto session = createSession(name, config);
        session->runSync(type);
        return session->result();
    }

    void submitBatch(const std::vector<SessionConfig>& configs,
                     NeutrinoType type = NeutrinoType::Neutrino) {
        std::vector<std::future<ResultSet>> futures;
        for (size_t i = 0; i < configs.size(); ++i) {
            auto session = createSession("batch_" + std::to_string(i), configs[i]);
            futures.push_back(session->runAsync(type));
            sessions_.push_back(std::move(session));
        }
        for (auto& f : futures) f.wait();
    }

    void collectAll(std::vector<ResultSet>& results) {
        results.clear();
        for (auto& s : sessions_) {
            results.push_back(s->result());
        }
        sessions_.clear();
    }

private:
    SessionManager() = default;
    ResourceScheduler scheduler_;
    std::vector<std::unique_ptr<Session>> sessions_;
};

} // namespace cudaprob3

#endif
