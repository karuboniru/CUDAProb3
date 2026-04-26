#ifndef CUDAPROB3_ENGINE_STREAM_POOL_HPP
#define CUDAPROB3_ENGINE_STREAM_POOL_HPP

#include <vector>
#include <cuda_runtime.h>

#include "cudaprob3/core/cuda_helpers.hpp"

namespace cudaprob3 {

class StreamPool {
public:
    explicit StreamPool(int deviceId, int numStreams = 4)
        : deviceId_(deviceId), next_(0)
    {
        cudaSetDevice(deviceId_);
        streams_.resize(numStreams);
        for (auto& s : streams_) {
            CUDA_CHECK(cudaStreamCreate(&s));
        }
    }

    ~StreamPool() {
        cudaSetDevice(deviceId_);
        for (auto s : streams_) {
            cudaStreamDestroy(s);
        }
    }

    StreamPool(const StreamPool&) = delete;
    StreamPool& operator=(const StreamPool&) = delete;

    cudaStream_t acquire() {
        cudaStream_t s = streams_[next_];
        next_ = (next_ + 1) % streams_.size();
        return s;
    }

    void synchronize_all() {
        cudaSetDevice(deviceId_);
        for (auto s : streams_) {
            cudaStreamSynchronize(s);
        }
    }

    int device_id() const { return deviceId_; }
    int size() const { return static_cast<int>(streams_.size()); }

private:
    int deviceId_;
    int next_;
    std::vector<cudaStream_t> streams_;
};

} // namespace cudaprob3

#endif
