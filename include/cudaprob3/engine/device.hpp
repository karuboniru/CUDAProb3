#ifndef CUDAPROB3_ENGINE_DEVICE_HPP
#define CUDAPROB3_ENGINE_DEVICE_HPP

#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>

#include "cudaprob3/core/cuda_helpers.hpp"

namespace cudaprob3 {

class StreamPool;
class MemoryPool;

struct DeviceInfo {
    int id;
    std::string name;
    int major, minor;
    int multiProcessorCount;
    size_t totalGlobalMem;
    int maxThreadsPerBlock;

    static std::vector<DeviceInfo> enumerate() {
        int count;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        std::vector<DeviceInfo> devices;
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
            devices.push_back({i, prop.name,
                               prop.major, prop.minor,
                               prop.multiProcessorCount,
                               prop.totalGlobalMem,
                               prop.maxThreadsPerBlock});
        }
        return devices;
    }
};

class GPUDevice {
public:
    explicit GPUDevice(int id);
    ~GPUDevice();

    int id() const { return info_.id; }
    const DeviceInfo& info() const { return info_; }
    cudaStream_t default_stream() const { return 0; }

    void synchronize();

    GPUDevice(const GPUDevice&) = delete;
    GPUDevice& operator=(const GPUDevice&) = delete;
    GPUDevice(GPUDevice&&);
    GPUDevice& operator=(GPUDevice&&);

private:
    DeviceInfo info_;
};

} // namespace cudaprob3

#endif
