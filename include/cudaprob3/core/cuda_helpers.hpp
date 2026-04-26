#ifndef CUDAPROB3_CORE_CUDA_HELPERS_HPP
#define CUDAPROB3_CORE_CUDA_HELPERS_HPP

#include <iostream>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + \
                                     ":" + std::to_string(__LINE__) + \
                                     " (" + cudaGetErrorName(err) + "): " + \
                                     cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuBLAS error at ") + __FILE__ + \
                                     ":" + std::to_string(__LINE__) + \
                                     " (code " + std::to_string(status) + ")"); \
        } \
    } while(0)

inline void cuda_sync_and_check(cudaStream_t stream = 0) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

inline void cuda_check_device(int deviceId) {
    int nDevices;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    if (nDevices == 0)
        throw std::runtime_error("No CUDA-capable GPU found");
    if (deviceId >= nDevices) {
        std::cerr << "Available GPUs:" << std::endl;
        for (int j = 0; j < nDevices; j++) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, j));
            std::cerr << "  Id " << j << " : " << prop.name
                      << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;
        }
        throw std::runtime_error("Requested GPU " + std::to_string(deviceId) + " is not available");
    }
    CUDA_CHECK(cudaSetDevice(deviceId));
    CUDA_CHECK(cudaFree(0)); // Force lazy context initialization
}

#endif
