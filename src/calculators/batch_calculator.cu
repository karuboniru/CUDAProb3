#include "batch_calculator.cuh"
#include "../kernels/oscillation_kernel.cuh"

#include <thrust/device_vector.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace cudaprob3 {

BatchResult
BatchCalculator::calculate(std::span<const OscillationParams* const> params,
                            NeutrinoType type) {
    const int B     = static_cast<int>(params.size());
    const int nGPU  = static_cast<int>(deviceIds_.size());
    if (B == 0) return BatchResult{};

    const int baseLoad = B / nGPU;
    const int extra    = B % nGPU;

    BatchResult result;
    result.n_cosines  = nCos_;
    result.n_energies = nE_;
    result.h_data.resize(static_cast<std::size_t>(B));

    const std::size_t singleResultSize = 9ULL * nCos_ * nE_;
    for (int i = 0; i < B; ++i)
        result.h_data[i].resize(singleResultSize);

    struct GPUState {
        int deviceId;
        int batchOffset;
        int batchCount;
        thrust::device_vector<OscParamsPOD>    d_params;
        thrust::device_vector<double>           d_results;
        std::vector<double, PinnedAllocator<double>> h_results;
        cudaStream_t computeStream = nullptr;
        cudaStream_t xferStream    = nullptr;
        cudaEvent_t  computeDone   = nullptr;
        cudaEvent_t  xferDone      = nullptr;
    };

    std::vector<GPUState> states(static_cast<std::size_t>(nGPU));

    int offset = 0;
    for (int g = 0; g < nGPU; ++g) {
        const int count = baseLoad + (g < extra ? 1 : 0);
        states[g].deviceId    = deviceIds_[g];
        states[g].batchOffset = offset;
        states[g].batchCount  = count;
        offset += count;

        if (count == 0) continue;
        cudaSetDevice(deviceIds_[g]);

        if (cudaStreamCreate(&states[g].computeStream) != cudaSuccess ||
            cudaStreamCreate(&states[g].xferStream)    != cudaSuccess ||
            cudaEventCreate (&states[g].computeDone)   != cudaSuccess ||
            cudaEventCreate (&states[g].xferDone)      != cudaSuccess)
            throw std::runtime_error("BatchCalculator: failed to create CUDA streams/events");

        states[g].d_params.resize(static_cast<std::size_t>(count));
        states[g].d_results.resize(singleResultSize * static_cast<std::size_t>(count));
        states[g].h_results.resize(singleResultSize * static_cast<std::size_t>(count));

        std::vector<OscParamsPOD> podBuf(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i)
            podBuf[i] = params[states[g].batchOffset + i]->computePOD();

        cudaMemcpyAsync(thrust::raw_pointer_cast(states[g].d_params.data()),
                        podBuf.data(), sizeof(OscParamsPOD) * count,
                        cudaMemcpyHostToDevice, states[g].computeStream);

        thrust::device_vector<double> d_cos(grid_->cosines().begin(),   grid_->cosines().end());
        thrust::device_vector<double> d_e  (grid_->energies().begin(),  grid_->energies().end());
        thrust::device_vector<double> d_r  (modelRadii_.begin(),         modelRadii_.end());
        thrust::device_vector<double> d_rho(modelDensities_.begin(),     modelDensities_.end());
        thrust::device_vector<int>    d_ml (maxlayers_.begin(),          maxlayers_.end());

        kernels::launchOscillationKernel(
            type,
            thrust::raw_pointer_cast(d_cos.data()), nCos_,
            thrust::raw_pointer_cast(d_e.data()),   nE_,
            thrust::raw_pointer_cast(d_r.data()),
            thrust::raw_pointer_cast(d_rho.data()),
            thrust::raw_pointer_cast(d_ml.data()),
            prodHeightCm_,
            thrust::raw_pointer_cast(states[g].d_params.data()),
            count,
            thrust::raw_pointer_cast(states[g].d_results.data()),
            states[g].computeStream);

        cudaEventRecord(states[g].computeDone, states[g].computeStream);
        cudaStreamWaitEvent(states[g].xferStream, states[g].computeDone, 0);

        const std::size_t bytes = states[g].h_results.size() * sizeof(double);
        cudaMemcpyAsync(states[g].h_results.data(),
                        thrust::raw_pointer_cast(states[g].d_results.data()),
                        bytes, cudaMemcpyDeviceToHost, states[g].xferStream);

        cudaEventRecord(states[g].xferDone, states[g].xferStream);
    }

    for (int g = 0; g < nGPU; ++g) {
        if (states[g].batchCount == 0) continue;
        cudaEventSynchronize(states[g].xferDone);

        for (int i = 0; i < states[g].batchCount; ++i) {
            const double* src = states[g].h_results.data()
                + static_cast<std::size_t>(i) * singleResultSize;
            std::memcpy(result.h_data[states[g].batchOffset + i].data(),
                        src, singleResultSize * sizeof(double));
        }

        cudaStreamDestroy(states[g].computeStream);
        cudaStreamDestroy(states[g].xferStream);
        cudaEventDestroy(states[g].computeDone);
        cudaEventDestroy(states[g].xferDone);
    }

    return result;
}

} // namespace cudaprob3
