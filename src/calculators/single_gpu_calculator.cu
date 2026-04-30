#include "single_gpu_calculator.cuh"
#include "../kernels/oscillation_kernel.cuh"

#include <thrust/copy.h>
#include <cstring>
#include <stdexcept>

#define CUERR_RET(call, msg)                                            \
    do {                                                                 \
        cudaError_t _e = (call);                                         \
        if (_e != cudaSuccess)                                           \
            return std::string(msg) + ": " + cudaGetErrorString(_e);    \
    } while (0)

namespace cudaprob3 {

template<typename T>
std::optional<std::string> SingleGPUCalculator<T>::init(
    const Config& cfg,
    std::span<const double> cosines,
    std::span<const double> energies,
    double productionHeightCm,
    std::span<const double> radii,
    std::span<const double> rhos,
    std::vector<int> maxlayers)
{
    deviceId_    = cfg.deviceId;
    nCos_        = static_cast<int>(cosines.size());
    nE_          = static_cast<int>(energies.size());
    prodHeightCm_ = productionHeightCm;

    int nDevices = 0;
    if (cudaGetDeviceCount(&nDevices) != cudaSuccess || deviceId_ >= nDevices)
        return "invalid device id " + std::to_string(deviceId_);

    cudaSetDevice(deviceId_);

    CUERR_RET(cudaStreamCreate(&computeStream_), "create computeStream");
    CUERR_RET(cudaStreamCreate(&xferStream_),    "create xferStream");
    CUERR_RET(cudaEventCreate(&computeDone_),    "create computeDone");
    CUERR_RET(cudaEventCreate(&xferDone_),       "create xferDone");

    d_cosines_   = thrust::device_vector<double>(cosines.begin(), cosines.end());
    d_energies_  = thrust::device_vector<double>(energies.begin(), energies.end());
    d_radii_     = thrust::device_vector<double>(radii.begin(), radii.end());
    d_rhos_      = thrust::device_vector<double>(rhos.begin(), rhos.end());
    d_maxlayers_ = thrust::device_vector<int>(maxlayers.begin(), maxlayers.end());

    const std::size_t resultCount = 9ULL * static_cast<std::size_t>(nCos_)
                                        * static_cast<std::size_t>(nE_);
    d_results_.resize(resultCount);
    h_results_.resize(resultCount);
    d_params_.resize(1);

    return std::nullopt;
}

template<typename T>
SingleGPUCalculator<T>::~SingleGPUCalculator() {
    if (computeStream_) { cudaSetDevice(deviceId_); cudaStreamDestroy(computeStream_); }
    if (xferStream_)    { cudaSetDevice(deviceId_); cudaStreamDestroy(xferStream_); }
    if (computeDone_)   cudaEventDestroy(computeDone_);
    if (xferDone_)      cudaEventDestroy(xferDone_);
}

template<typename T>
SingleGPUCalculator<T>::SingleGPUCalculator(SingleGPUCalculator<T>&& o) noexcept
    : deviceId_(o.deviceId_), nCos_(o.nCos_), nE_(o.nE_),
      prodHeightCm_(o.prodHeightCm_), useCUDAGraphs_(o.useCUDAGraphs_),
      computeStream_(o.computeStream_), xferStream_(o.xferStream_),
      computeDone_(o.computeDone_), xferDone_(o.xferDone_),
      d_cosines_(std::move(o.d_cosines_)),
      d_energies_(std::move(o.d_energies_)),
      d_radii_(std::move(o.d_radii_)),
      d_rhos_(std::move(o.d_rhos_)),
      d_maxlayers_(std::move(o.d_maxlayers_)),
      d_results_(std::move(o.d_results_)),
      d_params_(std::move(o.d_params_)),
      h_results_(std::move(o.h_results_)),
      graph_(std::move(o.graph_)),
      graphCaptured_(o.graphCaptured_)
{
    o.computeStream_ = nullptr; o.xferStream_ = nullptr;
    o.computeDone_   = nullptr; o.xferDone_   = nullptr;
    o.graphCaptured_ = false;
}

template<typename T>
SingleGPUCalculator<T>& SingleGPUCalculator<T>::operator=(SingleGPUCalculator<T>&& o) noexcept {
    if (this == &o) return *this;
    this->~SingleGPUCalculator();
    new (this) SingleGPUCalculator(std::move(o));
    return *this;
}

template<typename T>
void SingleGPUCalculator<T>::launchKernel(const OscParamsPOD<T>& pod, NeutrinoType type,
                                           int batchSize, cudaStream_t stream) {
    cudaSetDevice(deviceId_);

    cudaMemcpyAsync(thrust::raw_pointer_cast(d_params_.data()),
                    &pod, sizeof(OscParamsPOD<T>),
                    cudaMemcpyHostToDevice, stream);

    kernels::launchOscillationKernel<T>(
        type,
        thrust::raw_pointer_cast(d_cosines_.data()),   nCos_,
        thrust::raw_pointer_cast(d_energies_.data()),  nE_,
        thrust::raw_pointer_cast(d_radii_.data()),
        thrust::raw_pointer_cast(d_rhos_.data()),
        thrust::raw_pointer_cast(d_maxlayers_.data()),
        prodHeightCm_,
        thrust::raw_pointer_cast(d_params_.data()),
        batchSize,
        thrust::raw_pointer_cast(d_results_.data()),
        stream);
}

template<typename T>
void SingleGPUCalculator<T>::issueD2H(cudaStream_t stream) {
    const std::size_t bytes = h_results_.size() * sizeof(T);
    cudaMemcpyAsync(h_results_.data(),
                    thrust::raw_pointer_cast(d_results_.data()),
                    bytes, cudaMemcpyDeviceToHost, stream);
}

template<typename T>
ResultView<T> SingleGPUCalculator<T>::makeResultView() const noexcept {
    return ResultView<T>{
        std::span<const T>{h_results_.data(), h_results_.size()},
        nCos_, nE_
    };
}

template<typename T>
void SingleGPUCalculator<T>::calculateAsync(const OscillationParams& params, NeutrinoType type) {
    cudaSetDevice(deviceId_);
    const bool antineutrino = (type == NeutrinoType::Antineutrino);
    const OscParamsPOD<T> pod = params.computePOD<T>(antineutrino);
    launchKernel(pod, type, 1, computeStream_);
    cudaEventRecord(computeDone_, computeStream_);
    cudaStreamWaitEvent(xferStream_, computeDone_, 0);
    issueD2H(xferStream_);
    cudaEventRecord(xferDone_, xferStream_);
}

template<typename T>
void SingleGPUCalculator<T>::calculateDeviceOnly(const OscillationParams& params, NeutrinoType type) {
    cudaSetDevice(deviceId_);
    const bool antineutrino = (type == NeutrinoType::Antineutrino);
    const OscParamsPOD<T> pod = params.computePOD<T>(antineutrino);
    if (useCUDAGraphs_ && graphCaptured_) {
        cudaMemcpy(thrust::raw_pointer_cast(d_params_.data()),
                   &pod, sizeof(OscParamsPOD<T>), cudaMemcpyHostToDevice);
        graph_.replay(computeStream_);
    } else {
        launchKernel(pod, type, 1, computeStream_);
    }
    cudaEventRecord(computeDone_, computeStream_);
}

// Explicit instantiations for the CUDA-compiled member functions.
template class SingleGPUCalculator<float>;
template class SingleGPUCalculator<double>;

} // namespace cudaprob3
