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

std::optional<std::string> SingleGPUCalculator::init(
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

SingleGPUCalculator::~SingleGPUCalculator() {
    if (computeStream_) { cudaSetDevice(deviceId_); cudaStreamDestroy(computeStream_); }
    if (xferStream_)    { cudaSetDevice(deviceId_); cudaStreamDestroy(xferStream_); }
    if (computeDone_)   cudaEventDestroy(computeDone_);
    if (xferDone_)      cudaEventDestroy(xferDone_);
}

SingleGPUCalculator::SingleGPUCalculator(SingleGPUCalculator&& o) noexcept
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

SingleGPUCalculator& SingleGPUCalculator::operator=(SingleGPUCalculator&& o) noexcept {
    if (this == &o) return *this;
    this->~SingleGPUCalculator();
    new (this) SingleGPUCalculator(std::move(o));
    return *this;
}

void SingleGPUCalculator::launchKernel(const OscParamsPOD& pod, NeutrinoType type,
                                        int batchSize, cudaStream_t stream) {
    cudaSetDevice(deviceId_);

    cudaMemcpyAsync(thrust::raw_pointer_cast(d_params_.data()),
                    &pod, sizeof(OscParamsPOD),
                    cudaMemcpyHostToDevice, stream);

    kernels::launchOscillationKernel(
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

void SingleGPUCalculator::issueD2H(cudaStream_t stream) {
    const std::size_t bytes = h_results_.size() * sizeof(double);
    cudaMemcpyAsync(h_results_.data(),
                    thrust::raw_pointer_cast(d_results_.data()),
                    bytes, cudaMemcpyDeviceToHost, stream);
}

ResultView SingleGPUCalculator::makeResultView() const noexcept {
    return ResultView{
        std::span<const double>{h_results_.data(), h_results_.size()},
        nCos_, nE_
    };
}

void SingleGPUCalculator::calculateAsync(const OscillationParams& params, NeutrinoType type) {
    cudaSetDevice(deviceId_);
    const bool antineutrino = (type == NeutrinoType::Antineutrino);
    const OscParamsPOD pod = params.computePOD(antineutrino);
    launchKernel(pod, type, 1, computeStream_);
    cudaEventRecord(computeDone_, computeStream_);
    cudaStreamWaitEvent(xferStream_, computeDone_, 0);
    issueD2H(xferStream_);
    cudaEventRecord(xferDone_, xferStream_);
}

} // namespace cudaprob3
