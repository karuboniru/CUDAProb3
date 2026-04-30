#pragma once

#include "../../include/cudaprob3/types.hpp"
#include "../../include/cudaprob3/density_model.hpp"
#include "../../include/cudaprob3/grid.hpp"
#include "../../include/cudaprob3/oscillation_params.hpp"
#include "../physics/params_pod.cuh"
#include "../kernels/graph_builder.cuh"
#include "../memory/pinned_allocator.hpp"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#ifndef __CUDACC__
#  include <expected>
#  include <memory>
#endif

#include <optional>
#include <span>
#include <string>
#include <vector>

namespace cudaprob3 {

// Runs the oscillation probability kernel on a single GPU, templated on
// floating-point precision T (float or double).
// Grid/Earth-model inputs (cosines, energies, radii, rhos) remain double
// since they are physical constants where high precision is always desirable.
// Only the result buffer and OscParamsPOD use T.
template<typename T = double>
class SingleGPUCalculator {
public:
    struct Config {
        int    deviceId          = 0;
        bool   useCUDAGraphs     = false;
    };

#ifndef __CUDACC__
    // grid and model are borrowed (shared ownership).
    template <typename Grid, typename Model>
    static std::expected<SingleGPUCalculator<T>, std::string>
    create(const Config& cfg,
           std::shared_ptr<Grid> grid,
           std::shared_ptr<Model> model) {
        SingleGPUCalculator<T> c;
        if (auto err = c.init(cfg, grid->cosines(), grid->energies(),
                               grid->productionHeightKm() * 1e5,
                               model->radii(), model->densities(),
                               model->buildMaxlayers(grid->cosines())))
            return std::unexpected(*err);
        c.useCUDAGraphs_ = cfg.useCUDAGraphs;
        return c;
    }

    // Synchronous: compute + wait + D2H. Returns a view into the pinned buffer.
    [[nodiscard]] std::expected<ResultView<T>, std::string>
    calculate(const OscillationParams& params, NeutrinoType type);

    // Block until async calculation + D2H is complete.
    [[nodiscard]] std::expected<ResultView<T>, std::string> waitForResults();
#endif // !__CUDACC__

    ~SingleGPUCalculator();

    SingleGPUCalculator(const SingleGPUCalculator&) = delete;
    SingleGPUCalculator& operator=(const SingleGPUCalculator&) = delete;
    SingleGPUCalculator(SingleGPUCalculator&&) noexcept;
    SingleGPUCalculator& operator=(SingleGPUCalculator&&) noexcept;

    // Asynchronous: returns immediately. Call waitForResults() to block.
    void calculateAsync(const OscillationParams& params, NeutrinoType type);

    // Device-only: launches kernel on computeStream_, no D2H. d_results_ is
    // valid on the device after the kernel completes on computeStream_.
    void calculateDeviceOnly(const OscillationParams& params, NeutrinoType type);

    [[nodiscard]] int nCosines()  const noexcept { return nCos_; }
    [[nodiscard]] int nEnergies() const noexcept { return nE_; }

    [[nodiscard]] cudaStream_t getComputeStream() const noexcept { return computeStream_; }

    [[nodiscard]] std::span<const T> rawResults() const noexcept {
        return { h_results_.data(), h_results_.size() };
    }

    // Raw device pointer to results buffer (layout: [channel][cosine][energy]).
    // Valid after calculate() or calculateAsync() kernel completes.
    [[nodiscard]] const T* getDeviceResultPtr() const noexcept {
        return thrust::raw_pointer_cast(d_results_.data());
    }

    // Sub-grid support: cosine offset (for multi-GPU cosine split).
    int cosineOffset = 0;
    int cosineCount  = -1;

private:
    SingleGPUCalculator() = default;

    std::optional<std::string> init(
        const Config& cfg,
        std::span<const double> cosines,
        std::span<const double> energies,
        double productionHeightCm,
        std::span<const double> radii,
        std::span<const double> rhos,
        std::vector<int> maxlayers);

    void launchKernel(const OscParamsPOD<T>& pod, NeutrinoType type,
                      int batchSize, cudaStream_t stream);
    void issueD2H(cudaStream_t stream);
    ResultView<T> makeResultView() const noexcept;

    int deviceId_  = 0;
    int nCos_      = 0;
    int nE_        = 0;
    double prodHeightCm_ = 0;
    bool useCUDAGraphs_ = false;

    cudaStream_t computeStream_ = nullptr;
    cudaStream_t xferStream_    = nullptr;
    cudaEvent_t  computeDone_   = nullptr;
    cudaEvent_t  xferDone_      = nullptr;

    // Grid/Earth inputs stay double.
    thrust::device_vector<double> d_cosines_;
    thrust::device_vector<double> d_energies_;
    thrust::device_vector<double> d_radii_;
    thrust::device_vector<double> d_rhos_;
    thrust::device_vector<int>    d_maxlayers_;

    // Results and params use T.
    thrust::device_vector<T>             d_results_;
    thrust::device_vector<OscParamsPOD<T>> d_params_;
    std::vector<T, PinnedAllocator<T>>   h_results_;

    CUDAGraphHandle graph_;
    bool graphCaptured_ = false;
};

} // namespace cudaprob3
