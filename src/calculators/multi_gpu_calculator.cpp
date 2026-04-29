#include "multi_gpu_calculator.cuh"
#include "single_gpu_calculator.cuh"

#include <cstring>

namespace cudaprob3 {

std::expected<ResultView<double>, std::string>
MultiGPUCalculator::calculate(const OscillationParams& params, NeutrinoType type) {
    // Fire all GPUs asynchronously
    for (auto& c : calcs_)
        c.calculateAsync(params, type);

    // Wait for all, then merge
    for (auto& c : calcs_) {
        auto r = c.waitForResults();
        if (!r) return std::unexpected(r.error());
    }

    mergeResults();
    return ResultView<double>{
        std::span<const double>{h_results_.data(), h_results_.size()},
        nCos_, nE_
    };
}

void MultiGPUCalculator::mergeResults() {
    // Each GPU g has cosines at global indices g, g+nGPU, g+2*nGPU, ...
    // GPU g produced sub-results indexed [flavor * nSubCos * nE + local_icos * nE + ie]
    // We must scatter into the merged buffer at global_icos = g + k*nGPU.

    for (int g = 0; g < nGPU_; ++g) {
        const auto sub = calcs_[g].rawResults();
        int local_icos = 0;
        for (int global_icos = g; global_icos < nCos_; global_icos += nGPU_, ++local_icos) {
            const int nSubCos = (nCos_ - g + nGPU_ - 1) / nGPU_;  // sub-grid size for this GPU
            for (int f = 0; f < 9; ++f) {
                const double* src = sub.data()
                    + static_cast<std::size_t>(f) * nSubCos * nE_
                    + static_cast<std::size_t>(local_icos) * nE_;
                double* dst = h_results_.data()
                    + static_cast<std::size_t>(f) * nCos_ * nE_
                    + static_cast<std::size_t>(global_icos) * nE_;
                std::memcpy(dst, src, static_cast<std::size_t>(nE_) * sizeof(double));
            }
        }
    }
}

} // namespace cudaprob3
