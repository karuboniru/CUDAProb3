#pragma once

#include "../physics/params_pod.cuh"
#include "../../include/cudaprob3/types.hpp"
#include <cuda_runtime.h>

namespace cudaprob3 {
namespace kernels {

// Launch the oscillation probability kernel.
//   - Single-set mode: batchSize=1, d_params points to one OscParamsPOD
//   - Batch mode:      batchSize=B, d_params points to B OscParamsPODs
// Result layout: [batch * 9 * nCos * nE + flavor * nCos * nE + icos * nE + ie]
void launchOscillationKernel(
    NeutrinoType type,
    const double* d_cosines,  int nCos,
    const double* d_energies, int nE,
    const double* d_radii,
    const double* d_rhos,
    const int*    d_maxlayers,
    double productionHeightCm,
    const OscParamsPOD* d_params,
    int batchSize,
    double* d_results,
    cudaStream_t stream);

} // namespace kernels
} // namespace cudaprob3
