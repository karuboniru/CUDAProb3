#pragma once

#include "../physics/params_pod.cuh"
#include "../../include/cudaprob3/types.hpp"
#include <cuda_runtime.h>

namespace cudaprob3 {
namespace kernels {

// Launch the oscillation probability kernel, templated on floating-point type T.
//   - Single-set mode: batchSize=1, d_params points to one OscParamsPOD<T>
//   - Batch mode:      batchSize=B, d_params points to B OscParamsPOD<T>s
// Grid inputs (cosines, energies, radii, rhos) stay double for physical accuracy.
// Result layout: [batch * 9 * nCos * nE + flavor * nCos * nE + icos * nE + ie]
template<typename T>
void launchOscillationKernel(
    NeutrinoType type,
    const double* d_cosines,  int nCos,
    const double* d_energies, int nE,
    const double* d_radii,
    const double* d_rhos,
    const int*    d_maxlayers,
    double productionHeightCm,
    const OscParamsPOD<T>* d_params,
    int batchSize,
    T* d_results,
    cudaStream_t stream);

} // namespace kernels
} // namespace cudaprob3
