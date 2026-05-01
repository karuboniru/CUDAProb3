// Neutrino oscillation probability kernel, templated on floating-point type T.
// Block shape: (32, 8, 1)
//   threadIdx.x = energy lane within warp (0..31) → handles 1 energy bin
//   threadIdx.y = cosine lane within block (0..7)  → one cosine per warp
// Grid:  (⌈nE/32⌉, ⌈nCos/8⌉, 1)  for single-set mode
//        (⌈nE/32⌉, ⌈nCos/8⌉, B)  for batch mode (z = PMNS set index)
//
// Grid inputs (cosines, energies, radii, rhos) stay double for physical accuracy.
// OscParamsPOD<T>, result buffer, and physics math all use T.
//
// Warp cooperation: thread 0 of each warp loads layer geometry into shared
// memory; all 32 energy threads for a given cosine read from smem rather
// than each recomputing sqrt/geometry calls.

#include "oscillation_kernel.cuh"
#include "../physics/barger.cuh"
#include "../physics/earth_model.cuh"
#include "../math/constants.cuh"

#include <cooperative_groups.h>
#include <cstdint>

namespace cg = cooperative_groups;

namespace cudaprob3 {
namespace kernels {

static constexpr int kMaxLayers = 64;

template<typename T>
__global__
__launch_bounds__(256)
void oscillationKernel(
    NeutrinoType type,
    const double* __restrict__ d_cosines,    int nCos,
    const double* __restrict__ d_energies,   int nE,
    const double* __restrict__ d_radii,
    const double* __restrict__ d_rhos,
    const int*    __restrict__ d_maxlayers,
    double  productionHeightCm,
    const OscParamsPOD<T>* __restrict__ d_params,
    T* __restrict__ d_results
) {
    // Shared memory layout:
    //   [0 .. sizeof(OscParamsPOD<T>))      : OscParamsPOD<T>
    //   [kSmemDistOff .. kSmemRhoOff)        : T smem_dist[8 * kMaxLayers]
    //   [kSmemRhoOff  .. kSmemTotal)         : T smem_rho [8 * kMaxLayers]
    constexpr int kSmemParamsBytes = static_cast<int>(sizeof(OscParamsPOD<T>));
    constexpr int kSmemDistOff     = kSmemParamsBytes;
    constexpr int kSmemRhoOff      = kSmemDistOff + 8 * kMaxLayers * static_cast<int>(sizeof(T));
    constexpr int kSmemTotal       = kSmemRhoOff  + 8 * kMaxLayers * static_cast<int>(sizeof(T));

    static_assert(sizeof(OscParamsPOD<T>) % sizeof(T) == 0,
                  "OscParamsPOD<T> size must be a multiple of sizeof(T)");

    extern __shared__ char smem[];
    auto* smem_params = reinterpret_cast<OscParamsPOD<T>*>(smem);
    auto* smem_dist   = reinterpret_cast<T*>(smem + kSmemDistOff);
    auto* smem_rho    = reinterpret_cast<T*>(smem + kSmemRhoOff);

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);

    // batch_idx selects the PMNS parameter set
    const int batch_idx = static_cast<int>(blockIdx.z);
    const OscParamsPOD<T>* my_params = d_params + batch_idx;

    // Collaborative load of OscParamsPOD<T> into shared memory using T-sized words.
    // This is a raw byte copy — works for both float and double since
    // sizeof(OscParamsPOD<T>) is always a multiple of sizeof(T).
    {
        const int tid    = ty * 32 + tx;
        const int nWords = static_cast<int>(sizeof(OscParamsPOD<T>) / sizeof(T));
        const T* src     = reinterpret_cast<const T*>(my_params);
        T*       dst     = reinterpret_cast<T*>(smem_params);
        for (int d = tid; d < nWords; d += 256)
            dst[d] = src[d];
    }
    __syncthreads();

    const OscParamsPOD<T>& p = *smem_params;

    const int icos = static_cast<int>(blockIdx.y) * 8 + ty;
    const int ie   = static_cast<int>(blockIdx.x) * 32 + tx;

    if (icos >= nCos) return;

    const auto warp = cg::tiled_partition<32>(cg::this_thread_block());

    // Geometry in double for accuracy (Earth model is always double-precision).
    const double cosine_d        = d_cosines[icos];
    const int    maxLayer        = d_maxlayers[icos];
    const double REcm_d          = constants::REarthcm<double>();
    const double PathLength_d    =
        sqrt((REcm_d + productionHeightCm) * (REcm_d + productionHeightCm)
             - REcm_d * REcm_d * (1.0 - cosine_d * cosine_d))
        - REcm_d * cosine_d;
    const double TotalEarthLen_d = -2.0 * cosine_d * REcm_d;

    // Thread 0 of each warp precomputes layer geometry, narrowing to T for smem.
    if (tx == 0) {
        for (int lyr = 0; lyr <= maxLayer; ++lyr) {
            smem_dist[ty * kMaxLayers + lyr] = static_cast<T>(
                physics::getTraversedDistanceOfLayer(
                    d_radii, lyr, maxLayer, PathLength_d, TotalEarthLen_d, cosine_d));
            smem_rho[ty * kMaxLayers + lyr] = static_cast<T>(
                physics::getDensityOfLayer(d_rhos, lyr, maxLayer));
        }
    }
    warp.sync();

    if (ie >= nE) return;

    const T energy = static_cast<T>(d_energies[ie]);

    // Accumulate transition matrix over layers — all T arithmetic
    auto final_mat      = math::Complex3x3<T>::identity();
    auto core_to_mantle = math::Complex3x3<T>::identity();

    for (int lyr = 0; lyr <= maxLayer; ++lyr) {
        const T dist    = smem_dist[ty * kMaxLayers + lyr];
        const T density = smem_rho [ty * kMaxLayers + lyr];

        math::Complex3x3<T> A{};
        physics::get_transition_matrix(
            type, energy,
            density * constants::density_convert<T>(),
            dist / constants::km2cm<T>(),
            p, A);

        if (lyr == 0) {
            final_mat = A;
        } else if (lyr < maxLayer) {
            final_mat       = A * final_mat;
            core_to_mantle  = core_to_mantle * A;
        } else {
            final_mat = A * final_mat;
        }
    }
    final_mat = core_to_mantle * final_mat;

    // Store 9 probabilities
    const auto base_stride = static_cast<unsigned long long>(nCos)
                           * static_cast<unsigned long long>(nE);
    const auto cell_offset = static_cast<unsigned long long>(icos)
                           * static_cast<unsigned long long>(nE)
                           + static_cast<unsigned long long>(ie);
    const auto batch_offset = static_cast<unsigned long long>(batch_idx) * 9ULL * base_stride;

    final_mat.storeProbs(d_results + batch_offset + cell_offset, base_stride);
}

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
    cudaStream_t stream)
{
    const dim3 block(32, 8, 1);
    const dim3 grid(
        (static_cast<unsigned>(nE)   + 31) / 32,
        (static_cast<unsigned>(nCos) + 7)  / 8,
        static_cast<unsigned>(batchSize));

    constexpr int kSmemParamsBytes = static_cast<int>(sizeof(OscParamsPOD<T>));
    constexpr int kSmemDistOff     = kSmemParamsBytes;
    constexpr int kSmemRhoOff      = kSmemDistOff + 8 * kMaxLayers * static_cast<int>(sizeof(T));
    constexpr int kSmemTotal       = kSmemRhoOff  + 8 * kMaxLayers * static_cast<int>(sizeof(T));

    oscillationKernel<T><<<grid, block, kSmemTotal, stream>>>(
        type, d_cosines, nCos, d_energies, nE,
        d_radii, d_rhos, d_maxlayers,
        productionHeightCm, d_params, d_results);
}

// Explicit instantiations for float and double.
template __global__ void oscillationKernel<float>(
    NeutrinoType, const double*, int, const double*, int,
    const double*, const double*, const int*, double,
    const OscParamsPOD<float>*, float*);
template __global__ void oscillationKernel<double>(
    NeutrinoType, const double*, int, const double*, int,
    const double*, const double*, const int*, double,
    const OscParamsPOD<double>*, double*);

template void launchOscillationKernel<float>(
    NeutrinoType, const double*, int, const double*, int,
    const double*, const double*, const int*, double,
    const OscParamsPOD<float>*, int, float*, cudaStream_t);
template void launchOscillationKernel<double>(
    NeutrinoType, const double*, int, const double*, int,
    const double*, const double*, const int*, double,
    const OscParamsPOD<double>*, int, double*, cudaStream_t);

} // namespace kernels
} // namespace cudaprob3
