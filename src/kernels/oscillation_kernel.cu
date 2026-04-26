// Neutrino oscillation probability kernel.
// Block shape: (32, 8, 1)
//   threadIdx.x = energy lane within warp (0..31) → handles 1 energy bin
//   threadIdx.y = cosine lane within block (0..7)  → one cosine per warp
// Grid:  (⌈nE/32⌉, ⌈nCos/8⌉, 1)  for single-set mode
//        (⌈nE/32⌉, ⌈nCos/8⌉, B)  for batch mode (z = PMNS set index)
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

static constexpr int kSmemParamsBytes = static_cast<int>(sizeof(OscParamsPOD));
static constexpr int kSmemDistOff  = kSmemParamsBytes;
static constexpr int kSmemRhoOff   = kSmemDistOff + 8 * kMaxLayers * static_cast<int>(sizeof(double));
static constexpr int kSmemTotal    = kSmemRhoOff  + 8 * kMaxLayers * static_cast<int>(sizeof(double));

__global__
__launch_bounds__(256, 4)
void oscillationKernel(
    NeutrinoType type,
    const double* __restrict__ d_cosines,    int nCos,
    const double* __restrict__ d_energies,   int nE,
    const double* __restrict__ d_radii,
    const double* __restrict__ d_rhos,
    const int*    __restrict__ d_maxlayers,
    double  productionHeightCm,
    const OscParamsPOD* __restrict__ d_params,
    double* __restrict__ d_results
) {
    extern __shared__ char smem[];
    auto* smem_params = reinterpret_cast<OscParamsPOD*>(smem);
    auto* smem_dist   = reinterpret_cast<double*>(smem + kSmemDistOff);
    auto* smem_rho    = reinterpret_cast<double*>(smem + kSmemRhoOff);

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);

    // batch_idx selects the PMNS parameter set
    const int batch_idx = static_cast<int>(blockIdx.z);
    const OscParamsPOD* my_params = d_params + batch_idx;

    // Collaborative load of OscParamsPOD into shared memory
    static_assert(sizeof(OscParamsPOD) % 8 == 0);
    {
        const int tid     = ty * 32 + tx;
        const int nDoubles = static_cast<int>(sizeof(OscParamsPOD) / 8);
        const auto* src   = reinterpret_cast<const double*>(my_params);
        auto*       dst   = reinterpret_cast<double*>(smem_params);
        for (int d = tid; d < nDoubles; d += 256)
            dst[d] = src[d];
    }
    __syncthreads();

    const OscParamsPOD& p = *smem_params;

    const int icos = static_cast<int>(blockIdx.y) * 8 + ty;
    const int ie   = static_cast<int>(blockIdx.x) * 32 + tx;

    if (icos >= nCos) return;

    const auto warp = cg::tiled_partition<32>(cg::this_thread_block());

    const double cosine  = d_cosines[icos];
    const int maxLayer   = d_maxlayers[icos];
    const double REcm    = constants::REarthcm<double>();
    const double PathLength =
        sqrt((REcm + productionHeightCm) * (REcm + productionHeightCm)
             - REcm * REcm * (1.0 - cosine * cosine))
        - REcm * cosine;
    const double TotalEarthLen = -2.0 * cosine * REcm;

    // Thread 0 of each warp precomputes layer geometry into smem
    if (tx == 0) {
        for (int lyr = 0; lyr <= maxLayer; ++lyr) {
            smem_dist[ty * kMaxLayers + lyr] = physics::getTraversedDistanceOfLayer(
                d_radii, lyr, maxLayer, PathLength, TotalEarthLen, cosine);
            smem_rho [ty * kMaxLayers + lyr] = physics::getDensityOfLayer(
                d_rhos, lyr, maxLayer);
        }
    }
    warp.sync();

    if (ie >= nE) return;

    const double energy = d_energies[ie];

    // Accumulate transition matrix over layers
    auto final_mat       = math::Complex3x3<double>::identity();
    auto core_to_mantle  = math::Complex3x3<double>::identity();

    for (int lyr = 0; lyr <= maxLayer; ++lyr) {
        const double dist    = smem_dist[ty * kMaxLayers + lyr];
        const double density = smem_rho [ty * kMaxLayers + lyr];

        math::Complex3x3<double> A{};
        physics::get_transition_matrix(
            type, energy,
            density * constants::density_convert<double>(),
            dist / constants::km2cm<double>(),
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
    cudaStream_t stream)
{
    const dim3 block(32, 8, 1);
    const dim3 grid(
        (static_cast<unsigned>(nE)   + 31) / 32,
        (static_cast<unsigned>(nCos) + 7)  / 8,
        static_cast<unsigned>(batchSize));

    oscillationKernel<<<grid, block, kSmemTotal, stream>>>(
        type, d_cosines, nCos, d_energies, nE,
        d_radii, d_rhos, d_maxlayers,
        productionHeightCm, d_params, d_results);
}

} // namespace kernels
} // namespace cudaprob3
