#pragma once

#include "../math/constants.cuh"

#ifdef __CUDACC__
#  define CUDAPROB3_EM_HD __host__ __device__
#else
#  define CUDAPROB3_EM_HD
#endif

namespace cudaprob3 {
namespace physics {

// Density (g/cm³) of layer `layer` given the reflected layer structure.
// layer == 0 is the atmospheric (zero-density) segment.
// Layers 1..max_layer go inward; max_layer+1..2*max_layer-1 mirror back out.
template <typename T>
CUDAPROB3_EM_HD inline T getDensityOfLayer(const T* rhos, int layer, int max_layer) noexcept {
    if (layer == 0) return T(0);
    int i = (layer <= max_layer) ? (layer - 1) : (2*max_layer - layer - 1);
    return rhos[i];
}

// Path length (cm) through layer `layer`.
// PathLength and TotalEarthLength are in cm (already converted by the caller).
template <typename T>
CUDAPROB3_EM_HD inline T getTraversedDistanceOfLayer(
    const T* radii, int layer, int max_layer,
    T PathLength, T TotalEarthLength, T cosine_zenith) noexcept {

    if (cosine_zenith >= T(0))
        return PathLength;
    if (layer == 0)
        return PathLength - TotalEarthLength;

    int i = (layer >= max_layer) ? (-layer - 1 + 2*max_layer) : (layer - 1);

    const T RE = constants::REarth<T>();
    const T sin2 = T(1) - cosine_zenith * cosine_zenith;

    const T CrossThis = T(2) * sqrt(radii[i]   * radii[i]   - RE*RE*sin2);
    const T CrossNext = T(2) * sqrt(radii[i+1] * radii[i+1] - RE*RE*sin2);

    const T dist_km = (i < max_layer - 1) ? T(0.5)*(CrossThis - CrossNext) : CrossThis;
    return dist_km * constants::km2cm<T>();
}

} // namespace physics
} // namespace cudaprob3

#undef CUDAPROB3_EM_HD
