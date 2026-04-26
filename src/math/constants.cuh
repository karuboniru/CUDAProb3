#pragma once

#ifdef __CUDACC__
#  define CUDAPROB3_CONST_HD __host__ __device__
#else
#  define CUDAPROB3_CONST_HD
#endif

namespace cudaprob3 {
namespace constants {

template <typename T>
CUDAPROB3_CONST_HD inline constexpr T tworttwoGf() noexcept { return T(1.52588e-4); }

template <typename T>
CUDAPROB3_CONST_HD inline constexpr T km2cm() noexcept { return T(1.0e5); }

template <typename T>
CUDAPROB3_CONST_HD inline constexpr T REarth() noexcept { return T(6371.0); }

template <typename T>
CUDAPROB3_CONST_HD inline constexpr T REarthcm() noexcept { return REarth<T>() * km2cm<T>(); }

// ρ [g/cm³] × density_convert → effective matter potential factor
template <typename T>
CUDAPROB3_CONST_HD inline constexpr T density_convert() noexcept { return T(0.5); }

} // namespace constants
} // namespace cudaprob3

#undef CUDAPROB3_CONST_HD
