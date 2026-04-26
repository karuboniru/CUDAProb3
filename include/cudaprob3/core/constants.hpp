#ifndef CUDAPROB3_CORE_CONSTANTS_HPP
#define CUDAPROB3_CORE_CONSTANTS_HPP

namespace cudaprob3 {

struct Constants {
    __host__ __device__ static constexpr double tworttwoGf() { return 1.52588e-4; }
    __host__ __device__ static constexpr double km2cm() { return 1.0e5; }
    __host__ __device__ static constexpr double REarth() { return 6371.0; }
    __host__ __device__ static constexpr double REarthcm() { return REarth() * km2cm(); }
    __host__ __device__ static constexpr double density_convert() { return 0.5; }

    __host__ __device__ static constexpr double LoEfac() { return 2.534; }
};

} // namespace cudaprob3

#endif
