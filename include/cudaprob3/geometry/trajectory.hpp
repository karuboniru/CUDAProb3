#ifndef CUDAPROB3_GEOMETRY_TRAJECTORY_HPP
#define CUDAPROB3_GEOMETRY_TRAJECTORY_HPP

#include <cmath>
#include <vector>
#include "cudaprob3/core/constants.hpp"

namespace cudaprob3 {

struct TrajectoryParams {
    double cosZ;
    double productionHeightCm;
    int maxLayer;
    double pathLengthCm;
    double totalEarthLengthCm;
};

class TrajectoryBuilder {
public:
    TrajectoryBuilder() = default;

    __host__ __device__
    TrajectoryParams build(double cosZ, double productionHeightCm, int maxLayer) const {
        const double REcm = Constants::REarthcm();

        const double Rdet = REcm + productionHeightCm;
        const double pathLength = std::sqrt(Rdet * Rdet
                                            - REcm * REcm * (1.0 - cosZ * cosZ))
                                - REcm * cosZ;
        const double totalEarthLength = -2.0 * cosZ * REcm;

        return {cosZ, productionHeightCm, maxLayer, pathLength, totalEarthLength};
    }
};

} // namespace cudaprob3

#endif
