#ifndef CUDAPROB3_PROPAGATOR_PROPAGATOR_HPP
#define CUDAPROB3_PROPAGATOR_PROPAGATOR_HPP

#include <string>
#include <vector>

#include "cudaprob3/core/types.hpp"
#include "cudaprob3/physics/mixing.hpp"
#include "cudaprob3/geometry/earth_model.hpp"

namespace cudaprob3 {

struct PropagatorConfig {
    MixingParams mixing;
    double dm12sq, dm23sq;
    std::vector<double> energies;
    std::vector<double> cosines;
    EarthModel earthModel;
    double productionHeightKm = 0.0;
};

class Propagator {
public:
    virtual ~Propagator() = default;

    virtual void configure(const PropagatorConfig& config) = 0;
    virtual void calculate(NeutrinoType type) = 0;

    virtual double getProbability(int index_cosine, int index_energy,
                                  ProbType t) const = 0;

    virtual int nCosines() const = 0;
    virtual int nEnergies() const = 0;
};

} // namespace cudaprob3

#endif
