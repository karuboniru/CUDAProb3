#ifndef CUDAPROB3_GEOMETRY_EARTH_MODEL_HPP
#define CUDAPROB3_GEOMETRY_EARTH_MODEL_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cudaprob3/core/constants.hpp"

namespace cudaprob3 {

class EarthModel {
public:
    EarthModel() = default;

    EarthModel(const std::vector<double>& radii_km, const std::vector<double>& densities_gcm3) {
        setDensity(radii_km, densities_gcm3);
    }

    void loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file)
            throw std::runtime_error("Could not open density file: " + filename);
        std::vector<double> r, d;
        double radius, density;
        while (file >> radius >> density) {
            r.push_back(radius);
            d.push_back(density);
        }
        setDensity(r, d);
    }

    void setDensity(const std::vector<double>& radii_km, const std::vector<double>& densities_gcm3) {
        if (radii_km.size() != densities_gcm3.size())
            throw std::runtime_error("EarthModel: radii_km.size() != densities_gcm3.size()");
        if (radii_km.empty())
            throw std::runtime_error("EarthModel: empty density profile");

        std::vector<double> r = radii_km, d = densities_gcm3;

        bool needFlip = false;
        if (r.size() >= 2) {
            int sign = (r[1] - r[0] > 0) ? 1 : -1;
            for (size_t i = 1; i < r.size(); ++i) {
                if ((r[i] - r[i-1]) * sign < 0)
                    throw std::runtime_error("EarthModel: radii must be monotonic");
            }
            if (sign == 1) needFlip = true;
        }

        if (needFlip) {
            std::reverse(r.begin(), r.end());
            std::reverse(d.begin(), d.end());
        }

        radii_km_ = r;
        densities_gcm3_ = d;

        coslimits_.clear();
        for (size_t i = 0; i < radii_km_.size(); ++i) {
            double x = -std::sqrt(1.0 - (radii_km_[i] * radii_km_[i]
                                         / (Constants::REarth() * Constants::REarth())));
            if (i == 0) x = 0.0;
            coslimits_.push_back(x);
        }
    }

    __host__ __device__
    double density(int layer, int maxLayer) const {
        if (layer == 0) return 0.0; // atmosphere
        int i;
        if (layer <= maxLayer) i = layer - 1;
        else i = 2 * maxLayer - layer - 1;
        return densities_device_[i];
    }

    __host__ __device__
    double layerChordLength(int layer, int maxLayer, double pathLength,
                            double totalEarthLength, double cosZ) const {
        if (cosZ >= 0.0) return pathLength;
        if (layer == 0) return pathLength - totalEarthLength;

        int i;
        if (layer >= maxLayer) i = -layer - 1 + 2 * maxLayer;
        else i = layer - 1;

        const double RE = Constants::REarth();
        const double r_i   = radii_device_[i];
        const double r_i1  = radii_device_[i + 1];
        const double h2    = RE * RE * (1.0 - cosZ * cosZ);

        const double CrossThis = 2.0 * std::sqrt(r_i * r_i - h2);
        const double CrossNext = 2.0 * std::sqrt(r_i1 * r_i1 - h2);

        if (i < maxLayer - 1)
            return 0.5 * (CrossThis - CrossNext) * Constants::km2cm();
        else
            return CrossThis * Constants::km2cm();
    }

    __host__ __device__
    int maxLayersForCosine(double cosZ) const {
        int maxLayer = 0;
        for (size_t i = 0; i < n_layers_; ++i) {
            if (cosZ < coslimits_device_[i]) ++maxLayer;
            else break;
        }
        return maxLayer;
    }

    int nLayers() const { return static_cast<int>(radii_km_.size()); }
    const std::vector<double>& radii() const { return radii_km_; }
    const std::vector<double>& densities() const { return densities_gcm3_; }
    const std::vector<double>& coslimits() const { return coslimits_; }

    void setDevicePointers(const double* radii_dev, const double* densities_dev,
                           const double* coslimits_dev, int n_layers) {
        radii_device_ = radii_dev;
        densities_device_ = densities_dev;
        coslimits_device_ = coslimits_dev;
        n_layers_ = n_layers;
    }

private:
    std::vector<double> radii_km_;
    std::vector<double> densities_gcm3_;
    std::vector<double> coslimits_;

    // Device-side pointers (set before kernel launch)
    const double* radii_device_ = nullptr;
    const double* densities_device_ = nullptr;
    const double* coslimits_device_ = nullptr;
    int n_layers_ = 0;
};

} // namespace cudaprob3

#endif
