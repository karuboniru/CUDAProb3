#pragma once

#include <algorithm>
#include <cmath>
#include <span>
#include <string>
#include <vector>

#ifndef __CUDACC__
#  include <expected>
#  include <fstream>
#  include <string_view>
#endif

namespace cudaprob3 {

namespace detail {
    inline constexpr double kREarth = 6371.0; // km
}

// Earth density model. Stores shell radii (km) in descending order (outermost first)
// and corresponding densities (g/cm³). Computes per-cosine max layer counts.
class PREMModel {
public:
#ifndef __CUDACC__
    // Factory: parse a two-column (radius km, density g/cm³) file.
    static std::expected<PREMModel, std::string> fromFile(std::string_view path) {
        std::ifstream f{std::string(path)};
        if (!f) return std::unexpected("cannot open density file: " + std::string(path));

        std::vector<double> radii, rhos;
        double r, d;
        while (f >> r >> d) { radii.push_back(r); rhos.push_back(d); }
        if (radii.empty()) return std::unexpected("density file is empty");

        return fromVectors(std::move(radii), std::move(rhos));
    }

    // Factory: from already-loaded vectors.
    static std::expected<PREMModel, std::string>
    fromVectors(std::vector<double> radii, std::vector<double> rhos) {
        if (radii.size() != rhos.size())
            return std::unexpected("radii and rhos size mismatch");
        if (radii.empty())
            return std::unexpected("density model must not be empty");
        if (radii.size() >= 2) {
            const int sign = (radii[1] > radii[0]) ? 1 : -1;
            for (std::size_t i = 1; i < radii.size(); ++i)
                if ((radii[i] - radii[i-1]) * sign < 0)
                    return std::unexpected("radii are not monotonic");
            if (sign > 0) {
                std::ranges::reverse(radii);
                std::ranges::reverse(rhos);
            }
        }
        PREMModel m;
        m.radii_ = std::move(radii);
        m.rhos_  = std::move(rhos);
        m.buildCosLimits();
        return m;
    }
#endif // !__CUDACC__

    [[nodiscard]] std::span<const double> radii()     const noexcept { return radii_; }
    [[nodiscard]] std::span<const double> densities() const noexcept { return rhos_; }
    [[nodiscard]] std::span<const double> cosLimits() const noexcept { return cosLimits_; }

    [[nodiscard]] int maxLayersForCosine(double c) const noexcept {
        return static_cast<int>(
            std::ranges::count_if(cosLimits_, [c](double lim){ return c < lim; }));
    }

    [[nodiscard]] std::vector<int> buildMaxlayers(std::span<const double> cosines) const {
        std::vector<int> out(cosines.size());
        for (std::size_t i = 0; i < cosines.size(); ++i)
            out[i] = maxLayersForCosine(cosines[i]);
        return out;
    }

private:
    PREMModel() = default;

    void buildCosLimits() {
        cosLimits_.reserve(radii_.size());
        for (std::size_t i = 0; i < radii_.size(); ++i) {
            double x = (i == 0) ? 0.0
                : -std::sqrt(1.0 - (radii_[i]*radii_[i]) / (detail::kREarth*detail::kREarth));
            cosLimits_.push_back(x);
        }
    }

    std::vector<double> radii_;
    std::vector<double> rhos_;
    std::vector<double> cosLimits_;
};

} // namespace cudaprob3
