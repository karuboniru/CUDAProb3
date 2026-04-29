#pragma once

#include <span>

namespace cudaprob3 {

enum class ProbType : int {
    e_e = 0, e_m = 1, e_t = 2,
    m_e = 3, m_m = 4, m_t = 5,
    t_e = 6, t_m = 7, t_t = 8
};

enum class NeutrinoType { Neutrino, Antineutrino };

// Non-owning view over a flat result buffer, templated on floating-point type T.
// Buffer layout: [flavor_pair * n_cosines * n_energies + icos * n_energies + ie]
template<typename T = double>
struct ResultView {
    std::span<const T> data{};
    int n_cosines = 0;
    int n_energies = 0;

    [[nodiscard]] T probability(int icos, int ie, ProbType t) const noexcept {
        return data[static_cast<int>(t) * n_cosines * n_energies
                    + icos * n_energies + ie];
    }

    // All (cosine, energy) values for one flavor-pair channel.
    [[nodiscard]] std::span<const T> channel(ProbType t) const noexcept {
        const std::size_t off = static_cast<std::size_t>(static_cast<int>(t))
                              * static_cast<std::size_t>(n_cosines)
                              * static_cast<std::size_t>(n_energies);
        return data.subspan(off, static_cast<std::size_t>(n_cosines) * static_cast<std::size_t>(n_energies));
    }
};

} // namespace cudaprob3
