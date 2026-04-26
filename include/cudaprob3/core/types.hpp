#ifndef CUDAPROB3_CORE_TYPES_HPP
#define CUDAPROB3_CORE_TYPES_HPP

namespace cudaprob3 {

enum class ProbType : int {
    e_e = 0, e_m = 1, e_t = 2,
    m_e = 3, m_m = 4, m_t = 5,
    t_e = 6, t_m = 7, t_t = 8
};

enum class NeutrinoType : int {
    Neutrino = 0,
    Antineutrino = 1
};

} // namespace cudaprob3

#endif
