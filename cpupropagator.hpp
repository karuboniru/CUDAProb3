/*
This file is part of CUDAProb3++.

CUDAProb3++ is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CUDAProb3++ is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with CUDAProb3++.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CUDAPROB3_CPUPROPAGATOR_HPP
#define CUDAPROB3_CPUPROPAGATOR_HPP

#include "constants.hpp"
#include "propagator.hpp"

#include "physics.hpp"

#include <omp.h>
#include <vector>

namespace cudaprob3 {

/// \class CpuPropagator
/// \brief Multi-threaded CPU neutrino propagation. Derived from Propagator
/// @param FLOAT_T The floating point type to use for calculations, i.e float,
/// double
template <class FLOAT_T> class CpuPropagator : public Propagator<FLOAT_T> {
public:
  /// \brief Constructor
  ///
  /// @param n_cosines Number cosine bins
  /// @param n_energies Number of energy bins
  /// @param threads Number of threads
  CpuPropagator(int n_cosines, int n_energies, int threads,
                size_t costh_rebin_factor = 50)
      : Propagator<FLOAT_T>(n_cosines, n_energies, costh_rebin_factor) {

    resultList.resize(std::uint64_t(n_cosines) * std::uint64_t(n_energies) *
                      std::uint64_t(9));
    resultList_costh_rebined.resize(
        std::uint64_t(n_cosines) * std::uint64_t(n_energies) *
        std::uint64_t(9) / this->rebin_factor_costh);

    omp_set_num_threads(threads);
  }

  /// \brief Copy constructor
  /// @param other
  CpuPropagator(const CpuPropagator &other) : Propagator<FLOAT_T>(other) {
    *this = other;
  }

  /// \brief Move constructor
  /// @param other
  CpuPropagator(CpuPropagator &&other) noexcept : Propagator<FLOAT_T>(other) {
    *this = std::move(other);
  }

  /// \brief Copy assignment operator
  /// @param other
  CpuPropagator &operator=(const CpuPropagator &other) {
    Propagator<FLOAT_T>::operator=(other);

    resultList = other.resultList;
    resultList_costh_rebined = other.resultList_costh_rebined;

    return *this;
  }

  /// \brief Move assignment operator
  /// @param other
  CpuPropagator &operator=(CpuPropagator &&other) {
    Propagator<FLOAT_T>::operator=(std::move(other));

    resultList = std::move(other.resultList);
    resultList_costh_rebined = std::move(other.resultList_costh_rebined);

    return *this;
  }

public:
  void calculateProbabilities(NeutrinoType type) override {
    if (!this->isInit)
      throw std::runtime_error(
          "CpuPropagator::calculateProbabilities. Object has been moved from.");
    if (!this->isSetProductionHeight)
      throw std::runtime_error("CpuPropagator::calculateProbabilities. "
                               "production height was not set");

    // set neutrino parameters for core physics functions
    physics::setMixMatrix_host(this->Mix_U.data());
    physics::setMassDifferences_host(this->dm.data());
    memset(resultList_costh_rebined.data(), 0,
           resultList_costh_rebined.size() * sizeof(FLOAT_T));
    physics::calculate(type, this->cosineList.data(), this->cosineList.size(),
                       this->energyList.data(), this->energyList.size(),
                       this->radii.data(), this->rhos.data(),
                       this->maxlayers.data(),
                       this->ProductionHeightinCentimeter, resultList.data());
    resultList_costh_rebined_init = false;
  }

  FLOAT_T getProbability(int index_cosine, int index_energy,
                         ProbType t) override {
    std::uint64_t index = std::uint64_t(index_cosine) *
                              std::uint64_t(this->n_energies) *
                              std::uint64_t(9) +
                          std::uint64_t(index_energy) * std::uint64_t(9);
    return resultList[index + int(t)];
  }

  FLOAT_T getProbabilityRebin(int index_cosine, int index_energy, ProbType t) {
    // const auto index_cosine_rebined = index_cosine;
    const auto n_cosines_rebined = this->n_cosines / this->rebin_factor_costh;
    if (!resultList_costh_rebined_init) [[unlikely]] {
      memset(resultList_costh_rebined.data(), 0,
             resultList_costh_rebined.size() * sizeof(FLOAT_T));
      for (int i = 0; i < this->n_cosines; i++) {
        auto i_rebined = i / this->rebin_factor_costh;
        for (int j = 0; j < this->n_energies; j++) {
          for (int k = 0; k < 9; k++) {
            resultList_costh_rebined[std::uint64_t(i_rebined) *
                                         std::uint64_t(this->n_energies) *
                                         std::uint64_t(9) +
                                     std::uint64_t(j) * std::uint64_t(9) + k] +=
                resultList[std::uint64_t(i) * std::uint64_t(this->n_energies) *
                               std::uint64_t(9) +
                           std::uint64_t(j) * std::uint64_t(9) + k] / this->rebin_factor_costh;
          }
        }
      }
      resultList_costh_rebined_init = true;
    }
    return resultList_costh_rebined
        [std::uint64_t(index_cosine) * std::uint64_t(this->n_energies) *
             std::uint64_t(9) +
         std::uint64_t(index_energy) * std::uint64_t(9) + int(t)];
  }

private:
  std::vector<FLOAT_T> resultList;
  std::vector<FLOAT_T> resultList_costh_rebined;
  bool resultList_costh_rebined_init{false};
};

} // namespace cudaprob3

#endif
