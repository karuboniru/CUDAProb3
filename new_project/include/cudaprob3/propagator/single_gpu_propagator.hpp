#ifndef CUDAPROB3_PROPAGATOR_SINGLE_GPU_PROPAGATOR_HPP
#define CUDAPROB3_PROPAGATOR_SINGLE_GPU_PROPAGATOR_HPP

#include <cmath>
#include <memory>
#include <vector>

#include "cudaprob3/propagator/propagator.hpp"
#include "cudaprob3/core/matrix3x3.hpp"
#include "cudaprob3/core/constants.hpp"
#include "cudaprob3/core/cuda_helpers.hpp"
#include "cudaprob3/engine/stream_pool.hpp"
#include "cudaprob3/engine/memory_pool.hpp"
#include "cudaprob3/physics/mixing.hpp"
#include "cudaprob3/physics/matter.hpp"
#include "cudaprob3/physics/amplitudes.hpp"
#include "cudaprob3/geometry/earth_model.hpp"

namespace cudaprob3 {

// Free __global__ kernel (CUDA forbids member __global__ functions)
template <typename FLOAT_T>
__global__ __launch_bounds__(64, 8)
void calculateProbabilitiesKernel(
    NeutrinoType type,
    const FLOAT_T* cosinelist, int nCosines,
    const FLOAT_T* energylist, int nEnergies,
    const int* maxLayers, FLOAT_T prodHeightCm,
    const FLOAT_T* radii, const FLOAT_T* rhos,
    const FLOAT_T* coslimits, int nEarthLayers,
    const MatrixElement<double>* d_mixU,
    const double* d_axfac,
    const double* d_dm, const int* d_order,
    FLOAT_T* result)
{
    const int maxEnergiesPerPath = ((nEnergies + blockDim.x - 1) / blockDim.x) * blockDim.x;

    for (unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
         index < static_cast<unsigned>(nCosines * maxEnergiesPerPath);
         index += blockDim.x * gridDim.x)
    {
        const unsigned idxEnergy  = index % maxEnergiesPerPath;
        const unsigned idxCosine  = index / maxEnergiesPerPath;
        if (idxEnergy >= static_cast<unsigned>(nEnergies)) continue;

        FLOAT_T cosZ   = cosinelist[idxCosine];
        FLOAT_T energy = energylist[idxEnergy];
        int maxLayer   = maxLayers[idxCosine];

        const FLOAT_T REcm = Constants::REarthcm();
        const FLOAT_T RE   = Constants::REarth();
        const FLOAT_T detectorHeight = REcm + prodHeightCm;

        FLOAT_T pathLength = sqrt(detectorHeight * detectorHeight
                                  - REcm * REcm * (1.0 - cosZ * cosZ))
                           - REcm * cosZ;
        FLOAT_T totalEarthLength = -2.0 * cosZ * REcm;

        // Setup MixingMatrix from device pointers
        MixingMatrix mix;
        for (int ii = 0; ii < 9; ++ii) mix.U_.data[ii] = d_mixU[ii];
        mix.set_axfac_from_device(d_axfac);

        // Setup MassOrdering from device pointers
        MassOrdering massOrd;
        for (int ii = 0; ii < 9; ++ii) massOrd.flat_dm_[ii] = d_dm[ii];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                massOrd.dm_[i][j] = d_dm[i*3+j];
        for (int ii = 0; ii < 3; ++ii) massOrd.order_[ii] = d_order[ii];

        Matrix3x3<FLOAT_T> acc;
        acc.setIdentity();
        Matrix3x3<FLOAT_T> coreToMantle;
        coreToMantle.setIdentity();
        Matrix3x3<FLOAT_T> layerA, temp;

        TransitionAmplitude amplitude(mix, massOrd);

        for (int layer = 0; layer <= maxLayer; ++layer) {
            FLOAT_T distance, density;

            // Chord length
            if (cosZ >= 0.0) {
                distance = pathLength;
            } else if (layer == 0) {
                distance = pathLength - totalEarthLength;
            } else {
                int i = (layer >= maxLayer) ? -layer - 1 + 2 * maxLayer : layer - 1;
                FLOAT_T r_i = radii[i], r_i1 = radii[i+1];
                FLOAT_T h2 = RE * RE * (1.0 - cosZ * cosZ);
                FLOAT_T crossThis = 2.0 * sqrt(r_i * r_i - h2);
                FLOAT_T crossNext = 2.0 * sqrt(r_i1 * r_i1 - h2);
                if (i < maxLayer - 1)
                    distance = 0.5 * (crossThis - crossNext) * Constants::km2cm();
                else
                    distance = crossThis * Constants::km2cm();
            }

            // Density
            if (layer == 0) density = 0.0;
            else {
                int i = (layer <= maxLayer) ? layer - 1 : 2 * maxLayer - layer - 1;
                density = rhos[i];
            }

            amplitude.compute(
                distance / Constants::km2cm(), energy,
                density * Constants::density_convert(),
                type, layerA, 0.0);

            if (layer == 0) {
                copy_complex_matrix(layerA, acc);
            } else if (layer < maxLayer) {
                multiply_complex_matrix(layerA, acc, temp);
                copy_complex_matrix(temp, acc);
                multiply_complex_matrix(coreToMantle, layerA, temp);
                copy_complex_matrix(temp, coreToMantle);
            } else {
                multiply_complex_matrix(layerA, acc, temp);
                copy_complex_matrix(temp, acc);
            }
        }

        multiply_complex_matrix(coreToMantle, acc, temp);
        copy_complex_matrix(temp, acc);

        // Probabilities = |A_{ij}|^2
        for (int inflv = 0; inflv < 3; ++inflv) {
            for (int outflv = 0; outflv < 3; ++outflv) {
                FLOAT_T re = acc(inflv, outflv).re;
                FLOAT_T im = acc(inflv, outflv).im;
                unsigned long long resultIdx =
                    static_cast<unsigned long long>(nEnergies) * idxCosine + idxEnergy;
                result[resultIdx +
                    static_cast<unsigned long long>(nEnergies) * nCosines *
                    static_cast<unsigned long long>(inflv * 3 + outflv)] = re * re + im * im;
            }
        }
    }
}

template <typename FLOAT_T>
class SingleGPUPropagator : public Propagator {
public:
    SingleGPUPropagator(int deviceId, int nCosines, int nEnergies)
        : deviceId_(deviceId), nCosines_(nCosines), nEnergies_(nEnergies)
    {
        cuda_check_device(deviceId_);

        d_cosines_.allocate(deviceId_, nCosines_);
        d_energies_.allocate(deviceId_, nEnergies_);
        d_maxLayers_.allocate(deviceId_, nCosines_);

        size_t resultSize = static_cast<size_t>(nCosines_) * nEnergies_ * 9;
        d_results_.allocate(deviceId_, resultSize);
        h_results_.allocate(resultSize);

        CUDA_CHECK(cudaMalloc(&d_mixing_U_, 9 * sizeof(MatrixElement<double>)));
        CUDA_CHECK(cudaMalloc(&d_mixing_axfac_, 324 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_dm_, 9 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_order_, 3 * sizeof(int)));
    }

    ~SingleGPUPropagator() override {
        CUDA_CHECK(cudaSetDevice(deviceId_));
        if (d_mixing_U_)     CUDA_CHECK(cudaFree(d_mixing_U_));
        if (d_mixing_axfac_) CUDA_CHECK(cudaFree(d_mixing_axfac_));
        if (d_dm_)           CUDA_CHECK(cudaFree(d_dm_));
        if (d_order_)        CUDA_CHECK(cudaFree(d_order_));
        if (d_radii_)        CUDA_CHECK(cudaFree(d_radii_));
        if (d_rhos_)         CUDA_CHECK(cudaFree(d_rhos_));
        if (d_coslimits_)    CUDA_CHECK(cudaFree(d_coslimits_));
    }

    void configure(const PropagatorConfig& config) override {
        nCosines_ = static_cast<int>(config.cosines.size());
        nEnergies_ = static_cast<int>(config.energies.size());

        energies_   = config.energies;
        cosines_    = config.cosines;
        prodHeight_ = config.productionHeightKm * 1e5;

        mixing_ = std::make_unique<MixingMatrix>(MixingParams(
            config.mixing.theta12, config.mixing.theta13,
            config.mixing.theta23, config.mixing.dCP));

        massOrdering_ = std::make_unique<MassOrdering>();
        massOrdering_->setMassDifferences(config.dm12sq, config.dm23sq);

        earthModel_ = config.earthModel;

        maxLayers_.resize(nCosines_);
        const auto& coslimits = earthModel_.coslimits();
        for (int ic = 0; ic < nCosines_; ++ic)
            maxLayers_[ic] = std::count_if(coslimits.begin(), coslimits.end(),
                [&](double lim) { return config.cosines[ic] < lim; });

        copyToDevice();
    }

    void calculate(NeutrinoType type) override {
        CUDA_CHECK(cudaSetDevice(deviceId_));

        dim3 block(64);
        dim3 grid((nEnergies_ + 63) / 64 * nCosines_);

        calculateProbabilitiesKernel<FLOAT_T><<<grid, block>>>(
            type,
            d_cosines_.get(), nCosines_,
            d_energies_.get(), nEnergies_,
            d_maxLayers_.get(), static_cast<FLOAT_T>(prodHeight_),
            d_radii_, d_rhos_, d_coslimits_,
            static_cast<int>(earthModel_.nLayers()),
            d_mixing_U_, d_mixing_axfac_,
            d_dm_, d_order_,
            d_results_.get());

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        resultsResideOnHost_ = false;
    }

    double getProbability(int index_cosine, int index_energy,
                          ProbType t) const override {
        if (!resultsResideOnHost_)
            const_cast<SingleGPUPropagator*>(this)->copyResultsToHost();

        size_t idx = static_cast<size_t>(index_cosine) * nEnergies_ + index_energy;
        size_t off = static_cast<size_t>(static_cast<int>(t)) * nCosines_ * nEnergies_;
        return h_results_[idx + off];
    }

    int nCosines() const override  { return nCosines_; }
    int nEnergies() const override { return nEnergies_; }
    int deviceId() const { return deviceId_; }

    SingleGPUPropagator(const SingleGPUPropagator&) = delete;
    SingleGPUPropagator& operator=(const SingleGPUPropagator&) = delete;

private:
    void copyResultsToHost() {
        CUDA_CHECK(cudaSetDevice(deviceId_));
        CUDA_CHECK(cudaMemcpy(h_results_.get(), d_results_.get(),
                               nCosines_ * nEnergies_ * 9 * sizeof(FLOAT_T),
                               cudaMemcpyDeviceToHost));
        resultsResideOnHost_ = true;
    }

    void copyToDevice() {
        CUDA_CHECK(cudaSetDevice(deviceId_));

        CUDA_CHECK(cudaMemcpy(d_cosines_.get(), cosines_.data(),
                               nCosines_ * sizeof(FLOAT_T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_energies_.get(), energies_.data(),
                               nEnergies_ * sizeof(FLOAT_T), cudaMemcpyHostToDevice));

        std::vector<int> h_maxLayers(nCosines_);
        for (int i = 0; i < nCosines_; ++i) h_maxLayers[i] = maxLayers_[i];
        CUDA_CHECK(cudaMemcpy(d_maxLayers_.get(), h_maxLayers.data(),
                               nCosines_ * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_mixing_U_, mixing_->matrix().data,
                               9 * sizeof(MatrixElement<double>), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mixing_axfac_, mixing_->axfac_data(),
                               324 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dm_, massOrdering_->flat_dm_,
                               9 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_order_, massOrdering_->order_,
                               3 * sizeof(int), cudaMemcpyHostToDevice));

        int nLayers = earthModel_.nLayers();
        int nRadii = 2 * nLayers + 1;

        if (!d_radii_)     CUDA_CHECK(cudaMalloc(&d_radii_, nRadii * sizeof(double)));
        if (!d_rhos_)      CUDA_CHECK(cudaMalloc(&d_rhos_, nRadii * sizeof(double)));
        if (!d_coslimits_) CUDA_CHECK(cudaMalloc(&d_coslimits_, nLayers * sizeof(double)));

        CUDA_CHECK(cudaMemcpy(d_radii_, earthModel_.radii().data(),
                               nLayers * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rhos_, earthModel_.densities().data(),
                               nLayers * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coslimits_, earthModel_.coslimits().data(),
                               nLayers * sizeof(double), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int deviceId_;
    int nCosines_, nEnergies_;
    double prodHeight_ = 0.0;

    std::unique_ptr<MixingMatrix> mixing_;
    std::unique_ptr<MassOrdering> massOrdering_;
    EarthModel earthModel_;

    std::vector<double> energies_;
    std::vector<double> cosines_;
    std::vector<int> maxLayers_;

    DeviceBuffer<FLOAT_T> d_cosines_;
    DeviceBuffer<FLOAT_T> d_energies_;
    DeviceBuffer<int>      d_maxLayers_;
    DeviceBuffer<FLOAT_T> d_results_;
    PinnedBuffer<FLOAT_T> h_results_;

    MatrixElement<double>* d_mixing_U_ = nullptr;
    double* d_mixing_axfac_ = nullptr;
    double* d_dm_ = nullptr;
    int*    d_order_ = nullptr;
    double* d_radii_ = nullptr;
    double* d_rhos_ = nullptr;
    double* d_coslimits_ = nullptr;

    mutable bool resultsResideOnHost_ = false;
};

} // namespace cudaprob3

#endif
