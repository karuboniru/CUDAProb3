#ifndef CUDAPROB3_PHYSICS_MATTER_HPP
#define CUDAPROB3_PHYSICS_MATTER_HPP

#include <cmath>
#include <cstring>

#include "cudaprob3/core/constants.hpp"
#include "cudaprob3/core/complex.hpp"
#include "cudaprob3/core/types.hpp"
#include "cudaprob3/core/mdwrap.hpp"
#include "cudaprob3/physics/mixing.hpp"

namespace cudaprob3 {

class MassOrdering {
public:
    __host__ __device__ MassOrdering() {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) dm_[i][j] = 0.0;
            order_[i] = i;
        }
        for (int i = 0; i < 9; ++i) flat_dm_[i] = 0.0;
    }

    void setMassDifferences(double dm12sq, double dm23sq) {
        double mVac[3] = {0.0, dm12sq, dm12sq + dm23sq};
        const double delta = 5.0e-9;
        if (dm12sq == 0.0) mVac[0] -= delta;
        if (dm23sq == 0.0) mVac[2] += delta;

        auto DM = [this](int i, int j) -> double& { return flat_dm_[i * 3 + j]; };
        DM(0,0) = DM(1,1) = DM(2,2) = 0.0;
        DM(0,1) = mVac[0] - mVac[1]; DM(1,0) = -DM(0,1);
        DM(0,2) = mVac[0] - mVac[2]; DM(2,0) = -DM(0,2);
        DM(1,2) = mVac[1] - mVac[2]; DM(2,1) = -DM(1,2);

        // Copy to dm_[][] form
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                dm_[i][j] = flat_dm_[i * 3 + j];

        computeOrder();
    }

    __host__ __device__ double dm(int i, int j) const { return dm_[i][j]; }
    __host__ __device__ int    order(int i) const { return order_[i]; }

    __host__ __device__ MDView2D<const double> dm_view() const {
        return MDView2D<const double>(reinterpret_cast<const double*>(dm_), 3);
    }

    double flat_dm_[9];
    int order_[3];
    double dm_[3][3];

    void copy_to_device_from(const double* dev_dm, const int* dev_order) {
        // These are already on device; just set the pointers
        for (int ii = 0; ii < 9; ++ii) flat_dm_[ii] = dev_dm[ii];
        for (int ii = 0; ii < 3; ++ii) order_[ii] = dev_order[ii];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                dm_[i][j] = flat_dm_[i * 3 + j];
    }

private:
    void computeOrder() {
        auto DM = [this](int i, int j) { return flat_dm_[i * 3 + j]; };

        double alphaV = DM(0,1) + DM(0,2);
        double betaV  = DM(0,1) * DM(0,2);
        double gammaV = 0.0;

        double tmpV = alphaV * alphaV - 3.0 * betaV;
        if (tmpV <= 0.0) tmpV = 0.0;

        double argV = (2.0 * alphaV * alphaV * alphaV - 9.0 * alphaV * betaV
                       + 27.0 * gammaV) / (2.0 * std::sqrt(tmpV * tmpV * tmpV));
        if (std::fabs(argV) > 1.0) argV = argV / std::fabs(argV);

        double theta0V = std::acos(argV) / 3.0;
        double theta1V = theta0V - (2.0 * M_PI / 3.0);
        double theta2V = theta0V + (2.0 * M_PI / 3.0);

        double mMatV[3];
        double factor = -(2.0 / 3.0) * std::sqrt(tmpV);
        mMatV[0] = factor * std::cos(theta0V);
        mMatV[1] = factor * std::cos(theta1V);
        mMatV[2] = factor * std::cos(theta2V);
        tmpV = DM(0,0) - alphaV / 3.0;
        mMatV[0] += tmpV; mMatV[1] += tmpV; mMatV[2] += tmpV;

        for (int i = 0; i < 3; ++i) {
            double best = std::fabs(DM(i, 0) - mMatV[0]);
            int k = 0;
            for (int j = 1; j < 3; ++j) {
                double d = std::fabs(DM(i, 0) - mMatV[j]);
                if (d < best) { best = d; k = j; }
            }
            order_[i] = k;
        }
    }
};

class MatterEffects {
public:
    __host__ __device__ MatterEffects(const MixingMatrix& mixing, const MassOrdering& massOrdering)
        : mixing_(&mixing), massOrdering_(&massOrdering) {}

    __host__ __device__
    void computeEffectiveMasses(double Enu, double rho, NeutrinoType type,
                                double dmMatMat[3][3], double dmMatVac[3][3]) const {
        const double fac = (type == NeutrinoType::Antineutrino)
            ? Constants::tworttwoGf() * Enu * rho
            : -Constants::tworttwoGf() * Enu * rho;

        auto U  = [&](int r, int c) { return mixing_->operator()(r, c); };

        const double DM01 = massOrdering_->dm(0,1);
        const double DM02 = massOrdering_->dm(0,2);
        const double DM00 = massOrdering_->dm(0,0);

        const double alpha = fac + DM01 + DM02;
        const double beta  = DM01 * DM02
                           + fac * (DM01 * (1.0 - U(0,1).re * U(0,1).re
                                               - U(0,1).im * U(0,1).im)
                                  + DM02 * (1.0 - U(0,2).re * U(0,2).re
                                               - U(0,2).im * U(0,2).im));
        const double gamma = fac * DM01 * DM02
                           * (U(0,0).re * U(0,0).re + U(0,0).im * U(0,0).im);

        const double tmp = (alpha * alpha - 3.0 * beta < 0)
                         ? 0.0 : alpha * alpha - 3.0 * beta;

        const double argtmp = (2.0 * alpha * alpha * alpha - 9.0 * alpha * beta
                               + 27.0 * gamma) / (2.0 * std::sqrt(tmp * tmp * tmp));
        const double arg = (std::fabs(argtmp) > 1.0) ? argtmp / std::fabs(argtmp) : argtmp;

        const double theta0 = std::acos(arg) / 3.0;
        const double theta1 = theta0 - (2.0 * M_PI / 3.0);
        const double theta2 = theta0 + (2.0 * M_PI / 3.0);

        double mMatU[3];
        const double srt = -(2.0 / 3.0) * std::sqrt(tmp);
        mMatU[0] = srt * std::cos(theta0);
        mMatU[1] = srt * std::cos(theta1);
        mMatU[2] = srt * std::cos(theta2);
        const double tmp2 = DM00 - alpha / 3.0;
        mMatU[0] += tmp2; mMatU[1] += tmp2; mMatU[2] += tmp2;

        double mMat[3];
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            mMat[i] = mMatU[massOrdering_->order(i)];
        }

        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            #pragma unroll
            for (int j = 0; j < 3; ++j) {
                dmMatMat[i][j] = mMat[i] - mMat[j];
                dmMatVac[i][j] = mMat[i] - massOrdering_->dm(j, 0);
            }
        }
    }

private:
    const MixingMatrix* mixing_;
    const MassOrdering* massOrdering_;
};

} // namespace cudaprob3

#endif
