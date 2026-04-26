#ifndef CUDAPROB3_PHYSICS_MIXING_HPP
#define CUDAPROB3_PHYSICS_MIXING_HPP

#include <cmath>

#include "cudaprob3/core/complex.hpp"
#include "cudaprob3/core/matrix3x3.hpp"
#include "cudaprob3/core/mdwrap.hpp"

namespace cudaprob3 {

class MixingParams {
public:
    double theta12, theta13, theta23, dCP;

    MixingParams() : theta12(0), theta13(0), theta23(0), dCP(0) {}
    MixingParams(double th12, double th13, double th23, double dcp)
        : theta12(th12), theta13(th13), theta23(th23), dCP(dcp) {}
};

class MixingMatrix {
public:
    __host__ __device__ MixingMatrix() { setIdentity(); }

    __host__ __device__ explicit MixingMatrix(const MixingParams& p) {
        setFromMNS(p.theta12, p.theta13, p.theta23, p.dCP);
    }

    __host__ __device__ void setFromMNS(double theta12, double theta13, double theta23, double dCP) {
        const double s12 = std::sin(theta12), s13 = std::sin(theta13), s23 = std::sin(theta23);
        const double c12 = std::cos(theta12), c13 = std::cos(theta13), c23 = std::cos(theta23);
        const double sd = std::sin(dCP), cd = std::cos(dCP);

        U_(0,0).re =  c12 * c13;                   U_(0,0).im = 0.0;
        U_(0,1).re =  s12 * c13;                   U_(0,1).im = 0.0;
        U_(0,2).re =  s13 * cd;                    U_(0,2).im = -s13 * sd;
        U_(1,0).re = -s12 * c23 - c12 * s23 * s13 * cd;  U_(1,0).im = -c12 * s23 * s13 * sd;
        U_(1,1).re =  c12 * c23 - s12 * s23 * s13 * cd;  U_(1,1).im = -s12 * s23 * s13 * sd;
        U_(1,2).re =  s23 * c13;                   U_(1,2).im = 0.0;
        U_(2,0).re =  s12 * s23 - c12 * c23 * s13 * cd;  U_(2,0).im = -c12 * c23 * s13 * sd;
        U_(2,1).re = -c12 * s23 - s12 * c23 * s13 * cd;  U_(2,1).im = -s12 * c23 * s13 * sd;
        U_(2,2).re =  c23 * c13;                   U_(2,2).im = 0.0;

        computeAXFactors();
    }

    const Matrix3x3<double>& matrix() const { return U_; }
    __host__ __device__ MatrixElement<double> operator()(int i, int j) const { return U_(i, j); }

    __host__ __device__ MDView5D<double> axfac_view() { return MDView5D<double>(ax_factors_, 108, 36, 12, 4); }
    __host__ __device__ MDView5D<const double> axfac_view() const { return MDView5D<const double>(ax_factors_, 108, 36, 12, 4); }
    __host__ __device__ const double* axfac_data() const { return ax_factors_; }
    __host__ __device__ void set_axfac_from_device(const double* src) {
        for (int i = 0; i < 324; ++i) ax_factors_[i] = src[i];
    }

    Matrix3x3<double> U_;
    double ax_factors_[324];

    __host__ __device__ void setIdentity() {
        U_.setIdentity();
        for (int i = 0; i < 324; ++i) ax_factors_[i] = 0.0;
    }

    void computeAXFactors() {
        auto ax = axfac_view();
        for (int n = 0; n < 3; ++n)
            for (int m = 0; m < 3; ++m)
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        double re1 = U_(n,i).re, im1 = U_(n,i).im;
                        double re2 = U_(m,j).re, im2 = U_(m,j).im;
                        ax(n,m,i,j,0) = re1 * re2 + im1 * im2;
                        ax(n,m,i,j,1) = re1 * im2 - im1 * re2;
                        ax(n,m,i,j,2) = im1 * im2 + re1 * re2;
                        ax(n,m,i,j,3) = im1 * re2 - re1 * im2;
                    }
    }
};

} // namespace cudaprob3

#endif
