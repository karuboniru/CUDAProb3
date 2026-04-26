#ifndef CUDAPROB3_PHYSICS_AMPLITUDES_HPP
#define CUDAPROB3_PHYSICS_AMPLITUDES_HPP

#include <cmath>

#include "cudaprob3/core/constants.hpp"
#include "cudaprob3/core/complex.hpp"
#include "cudaprob3/core/matrix3x3.hpp"
#include "cudaprob3/core/types.hpp"
#include "cudaprob3/physics/mixing.hpp"
#include "cudaprob3/physics/matter.hpp"

namespace cudaprob3 {

class TransitionAmplitude {
public:
    __host__ __device__ TransitionAmplitude(const MixingMatrix& mixing, const MassOrdering& massOrdering)
        : mixing_(&mixing), matter_(mixing, massOrdering) {}

    __host__ __device__
    void compute(double L, double E, double rho, NeutrinoType type,
                 Matrix3x3<double>& A_out, double phase_offset = 0.0) const {
        double dmMatMat[3][3], dmMatVac[3][3];
        matter_.computeEffectiveMasses(E, rho, type, dmMatMat, dmMatVac);
        computeA(L, E, rho, dmMatVac, dmMatMat, type, A_out, phase_offset);
    }

private:
    __host__ __device__
    void compute_product(double L, double E, double rho,
                         const double dmMatVac[3][3],
                         const double dmMatMat[3][3],
                         NeutrinoType type,
                         MatrixElement<double> product[3][3][3]) const {
        const double fac = (type == NeutrinoType::Antineutrino)
            ? Constants::tworttwoGf() * E * rho
            : -Constants::tworttwoGf() * E * rho;

        auto U = [&](int r, int c) { return mixing_->operator()(r, c); };

        MatrixElement<double> twoEHmM[3][3][3];

        #pragma unroll
        for (int n = 0; n < 3; ++n) {
            #pragma unroll
            for (int m = 0; m < 3; ++m) {
                twoEHmM[n][m][0].re = -fac * (U(0,n).re * U(0,m).re + U(0,n).im * U(0,m).im);
                twoEHmM[n][m][0].im = -fac * (U(0,n).re * U(0,m).im - U(0,n).im * U(0,m).re);
                twoEHmM[n][m][1] = twoEHmM[n][m][0];
                twoEHmM[n][m][2] = twoEHmM[n][m][0];
            }
        }

        #pragma unroll
        for (int j = 0; j < 3; ++j) {
            twoEHmM[0][0][j].re -= dmMatVac[j][0];
            twoEHmM[1][1][j].re -= dmMatVac[j][1];
            twoEHmM[2][2][j].re -= dmMatVac[j][2];
        }

        #pragma unroll
        for (int i = 0; i < 3; ++i)
            #pragma unroll
            for (int j = 0; j < 3; ++j)
                #pragma unroll
                for (int k = 0; k < 3; ++k)
                    product[i][j][k].re = product[i][j][k].im = 0.0;

        #pragma unroll
        for (int i = 0; i < 3; ++i)
            #pragma unroll
            for (int j = 0; j < 3; ++j) {
                #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    product[i][j][0].re += twoEHmM[i][k][1].re * twoEHmM[k][j][2].re
                                         - twoEHmM[i][k][1].im * twoEHmM[k][j][2].im;
                    product[i][j][0].im += twoEHmM[i][k][1].re * twoEHmM[k][j][2].im
                                         + twoEHmM[i][k][1].im * twoEHmM[k][j][2].re;
                    product[i][j][1].re += twoEHmM[i][k][2].re * twoEHmM[k][j][0].re
                                         - twoEHmM[i][k][2].im * twoEHmM[k][j][0].im;
                    product[i][j][1].im += twoEHmM[i][k][2].re * twoEHmM[k][j][0].im
                                         + twoEHmM[i][k][2].im * twoEHmM[k][j][0].re;
                    product[i][j][2].re += twoEHmM[i][k][0].re * twoEHmM[k][j][1].re
                                         - twoEHmM[i][k][0].im * twoEHmM[k][j][1].im;
                    product[i][j][2].im += twoEHmM[i][k][0].re * twoEHmM[k][j][1].im
                                         + twoEHmM[i][k][0].im * twoEHmM[k][j][1].re;
                }
                product[i][j][0].re /= (dmMatMat[0][1] * dmMatMat[0][2]);
                product[i][j][0].im /= (dmMatMat[0][1] * dmMatMat[0][2]);
                product[i][j][1].re /= (dmMatMat[1][2] * dmMatMat[1][0]);
                product[i][j][1].im /= (dmMatMat[1][2] * dmMatMat[1][0]);
                product[i][j][2].re /= (dmMatMat[2][0] * dmMatMat[2][1]);
                product[i][j][2].im /= (dmMatMat[2][0] * dmMatMat[2][1]);
            }
    }

    __host__ __device__
    void computeA(double L, double E, double rho,
                  const double dmMatVac[3][3],
                  const double dmMatMat[3][3],
                  NeutrinoType type,
                  Matrix3x3<double>& A_out,
                  double phase_offset) const {
        MatrixElement<double> X[3][3];
        MatrixElement<double> product[3][3][3];

        compute_product(L, E, rho, dmMatVac, dmMatMat, type, product);

        #pragma unroll
        for (int i = 0; i < 3; ++i)
            #pragma unroll
            for (int j = 0; j < 3; ++j)
                X[i][j].re = X[i][j].im = 0.0;

        const double LoEfac = Constants::LoEfac();

        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            double arg = (k == 2)
                ? -LoEfac * dmMatVac[k][0] * L / E + phase_offset
                : -LoEfac * dmMatVac[k][0] * L / E;

            double s, c;
            sincos(arg, &s, &c);

            #pragma unroll
            for (int i = 0; i < 3; ++i)
                #pragma unroll
                for (int j = 0; j < 3; ++j) {
                    X[i][j].re += c * product[i][j][k].re - s * product[i][j][k].im;
                    X[i][j].im += c * product[i][j][k].im + s * product[i][j][k].re;
                }
        }

        A_out.setZero();
        auto ax = mixing_->axfac_view();

        for (int n = 0; n < 3; ++n)
            for (int m = 0; m < 3; ++m)
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        A_out(n,m).re += ax(n,m,i,j,0) * X[i][j].re + ax(n,m,i,j,1) * X[i][j].im;
                        A_out(n,m).im += ax(n,m,i,j,2) * X[i][j].im + ax(n,m,i,j,3) * X[i][j].re;
                    }
    }

    const MixingMatrix* mixing_;
    MatterEffects matter_;
};

} // namespace cudaprob3

#endif
