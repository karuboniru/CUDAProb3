#pragma once

// Under nvcc, functions are __host__ __device__ for host tests or __device__ only.
// Under a plain C++ compiler (host-only unit tests), no qualifiers are needed.
#ifdef __CUDACC__
#  ifdef CUDAPROB3_HOST_PHYSICS_TEST
#    define BARGER_HD __host__ __device__
#  else
#    define BARGER_HD __device__
#  endif
#else
#  define BARGER_HD
#endif

#include "../math/complex3x3.cuh"
#include "../math/constants.cuh"
#include "params_pod.cuh"
#include "../../include/cudaprob3/types.hpp"

#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace cudaprob3 {
namespace physics {
namespace detail {

template <typename T>
BARGER_HD inline void sincos_impl(T x, T* s, T* c) noexcept {
#ifdef __CUDACC__
    sincos(x, s, c);
#else
    *s = std::sin(x);
    *c = std::cos(x);
#endif
}

} // namespace detail

// Compute matter-modified mass eigenstates and their differences.
// Fills d_dmMatMat[i][j] = mMat[i] - mMat[j]
//       d_dmMatVac[i][j] = mMat[i] - DM(j,0)
template <typename T>
BARGER_HD void getMfast(T Enu, T rho, NeutrinoType type,
                        const OscParamsPOD& p,
                        T d_dmMatMat[3][3], T d_dmMatVac[3][3]) noexcept {
    T mMatU[3], mMat[3];

    const T fac = (type == NeutrinoType::Antineutrino)
        ?  constants::tworttwoGf<T>() * Enu * rho
        : -constants::tworttwoGf<T>() * Enu * rho;

    const T alpha = fac + p.DM(0,1) + p.DM(0,2);

    const T beta = p.DM(0,1)*p.DM(0,2)
        + fac * (p.DM(0,1) * (T(1) - p.U_re(0,1)*p.U_re(0,1) - p.U_im(0,1)*p.U_im(0,1))
               + p.DM(0,2) * (T(1) - p.U_re(0,2)*p.U_re(0,2) - p.U_im(0,2)*p.U_im(0,2)));

    const T gamma = fac * p.DM(0,1) * p.DM(0,2)
        * (p.U_re(0,0)*p.U_re(0,0) + p.U_im(0,0)*p.U_im(0,0));

    const T tmp_raw = alpha*alpha - T(3)*beta;
    const T tmp = tmp_raw < T(0) ? T(0) : tmp_raw;

    const T argtmp = (T(2)*alpha*alpha*alpha - T(9)*alpha*beta + T(27)*gamma)
                   / (T(2) * sqrt(tmp*tmp*tmp));
    const T arg = (fabs(argtmp) > T(1)) ? argtmp / fabs(argtmp) : argtmp;

    const T theta0 = acos(arg) / T(3);
    const T theta1 = theta0 - T(2)*T(M_PI)/T(3);
    const T theta2 = theta0 + T(2)*T(M_PI)/T(3);

    const T base   = -(T(2)/T(3)) * sqrt(tmp);
    const T shift  = p.DM(0,0) - alpha/T(3);
    mMatU[0] = base * cos(theta0) + shift;
    mMatU[1] = base * cos(theta1) + shift;
    mMatU[2] = base * cos(theta2) + shift;

#pragma unroll
    for (int i = 0; i < 3; ++i)
        mMat[i] = mMatU[p.ORDER(i)];

#pragma unroll
    for (int i = 0; i < 3; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            d_dmMatMat[i][j] = mMat[i] - mMat[j];
            d_dmMatVac[i][j] = mMat[i] - p.DM(j, 0);
        }
}

// Eq. (11) of Barger et al. — product of (2EH - M_j) matrices divided by
// eigenvalue differences.
template <typename T>
BARGER_HD void get_product(T L, T E, T rho,
                            const T d_dmMatVac[3][3], const T d_dmMatMat[3][3],
                            NeutrinoType type, const OscParamsPOD& p,
                            T prod_re[3][3][3], T prod_im[3][3][3]) noexcept {
    const T fac = (type == NeutrinoType::Antineutrino)
        ?  constants::tworttwoGf<T>() * E * rho
        : -constants::tworttwoGf<T>() * E * rho;

    // Build twoEHmM[n][m][j] = (2EH - M_j)[n][m]
    T twoEHmM_re[3][3][3], twoEHmM_im[3][3][3];
#pragma unroll
    for (int n = 0; n < 3; ++n)
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            const T base_re = -fac * (p.U_re(0,n)*p.U_re(0,m) + p.U_im(0,n)*p.U_im(0,m));
            const T base_im = -fac * (p.U_re(0,n)*p.U_im(0,m) - p.U_im(0,n)*p.U_re(0,m));
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                twoEHmM_re[n][m][j] = base_re;
                twoEHmM_im[n][m][j] = base_im;
            }
        }

#pragma unroll
    for (int j = 0; j < 3; ++j) {
        twoEHmM_re[0][0][j] -= d_dmMatVac[j][0];
        twoEHmM_re[1][1][j] -= d_dmMatVac[j][1];
        twoEHmM_re[2][2][j] -= d_dmMatVac[j][2];
    }

    // product[n][m][k] = sum_l twoEHmM[n][l][(k+1)%3] * twoEHmM[l][m][(k+2)%3]
    // divided by eigenvalue product
#pragma unroll
    for (int n = 0; n < 3; ++n)
#pragma unroll
        for (int m = 0; m < 3; ++m)
#pragma unroll
            for (int k = 0; k < 3; ++k) {
                prod_re[n][m][k] = T(0);
                prod_im[n][m][k] = T(0);
            }

#pragma unroll
    for (int n = 0; n < 3; ++n)
#pragma unroll
        for (int m = 0; m < 3; ++m)
#pragma unroll
            for (int l = 0; l < 3; ++l) {
                prod_re[n][m][0] += twoEHmM_re[n][l][1]*twoEHmM_re[l][m][2]
                                  - twoEHmM_im[n][l][1]*twoEHmM_im[l][m][2];
                prod_im[n][m][0] += twoEHmM_re[n][l][1]*twoEHmM_im[l][m][2]
                                  + twoEHmM_im[n][l][1]*twoEHmM_re[l][m][2];

                prod_re[n][m][1] += twoEHmM_re[n][l][2]*twoEHmM_re[l][m][0]
                                  - twoEHmM_im[n][l][2]*twoEHmM_im[l][m][0];
                prod_im[n][m][1] += twoEHmM_re[n][l][2]*twoEHmM_im[l][m][0]
                                  + twoEHmM_im[n][l][2]*twoEHmM_re[l][m][0];

                prod_re[n][m][2] += twoEHmM_re[n][l][0]*twoEHmM_re[l][m][1]
                                  - twoEHmM_im[n][l][0]*twoEHmM_im[l][m][1];
                prod_im[n][m][2] += twoEHmM_re[n][l][0]*twoEHmM_im[l][m][1]
                                  + twoEHmM_im[n][l][0]*twoEHmM_re[l][m][1];
            }

#pragma unroll
    for (int n = 0; n < 3; ++n)
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            const T d01 = d_dmMatMat[0][1], d02 = d_dmMatMat[0][2];
            const T d12 = d_dmMatMat[1][2], d10 = d_dmMatMat[1][0];
            const T d20 = d_dmMatMat[2][0], d21 = d_dmMatMat[2][1];
            prod_re[n][m][0] /= (d01 * d02);
            prod_im[n][m][0] /= (d01 * d02);
            prod_re[n][m][1] /= (d12 * d10);
            prod_im[n][m][1] /= (d12 * d10);
            prod_re[n][m][2] /= (d20 * d21);
            prod_im[n][m][2] /= (d20 * d21);
        }
}

// Compute the 3×3 transition amplitude matrix A for neutrino energy E
// travelling L km through matter of constant density rho.
// phase_offset is always 0.0 in the physics — removed from the signature.
template <typename T>
BARGER_HD void getA(T L, T E, T rho,
                    const T d_dmMatVac[3][3], const T d_dmMatMat[3][3],
                    NeutrinoType type, const OscParamsPOD& p,
                    math::Complex3x3<T>& A) noexcept {
    T prod_re[3][3][3], prod_im[3][3][3];
    get_product(L, E, rho, d_dmMatVac, d_dmMatMat, type, p, prod_re, prod_im);

    constexpr T LoEfac = T(2.534);

    // X[i][j] = sum_k exp(-i * LoEfac * dmMatVac[k][0] * L/E) * product[i][j][k]
    T X_re[3][3]{}, X_im[3][3]{};

#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const T arg = -LoEfac * d_dmMatVac[k][0] * L / E;
        T s, c;
        detail::sincos_impl(arg, &s, &c);
#pragma unroll
        for (int i = 0; i < 3; ++i)
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                X_re[i][j] += c * prod_re[i][j][k] - s * prod_im[i][j][k];
                X_im[i][j] += c * prod_im[i][j][k] + s * prod_re[i][j][k];
            }
    }

    // A[n][m] = sum_{i,j} AXFAC(n,m,i,j,*) * X[i][j]
#pragma unroll
    for (int n = 0; n < 3; ++n)
#pragma unroll
        for (int m = 0; m < 3; ++m) {
            A.re[n*3+m] = T(0);
            A.im[n*3+m] = T(0);
#pragma unroll
            for (int i = 0; i < 3; ++i)
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    A.re[n*3+m] += p.AXFAC(n,m,i,j,0) * X_re[i][j]
                                 + p.AXFAC(n,m,i,j,1) * X_im[i][j];
                    A.im[n*3+m] += p.AXFAC(n,m,i,j,2) * X_im[i][j]
                                 + p.AXFAC(n,m,i,j,3) * X_re[i][j];
                }
        }
}

// Full transition amplitude for a single constant-density segment.
template <typename T>
BARGER_HD void get_transition_matrix(NeutrinoType type, T Enu, T rho, T Len,
                                      const OscParamsPOD& p,
                                      math::Complex3x3<T>& Aout) noexcept {
    T d_dmMatVac[3][3], d_dmMatMat[3][3];
    getMfast(Enu, rho, type, p, d_dmMatMat, d_dmMatVac);
    getA(Len, Enu, rho, d_dmMatVac, d_dmMatMat, type, p, Aout);
}

} // namespace physics
} // namespace cudaprob3

#undef BARGER_HD
