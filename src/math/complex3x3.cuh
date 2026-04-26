#pragma once

#ifdef __CUDACC__
#  define CUDAPROB3_MATH_HD __host__ __device__
#else
#  define CUDAPROB3_MATH_HD
#endif

namespace cudaprob3 {
namespace math {

// Register-resident 3×3 complex matrix.
// Layout: re[i*3+j], im[i*3+j] — row-major, all ops inline in registers.
template <typename T>
struct Complex3x3 {
    T re[9]{};
    T im[9]{};

    CUDAPROB3_MATH_HD static Complex3x3 identity() noexcept {
        Complex3x3 m{};
        m.re[0] = T(1); m.re[4] = T(1); m.re[8] = T(1);
        return m;
    }

    // C = this * B  (accumulate into C, which must be zero-initialized)
    CUDAPROB3_MATH_HD Complex3x3 operator*(const Complex3x3& B) const noexcept {
        Complex3x3 C{};
#pragma unroll
        for (int i = 0; i < 3; ++i)
#pragma unroll
            for (int j = 0; j < 3; ++j)
#pragma unroll
                for (int k = 0; k < 3; ++k) {
                    const int ik = i*3+k, kj = k*3+j, ij = i*3+j;
                    C.re[ij] += re[ik]*B.re[kj] - im[ik]*B.im[kj];
                    C.im[ij] += im[ik]*B.re[kj] + re[ik]*B.im[kj];
                }
        return C;
    }

    // Store |this[outflv][inflv]|² for all 9 pairs into result[].
    // result layout: [inflv*3+outflv][icos*nE + ie]
    CUDAPROB3_MATH_HD void storeProbs(T* result, unsigned long long base_stride) const noexcept {
#pragma unroll
        for (int inflv = 0; inflv < 3; ++inflv)
#pragma unroll
            for (int outflv = 0; outflv < 3; ++outflv) {
                const T r = re[outflv*3 + inflv];
                const T im_ = im[outflv*3 + inflv];
                result[(unsigned long long)(inflv*3+outflv) * base_stride] = r*r + im_*im_;
            }
    }
};

template <typename T>
CUDAPROB3_MATH_HD inline constexpr T ct_sqr(T x) noexcept { return x*x; }

template <typename T>
CUDAPROB3_MATH_HD inline constexpr T ct_cube(T x) noexcept { return x*x*x; }

} // namespace math
} // namespace cudaprob3

#undef CUDAPROB3_MATH_HD
