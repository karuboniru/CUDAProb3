#pragma once

#ifdef __CUDACC__
#  define CUDAPROB3_HD __host__ __device__
#else
#  define CUDAPROB3_HD
#endif

namespace cudaprob3 {

// All per-PMNS-set data needed by the kernel, templated on floating-point type T.
// Passed as a pointer into a device array (one entry per PMNS set in batch mode).
//
// Indexing (matching original macros):
//   mix_re/im: U[i][j] → [i*3+j]
//   dm:        DM[i][j] → [i*3+j]
//   axfac:     AXFAC(n,m,i,j,e) → [n*108 + m*36 + i*12 + j*4 + e]   (3^4 * 4 = 324 entries)
//   order:     ORDER[i]
template<typename T = double>
struct OscParamsPOD {
    T   mix_re[9];
    T   mix_im[9];
    T   dm[9];
    T   axfac[324];   // 81 * 4 = 324
    int order[3];

    CUDAPROB3_HD T   U_re(int i, int j) const noexcept { return mix_re[i*3+j]; }
    CUDAPROB3_HD T   U_im(int i, int j) const noexcept { return mix_im[i*3+j]; }
    CUDAPROB3_HD T   DM(int i, int j)   const noexcept { return dm[i*3+j]; }
    CUDAPROB3_HD T   AXFAC(int n, int m, int i, int j, int e) const noexcept {
        return axfac[n*108 + m*36 + i*12 + j*4 + e];
    }
    CUDAPROB3_HD int ORDER(int i) const noexcept { return order[i]; }
};

} // namespace cudaprob3

#undef CUDAPROB3_HD
