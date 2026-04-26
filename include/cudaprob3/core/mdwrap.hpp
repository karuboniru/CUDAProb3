#ifndef CUDAPROB3_CORE_MDWRAP_HPP
#define CUDAPROB3_CORE_MDWRAP_HPP

// Lightweight multidimensional array view replacing mdspan.
// Works on both host and device without libcu++ mdspan compatibility issues.

namespace cudaprob3 {

template <typename T, unsigned... Dims>
struct DimSizes {
    static constexpr unsigned rank = sizeof...(Dims);
};

// 2D view
template <typename T>
class MDView2D {
public:
    __host__ __device__ MDView2D(T* data, int stride0)
        : data_(data), stride0_(stride0) {}

    __host__ __device__ T& operator()(int i, int j) {
        return data_[i * stride0_ + j];
    }
    __host__ __device__ const T& operator()(int i, int j) const {
        return data_[i * stride0_ + j];
    }
    __host__ __device__ T* data_handle() const { return data_; }

private:
    T* data_;
    int stride0_;
};

// 4D view (for AXFAC: n×m×i×j×k)
template <typename T>
class MDView5D {
public:
    __host__ __device__ MDView5D(T* data,
                                  int s0, int s1, int s2, int s3)
        : data_(data), s0_(s0), s1_(s1), s2_(s2), s3_(s3) {}

    __host__ __device__ T& operator()(int a, int b, int c, int d, int e) {
        return data_[a * s0_ + b * s1_ + c * s2_ + d * s3_ + e];
    }
    __host__ __device__ const T& operator()(int a, int b, int c, int d, int e) const {
        return data_[a * s0_ + b * s1_ + c * s2_ + d * s3_ + e];
    }
    __host__ __device__ T* data_handle() const { return data_; }

private:
    T* data_;
    int s0_, s1_, s2_, s3_;
};

} // namespace cudaprob3

#endif
