#ifndef CUDAPROB3_CORE_COMPLEX_HPP
#define CUDAPROB3_CORE_COMPLEX_HPP

namespace cudaprob3 {

template <typename T>
struct ComplexNumber {
    T re;
    T im;

    __host__ __device__ ComplexNumber() : re(0), im(0) {}
    __host__ __device__ ComplexNumber(T r, T i) : re(r), im(i) {}
};

} // namespace cudaprob3

#endif
