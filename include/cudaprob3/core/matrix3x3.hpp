#ifndef CUDAPROB3_CORE_MATRIX3X3_HPP
#define CUDAPROB3_CORE_MATRIX3X3_HPP

#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cudaprob3/core/complex.hpp"
#include "cudaprob3/core/cuda_helpers.hpp"

namespace cudaprob3 {

template <typename FLOAT_T>
struct alignas(sizeof(FLOAT_T) * 2) MatrixElement {
    FLOAT_T re, im;
};

template <typename FLOAT_T>
struct Matrix3x3 {
    MatrixElement<FLOAT_T> data[9];

    __host__ __device__ MatrixElement<FLOAT_T>& operator()(int i, int j) {
        return data[i * 3 + j];
    }
    __host__ __device__ const MatrixElement<FLOAT_T>& operator()(int i, int j) const {
        return data[i * 3 + j];
    }
    __host__ __device__ void setIdentity() {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                data[i * 3 + j].re = (i == j) ? 1.0 : 0.0;
                data[i * 3 + j].im = 0.0;
            }
        }
    }
    __host__ __device__ void setZero() {
        for (int i = 0; i < 9; ++i) {
            data[i].re = 0.0;
            data[i].im = 0.0;
        }
    }
};

enum class MatrixMultiplyStrategy {
    RegisterKernel,
    CuBlasBatched
};

inline const char* strategy_name(MatrixMultiplyStrategy s) {
    switch (s) {
        case MatrixMultiplyStrategy::RegisterKernel: return "RegisterKernel";
        case MatrixMultiplyStrategy::CuBlasBatched:   return "CuBlasBatched";
    }
    return "Unknown";
}

struct MatrixMultiplyBenchmark {
    double elapsed_ms;
    double gflops;
    MatrixMultiplyStrategy strategy;
    size_t batch_size;
};

class MatrixMultiplyEngine {
public:
    explicit MatrixMultiplyEngine(int deviceId, MatrixMultiplyStrategy strategy);
    ~MatrixMultiplyEngine();

    MatrixMultiplyBenchmark warmup_and_benchmark(size_t batch_size, int warmup_iters,
                                                  int bench_iters, cudaStream_t stream);

    MatrixMultiplyStrategy best_strategy(size_t batch_size, cudaStream_t stream);

    void multiply_batch(const Matrix3x3<double>* A,
                        const Matrix3x3<double>* B,
                        Matrix3x3<double>* C,
                        size_t batch_size,
                        cudaStream_t stream);

    const MatrixMultiplyStrategy& strategy() const { return strategy_; }

    MatrixMultiplyEngine(const MatrixMultiplyEngine&) = delete;
    MatrixMultiplyEngine& operator=(const MatrixMultiplyEngine&) = delete;

private:
    void multiply_register(const Matrix3x3<double>* A,
                           const Matrix3x3<double>* B,
                           Matrix3x3<double>* C,
                           size_t batch_size,
                           cudaStream_t stream);

    void multiply_cublas(const Matrix3x3<double>* A,
                         const Matrix3x3<double>* B,
                         Matrix3x3<double>* C,
                         size_t batch_size,
                         cudaStream_t stream);

    int deviceId_;
    MatrixMultiplyStrategy strategy_;
    cublasHandle_t cublasHandle_;

    double* d_Aarray_;
    double* d_Barray_;
    double* d_Carray_;
};

// Free device functions used directly in kernels
template <typename FLOAT_T>
__host__ __device__ void multiply_complex_matrix(const Matrix3x3<FLOAT_T>& A,
                                                  const Matrix3x3<FLOAT_T>& B,
                                                  Matrix3x3<FLOAT_T>& C) {
    C.setZero();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                C(i,j).re += A(i,k).re * B(k,j).re - A(i,k).im * B(k,j).im;
                C(i,j).im += A(i,k).im * B(k,j).re + A(i,k).re * B(k,j).im;
            }
        }
    }
}

template <typename FLOAT_T>
__host__ __device__ void copy_complex_matrix(const Matrix3x3<FLOAT_T>& A,
                                              Matrix3x3<FLOAT_T>& B) {
    for (int i = 0; i < 9; ++i) {
        B.data[i].re = A.data[i].re;
        B.data[i].im = A.data[i].im;
    }
}

} // namespace cudaprob3

#endif
