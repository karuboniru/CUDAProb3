#include <algorithm>
#include <cmath>
#include <chrono>
#include <vector>

#include "cudaprob3/core/matrix3x3.hpp"

namespace cudaprob3 {

// Fully unrolled register-based 3x3 complex matrix multiply kernel
// Each thread processes one pair of matrices
template <typename FLOAT_T>
__global__ void kernel_multiply_3x3_register(
    const MatrixElement<FLOAT_T>* __restrict__ A,
    const MatrixElement<FLOAT_T>* __restrict__ B,
    MatrixElement<FLOAT_T>* __restrict__ C,
    size_t batch_size)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const MatrixElement<FLOAT_T>* a = A + idx * 9;
    const MatrixElement<FLOAT_T>* b = B + idx * 9;
    MatrixElement<FLOAT_T>* c = C + idx * 9;

    // Row 0
    c[0].re = a[0].re * b[0].re - a[0].im * b[0].im
            + a[1].re * b[3].re - a[1].im * b[3].im
            + a[2].re * b[6].re - a[2].im * b[6].im;
    c[0].im = a[0].re * b[0].im + a[0].im * b[0].re
            + a[1].re * b[3].im + a[1].im * b[3].re
            + a[2].re * b[6].im + a[2].im * b[6].re;
    c[1].re = a[0].re * b[1].re - a[0].im * b[1].im
            + a[1].re * b[4].re - a[1].im * b[4].im
            + a[2].re * b[7].re - a[2].im * b[7].im;
    c[1].im = a[0].re * b[1].im + a[0].im * b[1].re
            + a[1].re * b[4].im + a[1].im * b[4].re
            + a[2].re * b[7].im + a[2].im * b[7].re;
    c[2].re = a[0].re * b[2].re - a[0].im * b[2].im
            + a[1].re * b[5].re - a[1].im * b[5].im
            + a[2].re * b[8].re - a[2].im * b[8].im;
    c[2].im = a[0].re * b[2].im + a[0].im * b[2].re
            + a[1].re * b[5].im + a[1].im * b[5].re
            + a[2].re * b[8].im + a[2].im * b[8].re;

    // Row 1
    c[3].re = a[3].re * b[0].re - a[3].im * b[0].im
            + a[4].re * b[3].re - a[4].im * b[3].im
            + a[5].re * b[6].re - a[5].im * b[6].im;
    c[3].im = a[3].re * b[0].im + a[3].im * b[0].re
            + a[4].re * b[3].im + a[4].im * b[3].re
            + a[5].re * b[6].im + a[5].im * b[6].re;
    c[4].re = a[3].re * b[1].re - a[3].im * b[1].im
            + a[4].re * b[4].re - a[4].im * b[4].im
            + a[5].re * b[7].re - a[5].im * b[7].im;
    c[4].im = a[3].re * b[1].im + a[3].im * b[1].re
            + a[4].re * b[4].im + a[4].im * b[4].re
            + a[5].re * b[7].im + a[5].im * b[7].re;
    c[5].re = a[3].re * b[2].re - a[3].im * b[2].im
            + a[4].re * b[5].re - a[4].im * b[5].im
            + a[5].re * b[8].re - a[5].im * b[8].im;
    c[5].im = a[3].re * b[2].im + a[3].im * b[2].re
            + a[4].re * b[5].im + a[4].im * b[5].re
            + a[5].re * b[8].im + a[5].im * b[8].re;

    // Row 2
    c[6].re = a[6].re * b[0].re - a[6].im * b[0].im
            + a[7].re * b[3].re - a[7].im * b[3].im
            + a[8].re * b[6].re - a[8].im * b[6].im;
    c[6].im = a[6].re * b[0].im + a[6].im * b[0].re
            + a[7].re * b[3].im + a[7].im * b[3].re
            + a[8].re * b[6].im + a[8].im * b[6].re;
    c[7].re = a[6].re * b[1].re - a[6].im * b[1].im
            + a[7].re * b[4].re - a[7].im * b[4].im
            + a[8].re * b[7].re - a[8].im * b[7].im;
    c[7].im = a[6].re * b[1].im + a[6].im * b[1].re
            + a[7].re * b[4].im + a[7].im * b[4].re
            + a[8].re * b[7].im + a[8].im * b[7].re;
    c[8].re = a[6].re * b[2].re - a[6].im * b[2].im
            + a[7].re * b[5].re - a[7].im * b[5].im
            + a[8].re * b[8].re - a[8].im * b[8].im;
    c[8].im = a[6].re * b[2].im + a[6].im * b[2].re
            + a[7].re * b[5].im + a[7].im * b[5].re
            + a[8].re * b[8].im + a[8].im * b[8].re;
}

// Explicit instantiation for double
template __global__ void kernel_multiply_3x3_register<double>(
    const MatrixElement<double>*, const MatrixElement<double>*,
    MatrixElement<double>*, size_t);

MatrixMultiplyEngine::MatrixMultiplyEngine(int deviceId, MatrixMultiplyStrategy strategy)
    : deviceId_(deviceId), strategy_(strategy), cublasHandle_(nullptr),
      d_Aarray_(nullptr), d_Barray_(nullptr), d_Carray_(nullptr)
{
    cuda_check_device(deviceId_);

    if (strategy_ == MatrixMultiplyStrategy::CuBlasBatched) {
        CUBLAS_CHECK(cublasCreate(&cublasHandle_));
    }
}

MatrixMultiplyEngine::~MatrixMultiplyEngine() {
    if (cublasHandle_) {
        CUDA_CHECK(cudaSetDevice(deviceId_));
        cublasDestroy(cublasHandle_);
    }
    if (d_Aarray_) {
        CUDA_CHECK(cudaSetDevice(deviceId_));
        CUDA_CHECK(cudaFree(d_Aarray_));
    }
    if (d_Barray_) {
        CUDA_CHECK(cudaSetDevice(deviceId_));
        CUDA_CHECK(cudaFree(d_Barray_));
    }
    if (d_Carray_) {
        CUDA_CHECK(cudaSetDevice(deviceId_));
        CUDA_CHECK(cudaFree(d_Carray_));
    }
}

void MatrixMultiplyEngine::multiply_register(
    const Matrix3x3<double>* A, const Matrix3x3<double>* B,
    Matrix3x3<double>* C, size_t batch_size, cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = (static_cast<int>(batch_size) + threads - 1) / threads;
    kernel_multiply_3x3_register<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const MatrixElement<double>*>(A),
        reinterpret_cast<const MatrixElement<double>*>(B),
        reinterpret_cast<MatrixElement<double>*>(C),
        batch_size);
    CUDA_CHECK(cudaGetLastError());
}

void MatrixMultiplyEngine::multiply_cublas(
    const Matrix3x3<double>* A, const Matrix3x3<double>* B,
    Matrix3x3<double>* C, size_t batch_size, cudaStream_t stream)
{
    CUBLAS_CHECK(cublasSetStream(cublasHandle_, stream));

    const cuDoubleComplex alpha = {1.0, 0.0};
    const cuDoubleComplex beta  = {0.0, 0.0};

    // For batched GEMM with contiguous storage, we use a stride-based approach
    // since cublasZgemmStridedBatched is optimized for this layout.
    // Each 3x3 matrix occupies 9 cuDoubleComplex elements.
    // Layout: row-major, so lda = ldb = ldc = 3.
    CUBLAS_CHECK(cublasZgemmStridedBatched(
        cublasHandle_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        3, 3, 3,           // m, n, k
        &alpha,
        reinterpret_cast<const cuDoubleComplex*>(A->data), 3, 9,  // A, lda, strideA
        reinterpret_cast<const cuDoubleComplex*>(B->data), 3, 9,  // B, ldb, strideB
        &beta,
        reinterpret_cast<cuDoubleComplex*>(C->data), 3, 9,       // C, ldc, strideC
        static_cast<int>(batch_size)
    ));
}

void MatrixMultiplyEngine::multiply_batch(
    const Matrix3x3<double>* A, const Matrix3x3<double>* B,
    Matrix3x3<double>* C, size_t batch_size, cudaStream_t stream)
{
    CUDA_CHECK(cudaSetDevice(deviceId_));
    switch (strategy_) {
        case MatrixMultiplyStrategy::RegisterKernel:
            multiply_register(A, B, C, batch_size, stream);
            break;
        case MatrixMultiplyStrategy::CuBlasBatched:
            multiply_cublas(A, B, C, batch_size, stream);
            break;
    }
}

MatrixMultiplyBenchmark MatrixMultiplyEngine::warmup_and_benchmark(
    size_t batch_size, int warmup_iters, int bench_iters, cudaStream_t stream)
{
    CUDA_CHECK(cudaSetDevice(deviceId_));

    Matrix3x3<double>* d_A = nullptr;
    Matrix3x3<double>* d_B = nullptr;
    Matrix3x3<double>* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, batch_size * sizeof(Matrix3x3<double>)));
    CUDA_CHECK(cudaMalloc(&d_B, batch_size * sizeof(Matrix3x3<double>)));
    CUDA_CHECK(cudaMalloc(&d_C, batch_size * sizeof(Matrix3x3<double>)));

    // Initialize with random-ish data
    std::vector<Matrix3x3<double>> h_A(batch_size), h_B(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        for (int j = 0; j < 9; ++j) {
            h_A[i].data[j].re = static_cast<double>(i * 9 + j) * 0.1;
            h_A[i].data[j].im = static_cast<double>(i * 9 + j) * 0.01;
            h_B[i].data[j].re = static_cast<double>(i * 9 + j + 1) * 0.05;
            h_B[i].data[j].im = static_cast<double>(i * 9 + j + 1) * 0.005;
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A.data(), batch_size * sizeof(Matrix3x3<double>),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B.data(), batch_size * sizeof(Matrix3x3<double>),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Warmup
    for (int i = 0; i < warmup_iters; ++i) {
        multiply_batch(d_A, d_B, d_C, batch_size, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < bench_iters; ++i) {
        multiply_batch(d_A, d_B, d_C, batch_size, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= bench_iters;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    double flops_per_mult = 9.0 * (3.0 * 6.0 + 3.0 * 2.0); // 3x3 complex: 9 outputs, each: 3 (complex mult) + 2 (complex add) x2 (re+im) = ~ 9*3*8
    double gflops = (static_cast<double>(batch_size) * flops_per_mult / (ms * 1e6));

    return {static_cast<double>(ms), gflops, strategy_, batch_size};
}

MatrixMultiplyStrategy MatrixMultiplyEngine::best_strategy(size_t batch_size,
                                                            cudaStream_t stream)
{
    // Test both strategies and return the faster one
    MatrixMultiplyEngine reg(deviceId_, MatrixMultiplyStrategy::RegisterKernel);
    MatrixMultiplyEngine blas(deviceId_, MatrixMultiplyStrategy::CuBlasBatched);

    auto bm_reg  = reg.warmup_and_benchmark(batch_size, 3, 10, stream);
    auto bm_blas = blas.warmup_and_benchmark(batch_size, 3, 10, stream);

    return (bm_reg.elapsed_ms <= bm_blas.elapsed_ms)
        ? MatrixMultiplyStrategy::RegisterKernel
        : MatrixMultiplyStrategy::CuBlasBatched;
}

} // namespace cudaprob3
