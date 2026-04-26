#include <iostream>
#include <cassert>
#include <cmath>
#include "cudaprob3/core/matrix3x3.hpp"
#include "cudaprob3/core/cuda_helpers.hpp"

using namespace cudaprob3;

__global__ void test_multiply_kernel(const MatrixElement<double>* a,
                                      const MatrixElement<double>* b,
                                      MatrixElement<double>* c) {
    Matrix3x3<double> A, B, C;
    for (int i = 0; i < 9; ++i) {
        A.data[i] = a[i];
        B.data[i] = b[i];
    }
    multiply_complex_matrix(A, B, C);
    for (int i = 0; i < 9; ++i) c[i] = C.data[i];
}

int main() {
    cuda_check_device(0);

    Matrix3x3<double> h_A, h_B, h_C;
    h_A.setIdentity();
    h_B.setIdentity();

    MatrixElement<double> *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, 9 * sizeof(MatrixElement<double>));
    cudaMalloc(&d_B, 9 * sizeof(MatrixElement<double>));
    cudaMalloc(&d_C, 9 * sizeof(MatrixElement<double>));

    cudaMemcpy(d_A, h_A.data, 9 * sizeof(MatrixElement<double>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data, 9 * sizeof(MatrixElement<double>), cudaMemcpyHostToDevice);

    test_multiply_kernel<<<1, 1>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data, d_C, 9 * sizeof(MatrixElement<double>), cudaMemcpyDeviceToHost);

    assert(std::fabs(h_C(0,0).re - 1.0) < 1e-6);
    assert(std::fabs(h_C(1,1).re - 1.0) < 1e-6);
    assert(std::fabs(h_C(2,2).re - 1.0) < 1e-6);

    std::cout << "test_matrix3x3: PASSED" << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
