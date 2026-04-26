#include <iostream>
#include <cassert>
#include <cmath>
#include "cudaprob3/physics/mixing.hpp"

using namespace cudaprob3;

int main() {
    MixingMatrix mix(MixingParams(0.5839, 0.1484, 0.7385, 3.9095));

    // Check unitarity: U * U^dag = I
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double re = 0.0, im = 0.0;
            for (int k = 0; k < 3; ++k) {
                re += mix(i,k).re * mix(j,k).re + mix(i,k).im * mix(j,k).im;
                im += mix(i,k).re * mix(j,k).im - mix(i,k).im * mix(j,k).re;
            }
            if (i == j) {
                assert(std::fabs(re - 1.0) < 1e-6);
                assert(std::fabs(im) < 1e-6);
            } else {
                assert(std::fabs(re) < 1e-6);
                assert(std::fabs(im) < 1e-6);
            }
        }
    }

    // Verify axfac precomputation matches direct computation
    auto ax = mix.axfac_view();
    for (int n = 0; n < 3; ++n) {
            for (int mm = 0; mm < 3; ++mm) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                    double re1 = mix(n,i).re, im1 = mix(n,i).im;
                    double re2 = mix(mm,j).re, im2 = mix(mm,j).im;
                    assert(std::fabs(ax(n,mm,i,j,0) - (re1*re2 + im1*im2)) < 1e-10);
                    assert(std::fabs(ax(n,mm,i,j,1) - (re1*im2 - im1*re2)) < 1e-10);
                    assert(std::fabs(ax(n,mm,i,j,2) - (im1*im2 + re1*re2)) < 1e-10);
                    assert(std::fabs(ax(n,mm,i,j,3) - (im1*re2 - re1*im2)) < 1e-10);
                }
            }
        }
    }

    std::cout << "test_mixing: PASSED" << std::endl;
    return 0;
}
