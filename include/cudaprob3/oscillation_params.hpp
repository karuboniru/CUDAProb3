#pragma once

#include "../../src/physics/params_pod.cuh"

#include <cmath>

namespace cudaprob3 {

// Value-semantic oscillation parameter set.
// computePOD() builds the full OscParamsPOD needed by the GPU kernel.
class OscillationParams {
public:
    OscillationParams(double theta12, double theta13, double theta23,
                      double deltaCP, double dm12sq, double dm23sq)
        : theta12_(theta12), theta13_(theta13), theta23_(theta23),
          deltaCP_(deltaCP), dm12sq_(dm12sq), dm23sq_(dm23sq) {}

    [[nodiscard]] double theta12() const noexcept { return theta12_; }
    [[nodiscard]] double theta13() const noexcept { return theta13_; }
    [[nodiscard]] double theta23() const noexcept { return theta23_; }
    [[nodiscard]] double deltaCP() const noexcept { return deltaCP_; }
    [[nodiscard]] double dm12sq()  const noexcept { return dm12sq_; }
    [[nodiscard]] double dm23sq()  const noexcept { return dm23sq_; }

    // Build the full kernel-ready parameter block.
    // For antineutrinos, the PMNS matrix is complex-conjugated (U -> U*),
    // which is equivalent to flipping the sign of deltaCP.
    [[nodiscard]] OscParamsPOD computePOD(bool antineutrino = false) const {
        OscParamsPOD p{};
        fillMixMatrix(p);
        if (antineutrino) {
            for (int i = 0; i < 9; ++i) p.mix_im[i] = -p.mix_im[i];
        }
        fillMassDifferences(p);
        fillAxfac(p);
        fillMassOrder(p);
        return p;
    }

private:
    void fillMixMatrix(OscParamsPOD& p) const {
        const double s12 = std::sin(theta12_), c12 = std::cos(theta12_);
        const double s13 = std::sin(theta13_), c13 = std::cos(theta13_);
        const double s23 = std::sin(theta23_), c23 = std::cos(theta23_);
        const double sd  = std::sin(deltaCP_), cd  = std::cos(deltaCP_);

        p.mix_re[0*3+0] = c12*c13;                    p.mix_im[0*3+0] = 0;
        p.mix_re[0*3+1] = s12*c13;                    p.mix_im[0*3+1] = 0;
        p.mix_re[0*3+2] = s13*cd;                     p.mix_im[0*3+2] = -s13*sd;
        p.mix_re[1*3+0] = -s12*c23 - c12*s23*s13*cd; p.mix_im[1*3+0] = -c12*s23*s13*sd;
        p.mix_re[1*3+1] =  c12*c23 - s12*s23*s13*cd; p.mix_im[1*3+1] = -s12*s23*s13*sd;
        p.mix_re[1*3+2] =  s23*c13;                   p.mix_im[1*3+2] = 0;
        p.mix_re[2*3+0] =  s12*s23 - c12*c23*s13*cd; p.mix_im[2*3+0] = -c12*c23*s13*sd;
        p.mix_re[2*3+1] = -c12*s23 - s12*c23*s13*cd; p.mix_im[2*3+1] = -s12*c23*s13*sd;
        p.mix_re[2*3+2] =  c23*c13;                   p.mix_im[2*3+2] = 0;
    }

    void fillMassDifferences(OscParamsPOD& p) const {
        double mVac0 = 0.0;
        double mVac1 = dm12sq_;
        double mVac2 = dm12sq_ + dm23sq_;
        constexpr double kDelta = 5.0e-9;
        if (dm12sq_ == 0.0) mVac0 -= kDelta;
        if (dm23sq_ == 0.0) mVac2 += kDelta;

        for (int i = 0; i < 3; ++i) p.dm[i*3+i] = 0.0;
        p.dm[0*3+1] = mVac0 - mVac1;  p.dm[1*3+0] = -p.dm[0*3+1];
        p.dm[0*3+2] = mVac0 - mVac2;  p.dm[2*3+0] = -p.dm[0*3+2];
        p.dm[1*3+2] = mVac1 - mVac2;  p.dm[2*3+1] = -p.dm[1*3+2];
    }

    static void fillAxfac(OscParamsPOD& p) {
        for (int n = 0; n < 3; ++n)
            for (int m = 0; m < 3; ++m)
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j) {
                        const double Ur = p.mix_re[n*3+i], Ui = p.mix_im[n*3+i];
                        const double Vr = p.mix_re[m*3+j], Vi = p.mix_im[m*3+j];
                        p.axfac[n*108+m*36+i*12+j*4+0] = Ur*Vr + Ui*Vi;
                        p.axfac[n*108+m*36+i*12+j*4+1] = Ur*Vi - Ui*Vr;
                        p.axfac[n*108+m*36+i*12+j*4+2] = Ui*Vi + Ur*Vr;
                        p.axfac[n*108+m*36+i*12+j*4+3] = Ui*Vr - Ur*Vi;
                    }
    }

    // Vacuum-only mass ordering — type-independent (no matter potential here).
    static void fillMassOrder(OscParamsPOD& p) {
        const double dm01 = p.dm[0*3+1], dm02 = p.dm[0*3+2];
        const double alphaV = dm01 + dm02;
        const double betaV  = dm01 * dm02;
        const double tmpV_raw = alphaV*alphaV - 3.0*betaV;
        const double tmpV = (tmpV_raw < 0) ? 0.0 : tmpV_raw;

        const double argtmp = (2.0*alphaV*alphaV*alphaV - 9.0*alphaV*betaV)
                            / (2.0 * std::sqrt(tmpV*tmpV*tmpV));
        const double arg = (std::fabs(argtmp) > 1.0) ? argtmp/std::fabs(argtmp) : argtmp;

        constexpr double kPi = 3.14159265358979323846;
        const double th0 = std::acos(arg) / 3.0;
        const double base  = -(2.0/3.0)*std::sqrt(tmpV);
        const double shift = p.dm[0*3+0] - alphaV/3.0;
        const double mMatV[3] = {
            base*std::cos(th0) + shift,
            base*std::cos(th0 - 2.0*kPi/3.0) + shift,
            base*std::cos(th0 + 2.0*kPi/3.0) + shift
        };

        for (int i = 0; i < 3; ++i) {
            double best = std::fabs(p.dm[i*3+0] - mMatV[0]);
            int k = 0;
            for (int j = 1; j < 3; ++j) {
                const double d = std::fabs(p.dm[i*3+0] - mMatV[j]);
                if (d < best) { best = d; k = j; }
            }
            p.order[i] = k;
        }
    }

    double theta12_, theta13_, theta23_, deltaCP_, dm12sq_, dm23sq_;
};

} // namespace cudaprob3
