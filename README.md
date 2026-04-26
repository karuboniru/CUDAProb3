# CUDAProb3 v2.0

GPU-accelerated 3-flavour neutrino oscillation propagator — a ground-up
C++17/CUDA rewrite of the original
[CUDAProb3++](https://doi.org/10.1016/j.cpc.2018.07.022).

## Overview

CUDAProb3 computes neutrino (and antineutrino) oscillation probabilities through
a radially-symmetric density profile (the Earth) using the Barger et al.
three-flavour framework with Mikheyev-Smirnov-Wolfenstein (MSW) matter effects.
It evaluates the transition probability *P*(ν<sub>α</sub> → ν<sub>β</sub>) for
all 9 oscillation channels simultaneously over a grid of neutrino energies and
zenith-angle cosines.

This rewrite ("rewrite-ds") drops the legacy CPU/OpenMP backend and focuses
exclusively on GPU computation with a modern, header-only OOP design.

### Key features

- **Header-only physics engine** — mixing, MSW effects, transition amplitudes
  all live in `include/`
- **Two matrix-multiply strategies** — register-based hand-written kernel or
  batched cuBLAS GEMM (`cublasZgemmStridedBatched`), auto-selected by benchmark
- **Multi-GPU support** — cosine partitioning with three strategies (Cyclic,
  Block, LoadBalanced)
- **Session manager** — singleton scheduler for batching multiple parameter sets
  and asynchronous execution
- **mdspan-style indexing** — lightweight `DeviceBuffer<T>` / `PinnedBuffer<T>`
  with RAII and multi-dimensional views

### Physics

The code implements the full 3×3 neutrino evolution through piecewise-constant
density layers. The PMNS mixing matrix is parameterised as

```
U = R₂₃(θ₂₃) · U₁₃(θ₁₃, δ_CP) · R₁₂(θ₁₂)
```

and the propagation Hamiltonian in matter is

```
H = (1 / 2E) [ U diag(0, Δm²₂₁, Δm²₃₁) U† ] + V(x)
```

where *V*(*x*) = √2 *G<sub>F</sub>* *N<sub>e</sub>*(*x*) diag(1, 0, 0) for
neutrinos (sign flipped for antineutrinos).

## Requirements

| Dependency | Purpose |
|-----------|---------|
| CUDA Toolkit (≥12, tested with 13.0) | GPU runtime + cuBLAS |
| g++ ≥14 (hard-coded path `/usr/bin/g++-15`) | Host + CUDA host compiler |
| CMake ≥3.20 | Build system |
| cuBLAS | Batched complex GEMM strategy |
| cuRAND | Reserved (not yet used) |

GPU: NVIDIA Ampere or newer (SM ≥86). Adjust `CMAKE_CUDA_ARCHITECTURES` in
`CMakeLists.txt` for other architectures.

## Quick start

```bash
# Clone & build
git clone git@github.com:karuboniru/CUDAProb3.git
cd CUDAProb3
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run the example (100×100 energy × cosine grid, single GPU)
./example_benchmark

# Custom grid size with specific GPUs
./example_benchmark 200 200 0 1

# Run unit tests
ctest
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `CUDA_PROB3_BUILD_TESTS` | ON | Build unit tests |
| `CUDA_PROB3_BUILD_EXAMPLES` | ON | Build example programs |
| `CUDA_PROB3_BUILD_BENCHMARKS` | ON | Build matrix multiply benchmarks |

## API overview

Everything is accessible through a single header:

```cpp
#include "cudaprob3/cudaprob3.hpp"
using namespace cudaprob3;
```

### Single-GPU propagation

```cpp
// 1. Build configuration
PropagatorConfig cfg;
cfg.mixing = MixingParams(0.5839, 0.1484, 0.7385, 3.9095);  // θ12, θ13, θ23, δCP
cfg.dm12sq = 7.42e-5;   // eV²
cfg.dm23sq = 2.517e-3;  // eV²
cfg.cosines  = {/* zenith cosines */};
cfg.energies = {/* neutrino energies in GeV */};
cfg.earthModel.loadFromFile("models/PREM_4layer.dat");
cfg.productionHeightKm = 0.0;   // atmospheric production height

// 2. Create and run propagator
SingleGPUPropagator<double> prop(/*gpuId=*/0, nCosines, nEnergies);
prop.configure(cfg);
prop.calculate(NeutrinoType::Neutrino);   // or NeutrinoType::Antineutrino

// 3. Retrieve probabilities
double p_ee = prop.getProbability(cosineIdx, energyIdx, ProbType::e_e);
double p_em = prop.getProbability(cosineIdx, energyIdx, ProbType::e_m);
// ... 9 channels total: e_e, e_m, e_t, m_e, m_m, m_t, t_e, t_m, t_t
```

### Multi-GPU propagation

```cpp
std::vector<int> gpuIds = {0, 1};   // use GPUs 0 and 1

MultiGPUPropagator prop(gpuIds, nCosines, nEnergies,
                        WorkDistributionStrategy::Cyclic);
prop.configure(cfg);
prop.calculate(NeutrinoType::Neutrino);
double p = prop.getProbability(0, 0, ProbType::e_e);
```

Work distribution strategies:

| Strategy | Behaviour |
|----------|-----------|
| `Cyclic` | Round-robin assignment of cosine bins across GPUs |
| `Block` | Contiguous chunks of cosines per GPU |
| `LoadBalanced` | Greedy assignment weighted by layer count (deep trajectories are more expensive) |

### Session manager (async batching)

```cpp
auto& mgr = SessionManager::instance();

SessionConfig sCfg{cfg, {0}, WorkDistributionStrategy::Block};
auto session = mgr.createSession("my_fit", sCfg);

auto future = session->runAsync(NeutrinoType::Neutrino);
// ... do other work ...

ResultSet result = future.get();
double p = result.p(cosineIdx, energyIdx, ProbType::e_e);
```

### Earth model files

Earth density models are plain-text files with one layer per line:

```
radius_km    density_g/cm³
```

Lines can be in ascending or descending radius order. Example model files are
at `example/models/PREM_4layer.dat`.

### Probability types

| Enum | Channel |
|------|---------|
| `ProbType::e_e` | ν<sub>e</sub> → ν<sub>e</sub> |
| `ProbType::e_m` | ν<sub>e</sub> → ν<sub>μ</sub> |
| `ProbType::e_t` | ν<sub>e</sub> → ν<sub>τ</sub> |
| `ProbType::m_e` | ν<sub>μ</sub> → ν<sub>e</sub> |
| `ProbType::m_m` | ν<sub>μ</sub> → ν<sub>μ</sub> |
| `ProbType::m_t` | ν<sub>μ</sub> → ν<sub>τ</sub> |
| `ProbType::t_e` | ν<sub>τ</sub> → ν<sub>e</sub> |
| `ProbType::t_m` | ν<sub>τ</sub> → ν<sub>μ</sub> |
| `ProbType::t_t` | ν<sub>τ</sub> → ν<sub>τ</sub> |

## Architecture

```
include/cudaprob3/
├── core/           Complex numbers, 3×3 matrices, CUDA helpers, physics constants
├── engine/         GPU device, memory pool (RAII), CUDA stream pool
├── geometry/       Earth density model, trajectory chord-length computation
├── physics/        PMNS mixing (with precomputed A×fac), MSW matter effects, amplitudes
├── propagator/     Abstract Propagator, SingleGPUPropagator, MultiGPUPropagator, work distribution
└── session/        Session (sync/async), SessionManager singleton
```

The entire physics pipeline is header-only. Only `src/core/matrix3x3.cu`
contains a non-trivial compilation unit (the register-kernel CUDA implementation
of batched 3×3 complex matrix multiply).

## References

Barger, V., Whisnant, K., Pakvasa, S., & Phillips, R. J. N. (1980).
Matter effects on three-neutrino oscillations. *Physical Review D*,
22(11), 2718.

Kallenborn, F., et al. (2018). CUDAProb3++ — CUDA implementation of
neutrino oscillation probability calculator Prob3++.
*Computer Physics Communications*, 234, 235–244.
[doi:10.1016/j.cpc.2018.07.022](https://doi.org/10.1016/j.cpc.2018.07.022)

Original Prob3++:
[http://webhome.phy.duke.edu/~raw22/public/Prob3++/](http://webhome.phy.duke.edu/~raw22/public/Prob3++/)

## License

GNU Lesser General Public License v3 (LGPLv3) — see the repository history for
the full license text inherited from the original CUDAProb3++.
