# CUDAProb3

GPU-accelerated 3-flavor neutrino oscillation probabilities over a 2-D grid of
(cosine-zenith × energy) bins, using the Barger *et al.* formalism and a
piecewise-constant PREM Earth density model.

This is a clean C++23 / CUDA rewrite of the original CUDAProb3++ library.
It eliminates global `__constant__` state, adds true batch-PMNS mode, multi-GPU
support, and CUDA Graph replay for tight likelihood-minimizer loops.

## Requirements

| Component | Minimum |
|-----------|---------|
| CUDA toolkit | 12.0 (Thrust/CUB bundled via CCCL) |
| GPU architecture | sm_80 (Ampere) or newer |
| C++ host compiler | GCC 13+ or Clang 17+ (C++23) |
| CMake | 3.25 |

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Optional flags:

| Flag | Default | Effect |
|------|---------|--------|
| `CUDAPROB3_BUILD_TESTS` | `OFF` | Catch2 test suite (fetches Catch2 v3 via FetchContent) |
| `CUDAPROB3_BUILD_EXAMPLES` | `OFF` | Builds `example/main.cpp` |
| `CUDAPROB3_BUILD_BENCHMARKS` | `OFF` | Timing benchmarks |

```bash
# Build with tests
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCUDAPROB3_BUILD_TESTS=ON
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

## Usage

Include the umbrella header and link against the static library:

```cpp
#include <cudaprob3/cudaprob3.hpp>
using namespace cudaprob3;
```

### Single GPU

```cpp
// Load Earth density model
auto model = PREMModel::fromFile("models/PREM_12layer.dat").value();

// Build a (cosine × energy) grid
auto cosVec = /* std::vector<double> of cosine-zenith values in [-1, 0] */;
auto eVec   = /* std::vector<double> of energies in GeV */;
auto grid   = std::make_shared<ArbitraryGrid>(cosVec, eVec, /*prodHeightKm=*/22.0);

// Create calculator
SingleGPUCalculator::Config cfg;
cfg.deviceId      = 0;
cfg.useCUDAGraphs = true;   // enables graph replay after the first call
auto calc = SingleGPUCalculator::create(cfg, grid,
                std::make_shared<PREMModel>(std::move(model))).value();

// Set oscillation parameters (all angles in radians, masses in eV²)
OscillationParams params(theta12, theta13, theta23, dcp, dm12sq, dm23sq);

// Calculate — synchronous, returns a non-owning view into a pinned buffer
auto result = calc.calculate(params, NeutrinoType::Neutrino).value();

// Query a single probability
double p = result.probability(/*icos=*/42, /*ie=*/10, ProbType::m_e);

// Or iterate the full (icos, ie) grid for one channel
for (int ic = 0; ic < result.nCosines(); ++ic)
    for (int ie = 0; ie < result.nEnergies(); ++ie)
        use(result.probability(ic, ie, ProbType::m_m));
```

`ResultView` is a lightweight, non-owning view. It remains valid until the next
call to `calculate()` on the same calculator.

### Multi-GPU

Cosine bins are distributed cyclically across GPUs (this naturally load-balances
the variable per-cosine layer count):

```cpp
auto calc = MultiGPUCalculator::create(
    {0, 1},   // device IDs
    grid, std::make_shared<PREMModel>(std::move(model))).value();

auto result = calc.calculate(params, NeutrinoType::Neutrino).value();
```

The returned `ResultView` covers the full (nCosines × nEnergies) grid — the
multi-GPU split is invisible to the caller.

### Batch mode

Processes **B PMNS parameter sets** in a single 3-D kernel launch
(energy × cosine × batch). Ideal for likelihood minimization over oscillation
parameters.

```cpp
auto batch = BatchCalculator::create(
    {0},      // device IDs
    grid, std::make_shared<PREMModel>(std::move(model)),
    /*chunkSize=*/64).value();

// Build a list of parameter-set pointers (B sets)
std::vector<OscillationParams> pVec = { /* ... */ };
std::vector<const OscillationParams*> ptrs;
for (auto& p : pVec) ptrs.push_back(&p);

// Returns a BatchResult — one ResultView per parameter set
BatchResult results = batch.calculate(std::span{ptrs}, NeutrinoType::Neutrino);

double p = results[b].probability(icos, ie, ProbType::e_e);
```

`BatchResult` owns its data (pinned memory). Each `results[b]` is a
`ResultView` into that buffer.

### Neutrino types and probability channels

```cpp
NeutrinoType::Neutrino       // normal matter interaction sign
NeutrinoType::Antineutrino   // flips matter potential

ProbType::e_e   ProbType::e_m   ProbType::e_t   // nu_e initial state
ProbType::m_e   ProbType::m_m   ProbType::m_t   // nu_mu initial state
ProbType::t_e   ProbType::t_m   ProbType::t_t   // nu_tau initial state
```

### CUDA Graphs

When `cfg.useCUDAGraphs = true`, the first `calculate()` call captures a CUDA
Graph (H2D parameter upload + kernel launch). All subsequent calls replay the
graph with updated parameters — no re-submission overhead from the CPU side.
This is transparent to the caller.

### Error handling

`create()` and `calculate()` return `std::expected<T, std::string>`.
Use `.value()` to get the result (throws `std::bad_expected_access` on error)
or check with `if (!result)` and inspect `result.error()`.

## Density models

Four PREM Earth models are included under `models/`:

| File | Layers |
|------|--------|
| `PREM_4layer.dat` | 4 |
| `PREM_10layer.dat` | 10 |
| `PREM_12layer.dat` | 12 (default, used in tests) |
| `PREM_59layer.dat` | 59 |

Format: each line is `radius_km  density_g_cm3`.

## Numerical accuracy

All 9 × 200 × 200 = 360 000 probability values agree with the original
CUDAProb3++ to within **1 × 10⁻¹⁰** (max observed difference: 3.5 × 10⁻¹³,
pure floating-point rounding).

## Publication

The original algorithm is described in:

> Felix Kallenborn, Christian Hundt, Sebastian Böser, Bertil Schmidt,
> *Massively parallel computation of atmospheric neutrino oscillations on
> CUDA-enabled accelerators*, Computer Physics Communications, Volume 234,
> https://doi.org/10.1016/j.cpc.2018.07.022
