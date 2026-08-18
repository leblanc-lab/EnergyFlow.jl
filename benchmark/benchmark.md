# EnergyFlow.jl Benchmark Guide

The benchmark directory contains two maintained workflows:

- a `BenchmarkTools` suite used by AirspeedVelocity; and
- matched Julia/Python scripts for comparing EnergyFlow.jl with POT.

Run benchmarks from a clean checkout and record the commit, hardware, Julia or
Python version, and thread count with any result you publish.

## Setup

EnergyFlow.jl requires Julia 1.9 or later. From the repository root, create the
benchmark environment with the local package checkout:

```bash
julia --project=benchmark -e \
  'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

The Python comparisons require NumPy and
[POT](https://pythonot.github.io/):

```bash
cd benchmark
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install numpy pot
```

All maintained scripts use the sample events in `data/sk_example_PU.hepmc`.

## BenchmarkTools suite

From the repository root:

```bash
julia --project=benchmark -t 8 -e \
  'include("benchmark/benchmarks.jl"); using BenchmarkTools; run(SUITE; verbose=true)'
```

The suite covers cross-pairwise EMD and event isotropy for the four exact
backends. The labeled benchmark workflow runs the same `SUITE` through
AirspeedVelocity.

## Julia/POT comparisons

POT's `ot.lp.emd2` is called serially, one problem at a time. A Julia run with
one thread is therefore the like-for-like solver comparison; a multithreaded
Julia run measures EnergyFlow.jl's parallel workload throughput. Warmup calls
are excluded on both sides.

### Pairwise EMD

From `benchmark/`, with the Python environment activated:

```bash
julia --project=. emds_benchmark.jl
python emds_benchmark_python.py
julia --project=. emds_compare.jl
```

The Julia and Python scripts solve the same 10-vs-90 and 50-vs-50 event sets
with Euclidean and periodic `(eta, phi)` ground metrics, both normalized and
unnormalized. They write:

- `result/emds_julia.md`
- `result/emds_python.md`
- `result/emds_compare.md`

Add `-t N` to the Julia benchmark only when you intentionally want a threaded
throughput measurement. Do not compare that timing with serial POT as a
single-thread solver speedup.

### Event isotropy

Run the Julia script once for each thread count you want to retain, then run
the Python reference and comparison:

```bash
julia --project=. isotropy_benchmark.jl
julia --project=. -t 8 isotropy_benchmark.jl
python isotropy_benchmark_python.py
julia --project=. isotropy_compare.jl
```

This covers ring, cylinder, and spherical reference geometries. The Julia
script uses EnergyFlow.jl's public isotropy helpers; the Python script contains
self-contained NumPy equivalents and solves with POT. Output files are:

- `result/isotropy_julia_tN.md`, where `N` is the Julia thread count
- `result/isotropy_python.md`
- `result/isotropy_compare.md`

The comparison table includes both timing and the difference in mean isotropy,
so numerical agreement is checked alongside performance.

## Test-value cross-check

The Julia test suite uses hard-coded analytic values and has no Python
dependency. To recompute the small reference cases with POT:

```bash
source benchmark/.venv/bin/activate
python test/pot_reference.py
```

Run that command from the repository root.

## Legacy single-pair notebooks

`single_emd_benchmark.ipynb` and `single_emd_benchmark_python.ipynb` are
interactive, single-pair explorations. Open them in Jupyter with the
appropriate Julia or Python kernel. They are not part of the maintained
AirspeedVelocity suite or the matched pairwise/isotropy comparison pipeline.

The generated `benchmark/result/` directory is ignored by Git. Treat result
files as machine- and revision-specific artifacts, not package-wide performance
guarantees.
