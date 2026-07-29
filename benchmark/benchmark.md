# EnergyFlow.jl Benchmark Guide

## Prerequisites

- Julia 1.9+
- Python with **numpy + pot** (POT) — this is all the pairwise-EMD, event-isotropy,
  and test cross-check comparisons need. A throwaway venv is the quickest path:
  ```bash
  cd EnergyFlow.jl/benchmark
  python -m venv .venv-iso && source .venv-iso/bin/activate
  pip install numpy pot
  ```
  All Python comparison scripts call POT's exact solver (`ot.lp.emd2`) directly
  and reproduce EnergyFlow.jl's `emd`/`emds` exactly (same ground distances,
  R/beta/norm handling), so the numbers are directly comparable.
- Only the legacy **single-pair notebooks** additionally need `wasserstein` and
  `EnergyFlow` (`conda activate wasserstein`; for `wasserstein` we recommend a
  Linux/Windows system). The pairwise benchmark no longer needs them.
- Test data: `EnergyFlow.jl/data/event0_n*.csv` and `sk_example_PU.hepmc`

## Run the full benchmark suite

Runs the BenchmarkTools `SUITE` in `benchmarks.jl` (pairwise EMD + isotropy,
all backends) — the same suite CI runs via AirspeedVelocity:
```bash
cd EnergyFlow.jl/benchmark
julia --project=. -t 8 -e 'include("benchmarks.jl"); using BenchmarkTools; run(SUITE; verbose=true)'
```

## Produce the full set of POT comparisons

Runs both benchmarks (pairwise EMD + isotropy) on the Julia and POT sides, then
merges each into a side-by-side table. Assumes the numpy + pot venv from
Prerequisites exists at `.venv-iso`. Results land in `result/emds_compare.md` and
`result/isotropy_compare.md`.

**Fairness note:** both sides solve identical problems with an exact
network-simplex solver in Float64, but POT (`ot.lp.emd2`) is single-threaded — it
solves one problem at a time in a plain Python loop, while Julia's `emds`
parallelizes with `Threads.@threads`. So **thread count is the axis to control**:
the single-threaded (no `-t`) Julia run is the like-for-like solver comparison,
and a `-t N` run shows the realistic threaded workload. The isotropy benchmark is
run at both below (its compare table shows a `thr = 1` and a `thr = 8` row per
setup); the pairwise benchmark is run single-threaded — add `-t 8` if you also
want its threaded numbers.
```bash
cd EnergyFlow.jl/benchmark
source .venv-iso/bin/activate

# Julia side — pairwise single-threaded, isotropy at 1 and 8 threads
julia --project=. emds_benchmark.jl
julia --project=. isotropy_benchmark.jl
julia --project=. -t 8 isotropy_benchmark.jl

# POT side (single-threaded)
python emds_benchmark_python.py
python isotropy_benchmark_python.py

# Merge into Julia-vs-POT tables
julia --project=. emds_compare.jl
julia --project=. isotropy_compare.jl
```

## Single-pair EMD Benchmark

For the single-pair EMD benchmark, julia was run with 1000 rep, and 10 for python. Also, the benchmark scripts is written in Jupyter notebook for better visualization of results. It can also run in pure Julia / Python scripts.

### Julia (run in EnergyFlow.jl/benchmark/)
```bash
cd EnergyFlow.jl/benchmark
julia single_emd_benchmark.ipynb   # or open in Jupyter
```

### Python (run in EnergyFlow.jl/benchmark/)
```bash
cd EnergyFlow.jl/benchmark
conda activate wasserstein
jupyter notebook single_emd_benchmark_python.ipynb
```

## Pairwise EMD Benchmark

Direct Julia-vs-POT comparison over the same event pairs (10-vs-90 and 50-vs-50
splits × Euclidean/EtaPhi metrics, normalized and unnormalized). The Python side
calls POT's exact solver (`ot.lp.emd2`) directly — numpy + pot only, no
`wasserstein`/`EnergyFlow` install.

### Julia
```bash
cd EnergyFlow.jl/benchmark
julia --project=. emds_benchmark.jl          # single thread; add -t 8 for 8 threads
```

### Python (POT)
```bash
cd EnergyFlow.jl/benchmark
source .venv-iso/bin/activate                # numpy + pot only (see Prerequisites)
python emds_benchmark_python.py
```

### Side-by-side comparison
After running both, merge the result tables into one Julia-vs-POT view (default
exact backend `:ns64` vs `ot.lp.emd2`, with speedup per split/setup):
```bash
cd EnergyFlow.jl/benchmark
julia --project=. emds_compare.jl            # writes result/emds_compare.md
```

## Event Isotropy Benchmark

Computes the event isotropy event shape (Cesarotti & Thaler, arXiv:2004.06125;
measured by ATLAS in arXiv:2305.16930) of each event against quasi-uniform
references: ring (N=2, N=128) and cylinder (16 φ-slices, |y| ≤ 4) for hadron
colliders, and sphere (192 / 48 HEALPix points) for lepton colliders. The
Julia and Python scripts use identical reference events, ground distances, and
event selection, so the mean isotropy values they report should agree to ≥ 6
decimal places — the Python script calls POT's exact solver (`ot.lp.emd2`)
directly. Shared definitions live in `isotropy_defs.jl` (also used by the CI
suite in `benchmarks.jl`, group `"isotropy"`). The sphere reference reproduces
`astropy_healpix.pix2vec` in pure code, so neither script needs a HEALPix
dependency.

### Julia
Each run writes a thread-tagged file (`result/isotropy_julia_t{N}.md`); run it
once per thread count you want in the comparison:
```bash
cd EnergyFlow.jl/benchmark
julia --project=. isotropy_benchmark.jl       # 1 thread  -> isotropy_julia_t1.md
julia --project=. -t 8 isotropy_benchmark.jl  # 8 threads -> isotropy_julia_t8.md
```

### Python (POT)
Uses the same numpy + pot environment as the pairwise benchmark (see
Prerequisites):
```bash
cd EnergyFlow.jl/benchmark
source .venv-iso/bin/activate
python isotropy_benchmark_python.py
```
Already have a conda env with POT? Just `conda activate <env>` instead.

### Side-by-side comparison
After running the benchmarks, merge their result tables into one Julia-vs-POT
view — both exact backends (`:ns64`, `:ot64`) vs `ot.lp.emd2`, with a row per
thread count found (`thr = 1` is the like-for-like solver comparison, higher
threads the realistic workload) and the mean-isotropy agreement per setup:
```bash
cd EnergyFlow.jl/benchmark
julia --project=. isotropy_compare.jl   # writes result/isotropy_compare.md
```

## Test Cross-check (POT reference)

The Julia test suite asserts EMD values against hardcoded analytic numbers so it
stays Python-free in CI. `test/pot_reference.py` recomputes those same small
cases with POT (`ot.lp.emd2`) and prints them next to the value each test
expects, confirming the hardcoded numbers are POT-correct:
```bash
source EnergyFlow.jl/benchmark/.venv-iso/bin/activate   # numpy + pot
python EnergyFlow.jl/test/pot_reference.py
```

## Results

All results are saved to `EnergyFlow.jl/benchmark/result/`:
- `single_emd_julia.md` — Julia single-pair timings
- `single_emd_python.md` — Python single-pair timings
- `emds_julia.md` — Julia pairwise timings
- `emds_python.md` — POT pairwise timings
- `emds_compare.md` — Julia (`:ns64`, `:ot64`) vs POT pairwise, side by side
- `isotropy_julia_t{N}.md` — Julia event-isotropy timings + mean values, per thread count N
- `isotropy_python.md` — POT event-isotropy timings + mean values
- `isotropy_compare.md` — Julia (`:ns64`, `:ot64`) vs POT isotropy, single- and multi-thread rows
