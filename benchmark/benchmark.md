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
- **matplotlib** in the same venv if you want to regenerate the paper figure
  (`pip install matplotlib`); nothing else needs it.

## Provenance and timing methodology

Every result file opens with an `## Environment` table recording the CPU model,
core count, OS, Julia/Python and package versions, thread count, git commit
(flagged if the working tree was dirty), and SLURM job details when present.
This is written by `envinfo.jl` / `envinfo.py`, which every benchmark script
calls — a timing table without the machine it ran on is not reproducible.

Two rules apply to every timing:

- **Each backend is warmed up individually before it is timed.** Julia compiles
  on first call, so warming up only the default backend — as these scripts did
  previously — makes every *other* backend report its compilation as though it
  were solve time. This was worth several tens of milliseconds on the small
  isotropy workloads, enough to invert some Julia-vs-POT comparisons.
- **Every point is a median over repetitions**, not a single measurement, with
  the repetition count chosen adaptively so each point gets a fixed timing
  budget regardless of problem size. Both the median and the minimum are
  reported so an outlier-prone point is visible as a gap between them.

## Everything at once

`run_scaling_benchmarks.sh` runs the whole set — both single-pair benchmarks,
the pairwise benchmark swept over thread counts, event isotropy, the comparison
tables, and the paper figure:

```bash
cd EnergyFlow.jl/benchmark
./run_scaling_benchmarks.sh              # laptop
sbatch run_scaling_benchmarks.sh         # SLURM, requests an exclusive node
```

It reads `JULIA`, `PYTHON`, `VENV`, and `THREAD_COUNTS` from the environment if
you need to override them. On a batch system it requests the node exclusively:
the thread-scaling measurement is meaningless if other jobs share the cores.
The remaining sections describe running each piece individually.

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

## Single-pair EMD scaling benchmark

Times one EMD between two events of equal multiplicity, sweeping n from 2 to
3000 over the `data/event{0,1}_n*.csv` samples. This produces panel (a) of the
paper figure. Use the scripts rather than the notebooks for anything you intend
to publish: they warm up each backend individually and report medians, and they
run unattended on a batch node.

```bash
cd EnergyFlow.jl/benchmark
julia --project=. single_emd_benchmark.jl        # -> result/single_emd_julia.md
source .venv-iso/bin/activate
python single_emd_benchmark_python.py            # -> result/single_emd_python.md
```

The Python side times up to three implementations: POT's `ot.lp.emd2` called
directly (the like-for-like solver comparison), Python EnergyFlow's `emd_pot`
(what an analysis actually pays per call, wrapper included), and the
`wasserstein` C++ library. The latter two are optional — if `energyflow` or
`wasserstein` is not installed the script says so and skips that curve, so this
runs in the plain numpy + POT venv.

For a quick check without the expensive large-n points:

```bash
ENERGYFLOW_BENCH_SIZES=2,10,50 ENERGYFLOW_BENCH_TARGET=0.3 julia --project=. single_emd_benchmark.jl
```

The older `single_emd_benchmark*.ipynb` notebooks remain for interactive
exploration; they sweep asymmetric event sizes the scripts do not.

## Threaded pairwise comparison (fair parallel test)

`ot.lp.emd2` is single-threaded and driven from a Python loop, so comparing an
N-thread Julia `emds` against it measures Julia's threading against Python
having none. The `wasserstein` library — the compiled backend behind Python
EnergyFlow — has a genuinely multithreaded pairwise driver, so running it at
the same thread count is the like-for-like test:

```bash
cd EnergyFlow.jl/benchmark
julia --project=. -t 8 emds_benchmark.jl            # -> result/emds_julia_t8.md
python wass_pairwise_benchmark.py --threads 8       # -> result/wass_pairwise_t8.md
```

Panel (b) of the figure overlays the two. POT still appears, drawn as a
horizontal reference line rather than a curve, and is a fair comparison only
against the single-thread point.

## Large distance matrix

The HepMC sample holds 100 events, capping a pairwise matrix at 100×100.
`make_large_sample.jl` builds a larger sample by drawing those events with
replacement and applying an independent random azimuthal rotation to each draw.
A φ rotation is a symmetry of the detector geometry and the multiplicity
distribution — which is what sets solve cost — is preserved exactly, so this is
valid for timing and for nothing else. The resulting matrix is not a physics
measurement.

Both languages read the same generated CSV, so they solve byte-identical
problems:

```bash
cd EnergyFlow.jl/benchmark
julia --project=. make_large_sample.jl 1000                          # -> result/large_sample.csv
julia --project=. -t 48 large_pairwise_benchmark.jl                  # -> result/large_pairwise_julia_t48.md
python wass_pairwise_benchmark.py --threads 48 --sample result/large_sample.csv
```

Note what this does and does not show. Pairwise EMD is embarrassingly parallel,
so the per-pair advantage at 500k pairs is the same as at 2500 pairs — the
ratio does not improve with matrix size. What a large matrix demonstrates is
absolute tractability at the scale analyses actually use.

## Paper figure

```bash
cd EnergyFlow.jl/benchmark
python scaling_plot.py                           # -> paper/scaling.png
```

Two panels: (a) single-pair time against multiplicity, log-log, all
implementations; (b) pairwise wall time against Julia thread count, with the
single-threaded POT time as a reference line and ideal linear scaling as a
guide. Panel (b) needs at least two `result/emds_julia_t{N}.md` files and is
omitted otherwise, so the figure still builds from a laptop run. Point it at an
archived result set with `ENERGYFLOW_RESULT_DIR=/path/to/result`.

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

Each run writes a thread-tagged `result/emds_julia_t{N}.md`, so sweeping thread
counts accumulates results instead of overwriting them (this feeds panel (b) of
the paper figure). A single-threaded run additionally writes the untagged
`result/emds_julia.md` that `emds_compare.jl` reads.

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

All results are saved to `EnergyFlow.jl/benchmark/result/`, each with an
`## Environment` provenance table:
- `single_emd_julia.md` — Julia single-pair scaling timings
- `single_emd_python.md` — Python single-pair scaling timings (POT / EnergyFlow / wasserstein)
- `emds_julia_t{N}.md` — Julia pairwise timings, per thread count N
- `emds_julia.md` — Julia pairwise timings, single-threaded (read by `emds_compare.jl`)
- `emds_python.md` — POT pairwise timings
- `emds_compare.md` — Julia (`:ns64`, `:ot64`) vs POT pairwise, side by side
- `isotropy_julia_t{N}.md` — Julia event-isotropy timings + mean values, per thread count N
- `isotropy_python.md` — POT event-isotropy timings + mean values
- `isotropy_compare.md` — Julia (`:ns64`, `:ot64`) vs POT isotropy, single- and multi-thread rows

The paper figure is written outside this directory, to `paper/scaling.png`.
