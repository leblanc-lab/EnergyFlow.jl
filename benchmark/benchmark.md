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
the pairwise benchmark swept over thread counts and again over matrix size,
event isotropy, the Sinkhorn accuracy sweep, the comparison tables, and both
paper figures:

```bash
cd EnergyFlow.jl/benchmark
./run_scaling_benchmarks.sh              # laptop
sbatch run_scaling_benchmarks.sh         # SLURM, requests an exclusive node
```

It reads `JULIA`, `PYTHON`, `VENV`, and `THREAD_COUNTS` from the environment if
you need to override them. On a batch system it requests the node exclusively:
the thread-scaling measurement is meaningless if other jobs share the cores.

Individual groups take hours, so `BENCH_ONLY` re-runs a subset rather than the
whole job — after fixing one benchmark, or when only the figures need rebuilding:

```bash
BENCH_ONLY="sinkhorn figure" sbatch run_scaling_benchmarks.sh
```

Valid groups are `single`, `pairwise`, `pairs`, `large`, `isotropy`, `sinkhorn`,
`tables`, and `figure`; an unrecognized name is an error rather than a silently
empty run. The remaining sections describe running each piece individually.

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

## Pairwise scaling in matrix size

The thread sweep above fixes the workload at 2500 pairs and varies parallelism.
This sweep does the opposite — fixed thread count, matrices from ~5k to ~3.9M
pairs — which is the axis you extrapolate along when sizing a real run:

```bash
cd EnergyFlow.jl/benchmark
julia --project=. make_large_sample.jl 2800 result/scaling_sample.csv
julia --project=. -t 48 pairs_scaling_benchmark.jl       # -> result/pairs_scaling_julia_t48.md
python wass_pairwise_benchmark.py --threads 48 \
    --sample result/scaling_sample.csv \
    --sizes 100,200,350,500,700,1000,1400,2000,2800      # -> result/pairs_scaling_wass_t48.md
```

Both halves take *prefixes* of one sample rather than generating their own, so
the size-N run in each language solves byte-identical problems.
`make_large_sample.jl` draws from a seeded stream in order, which is what makes
prefixes well defined: the first N events of the 2800-event sample are identical
to a freshly generated N-event one. The second argument keeps this sample
separate from `large_sample.csv`, so the large-matrix benchmark below is
unaffected.

The result table reports pairs per second. A constant rate means wall time is
linear in the pair count, which is what an embarrassingly parallel workload
should give once the matrix is big enough to keep every thread fed; the
informative part is the small-matrix end, where fixed overhead and load
imbalance across uneven multiplicities pull the rate below its asymptote.
`ENERGYFLOW_PAIRS_SIZES` (Julia) and `--sizes` (Python) override the sweep and
must be given the same values. Points over 30 s report a single timed run after
a warmup instead of a median over three, on both sides.

This feeds panel (c) of `paper/scaling.png`.

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

## Sinkhorn accuracy vs cost

The `:sinkhorn` backend solves the entropy-regularised problem, so a timing of
it means nothing without the error of the value that timing bought. This sweep
records both, over regularisation strength ε and multiplicity n, scoring every
Sinkhorn value against the exact solve of the same problem:

```bash
cd EnergyFlow.jl/benchmark
julia --project=. sinkhorn_benchmark.jl          # -> result/sinkhorn_julia.md
source .venv-iso/bin/activate
python sinkhorn_benchmark_python.py              # -> result/sinkhorn_python.md
python sinkhorn_plot.py                          # -> paper/sinkhorn.png
```

The POT half runs `ot.sinkhorn2(method='sinkhorn_log')` — POT's log-domain
implementation, the same formulation as `src/Sinkhorn.jl`, and the one that does
not underflow at these ε values. It answers the question the Julia timings alone
cannot: where the approximate backend is slower than the exact one, this
separates "this implementation" from "Sinkhorn". Both return ⟨γ, C⟩ with no
entropic term, and converged values agree to ~8 significant figures.

Two things the sweep is careful about, both of which change the conclusion if
ignored:

- **Convergence is recorded, not assumed.** At the 5000-iteration default, every
  point below ε ≈ 0.01 stops early, and an unconverged run reports a cost that
  is too *low* — the iterates never reached a feasible plan. Read naively that
  looks like accuracy improving with ε and then degrading, when it is only the
  iteration cap. These scripts use `max_iter = 100000`, flag any point that
  still hits it, and the figure draws those hollow.
- **The reference is the exact solve of the same problem.** Relative errors
  reach ~1e-5, far below the level at which a hardcoded reference would hold.

The sweep is a bounded grid: the full ε sweep at `ENERGYFLOW_SINKHORN_EPS_SIZES`
(default 50, 100, 200) and an n sweep at ε ∈ {0.05, 0.01} over
`ENERGYFLOW_SINKHORN_SIZES` (default 10 … 500). A full cross product is not run —
cost grows as n² per iteration *and* the iteration count grows as ε falls, so
the expensive corner would dominate the job for points no figure needs.
`ENERGYFLOW_SINKHORN_EPS` and `ENERGYFLOW_SINKHORN_MAXITER` override the ε grid
and the cap; both scripts read the same variables and must be given the same
values, since the two halves share the figure's axes.

Because this sweep runs one backend rather than several, compilation is warmed
up once at startup instead of per point, and solves over 10 s report their pilot
measurement rather than repeating a two-minute call to get the same number.

## Network-simplex pivot rules

Compares the three entering-arc rules in `NetworkSimplex.jl`: `:serial` (block
search, the default), `:parallel_block` (the Kara & Özturan scheme), and
`:full_parallel` (parallel Dantzig, documented in the source as a
correctness/performance baseline rather than a production mode).

```bash
cd EnergyFlow.jl/benchmark
for t in 1 2 4 8 16 32 48; do
    julia --project=. -t $t pivot_benchmark.jl     # -> result/pivot_julia_t$t.md
done
python pivot_plot.py                               # -> paper/pivot.png
```

Two things about the setup are deliberate.

**It measures single-pair solves, not pairwise.** Parallel pivoting parallelises
*within* one solve, while `emds` already gets embarrassing parallelism *across*
pairs. Running both would oversubscribe the machine and measure scheduler
contention rather than either scheme, so the regime tested is the only one where
a parallel pivot rule can help: a single solve large enough that one
entering-arc search is worth splitting.

**It reaches into the struct.** `pivot_mode` is not exposed through `emd`/`emds`
— those build their workspaces internally — so the benchmark sets
`ws.ns.pivot_mode` on an explicit `EMDWorkspace`, as `test/test_emdsolver.jl`
does. There is currently no way to select a pivot rule for a *pairwise*
calculation at all.

Report both columns the result table gives you, because they disagree. `Iters`
is the pivot count, which is the algorithmic claim and is free of any threading
effect: a rule reaching the same optimum in fewer pivots is choosing better
entering arcs. Wall time is whether those choices paid for themselves.
`ArcScans` is not instrumented for `:parallel_block` and reports 0 there rather
than a measurement. `ENERGYFLOW_PIVOT_SIZES` overrides the multiplicities, and
`ENERGYFLOW_PIVOT_FULLMAX` (default 1000) caps `:full_parallel`, whose cost
grows as arcs × pivots with no block to bound the scan — at n = 1000 one solve
is already 4.4 billion arc evaluations.

## Paper figures

```bash
cd EnergyFlow.jl/benchmark
python scaling_plot.py                           # -> paper/scaling.png
python sinkhorn_plot.py                          # -> paper/sinkhorn.png
python isotropy_plot.py                          # -> paper/isotropy.png
python pivot_plot.py                             # -> paper/pivot.png
```

`scaling.png` has three panels: (a) single-pair time against multiplicity,
log-log, all implementations; (b) pairwise wall time against Julia thread count,
with the single-threaded POT time as a reference line and ideal linear scaling
as a guide; (c) pairwise wall time against matrix size at fixed thread count,
against linear growth. Each panel is omitted if its inputs are missing — (b)
needs at least two `result/emds_julia_t{N}.md` files, (c) needs
`result/pairs_scaling_*.md` — so the figure still builds from a laptop run or
from one benchmark rerun on its own.

`sinkhorn.png` has three: (a) relative error against ε, with POT's Sinkhorn
overlaid to show the two implementations agree on the values being timed;
(b) wall time against the accuracy achieved at one multiplicity, with both exact
solvers as reference lines; (c) time against multiplicity at fixed ε. It builds
from `result/sinkhorn_julia.md` alone, omitting the POT comparison with a note.

`isotropy.png` has three: (a) wall time per reference geometry for each backend
and POT; (b) speedup over the single-threaded POT reference at each Julia thread
count; (c) `|mean isotropy − POT|` per backend and geometry, the panel that says
whether a timing was earned on the right answer. It prints any backend/geometry
pair disagreeing by more than 1e-6 — these are exact solvers on identical
problems, so a nonzero difference in a Float64 backend is a solver defect rather
than a tolerance, and the Float32 backends should show a precision-limited
staircase that grows with problem size and nothing more.

Both scripts take `--output` and `--dpi`, warn if their inputs came from more
than one machine, and read an archived result set via
`ENERGYFLOW_RESULT_DIR=/path/to/result`.

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
- `sinkhorn_julia.md` — Julia `:sinkhorn` time, value, relative error and convergence over (ε, n)
- `sinkhorn_python.md` — the same sweep through POT's `ot.sinkhorn2(method='sinkhorn_log')`
- `pairs_scaling_julia_t{N}.md` — Julia exact-backend wall time and throughput vs matrix size
- `pairs_scaling_wass_t{N}.md` — the same sweep through `wasserstein.PairwiseEMD`
- `pivot_julia_t{N}.md` — pivot count and wall time per entering-arc rule, per thread count

The paper figures are written outside this directory, to `paper/scaling.png`,
`paper/sinkhorn.png`, `paper/isotropy.png` and `paper/pivot.png`.
