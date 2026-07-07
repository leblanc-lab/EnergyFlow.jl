# EnergyFlow.jl Benchmark Guide

## Prerequisites

- Julia 1.9+
- Python EF benchmarking conda env: `conda activate wasserstein`
  - For conda environment, we need to have 'pythonOT', 'wasserstein', 'EnergyFlow' installed. For wasserstein library, we recommend using window/linux system.
- Test data: `EnergyFlow.jl/data/event0_n*.csv` and `sk_example_PU.hepmc`

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

### Julia — single thread
```bash
cd EnergyFlow.jl/benchmark
julia --project=. emds_benchmark.jl
```

### Julia — 8 threads
```bash
cd EnergyFlow.jl/benchmark
julia --project=. -t 8 emds_benchmark.jl
```

### Python — single thread
```bash
cd EnergyFlow.jl/benchmark
conda activate wasserstein
python emds_benchmark_python.py
```

### Python — 8 threads
`OMP_NUM_THREADS` controls both wasserstein's OpenMP threads and EnergyFlow's
`n_jobs` parameter (read from the env var inside the script).

Bash / zsh:
```bash
cd EnergyFlow.jl/benchmark
conda activate wasserstein
OMP_NUM_THREADS=8 python emds_benchmark_python.py
```

PowerShell:
```powershell
cd EnergyFlow.jl/benchmark
conda activate wasserstein
$env:OMP_NUM_THREADS=8
python emds_benchmark_python.py
```

Windows cmd.exe:
```cmd
cd EnergyFlow.jl\benchmark
conda activate wasserstein
set OMP_NUM_THREADS=8
python emds_benchmark_python.py
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
```bash
cd EnergyFlow.jl/benchmark
julia --project=. -t 8 isotropy_benchmark.jl
```

### Python (POT)
Unlike the other benchmarks, this one only needs `numpy` and `pot` (POT) — no
`wasserstein`/`EnergyFlow` install — so it runs in any environment with those
two packages. A throwaway venv is the quickest path:
```bash
cd EnergyFlow.jl/benchmark
python -m venv .venv-iso && source .venv-iso/bin/activate
pip install numpy pot
python isotropy_benchmark_python.py
```
Already have a conda env with POT? Just `conda activate <env>` instead of the
venv lines.

### Side-by-side comparison
After running both scripts, merge their result tables into one Julia-vs-POT
view (default exact backend `:ns64` vs `ot.lp.emd2`, with speedup and
mean-isotropy difference per setup):
```bash
cd EnergyFlow.jl/benchmark
julia --project=. isotropy_compare.jl   # writes result/isotropy_compare.md
```

## Results

All results are saved to `EnergyFlow.jl/benchmark/result/`:
- `single_emd_julia.md` — Julia single-pair timings
- `single_emd_python.md` — Python single-pair timings
- `emds_julia.md` — Julia pairwise timings
- `emds_python.md` — Python pairwise timings
- `isotropy_julia.md` — Julia event-isotropy timings + mean values
- `isotropy_python.md` — POT event-isotropy timings + mean values
