# EnergyFlow.jl Benchmark Guide

## Prerequisites

- Julia 1.12+
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

## Results

All results are saved to `EnergyFlow.jl/benchmark/result/`:
- `single_emd_julia.md` — Julia single-pair timings
- `single_emd_python.md` — Python single-pair timings
- `emds_julia.md` — Julia pairwise timings
- `emds_python.md` — Python pairwise timings
