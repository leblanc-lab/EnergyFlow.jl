#!/usr/bin/env bash
# Run the full benchmark set that feeds the JOSS paper figure and tables.
#
# Local:  cd benchmark && ./run_scaling_benchmarks.sh
# OSCAR:  sbatch benchmark/run_scaling_benchmarks.sh
#
# On a batch node, request the whole node so no other job shares the cores —
# the thread-scaling panel is meaningless if the CPUs are contended.
#
#SBATCH --job-name=energyflow-bench
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --output=benchmark-%j.out
#
# Adjust the environment section below for your site: OSCAR needs a `module
# load` for Julia, and the Python side needs the numpy + POT virtualenv
# described in benchmark.md. Both are left as variables rather than hardcoded
# so this runs unmodified on a laptop too.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

JULIA="${JULIA:-julia}"
PYTHON="${PYTHON:-python}"
VENV="${VENV:-.venv-iso}"

# Thread counts for the scaling panel. Defaults to powers of two up to the
# number of CPUs actually available to this job, plus the full width.
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
    MAX_THREADS="${SLURM_CPUS_PER_TASK}"
elif command -v nproc >/dev/null 2>&1; then
    MAX_THREADS="$(nproc)"
else
    MAX_THREADS="$(sysctl -n hw.ncpu)"
fi

if [[ -z "${THREAD_COUNTS:-}" ]]; then
    THREAD_COUNTS=""
    t=1
    while (( t <= MAX_THREADS )); do
        THREAD_COUNTS="${THREAD_COUNTS} ${t}"
        t=$(( t * 2 ))
    done
    # Include the full width when it is not itself a power of two (48, say).
    if [[ " ${THREAD_COUNTS} " != *" ${MAX_THREADS} "* ]]; then
        THREAD_COUNTS="${THREAD_COUNTS} ${MAX_THREADS}"
    fi
fi

echo "=== Environment ==="
echo "Julia:          $(${JULIA} --version)"
echo "Max threads:    ${MAX_THREADS}"
echo "Thread counts:  ${THREAD_COUNTS}"
echo

echo "=== Instantiating Julia benchmark environment ==="
${JULIA} --project=. -e 'using Pkg; Pkg.develop(path=".."); Pkg.instantiate()'

if [[ -d "${VENV}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    PYTHON=python
    echo "Activated Python environment ${VENV}"
else
    echo "No ${VENV} found; using ${PYTHON} as-is (needs numpy + pot + matplotlib)."
fi
echo

echo "=== Single-pair scaling (figure panel a) ==="
${JULIA} --project=. single_emd_benchmark.jl
${PYTHON} single_emd_benchmark_python.py
echo

echo "=== Pairwise EMD across thread counts (figure panel b) ==="
for t in ${THREAD_COUNTS}; do
    echo "--- ${t} thread(s) ---"
    ${JULIA} --project=. -t "${t}" emds_benchmark.jl
done
${PYTHON} emds_benchmark_python.py
echo

echo "=== Event isotropy (paper table) ==="
for t in 1 8; do
    if (( t <= MAX_THREADS )); then
        echo "--- ${t} thread(s) ---"
        ${JULIA} --project=. -t "${t}" isotropy_benchmark.jl
    fi
done
${PYTHON} isotropy_benchmark_python.py
echo

echo "=== Comparison tables ==="
${JULIA} --project=. emds_compare.jl
${JULIA} --project=. isotropy_compare.jl
echo

echo "=== Figure ==="
${PYTHON} scaling_plot.py

echo
echo "Done. Figure: paper/scaling.png; tables: benchmark/result/"
