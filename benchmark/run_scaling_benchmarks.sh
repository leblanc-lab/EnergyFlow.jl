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
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=benchmark-%j.out
#
# --mem=0 requests all memory on the node. Without it an exclusive job can still
# be capped at (memory-per-CPU x allocated CPUs), which is far too little for
# the n = 3000 solves if the CPU allocation comes back smaller than requested.
#
# The script prints its CPU allocation before doing any work and refuses to run
# the thread sweep inside a single-CPU allocation. If that guard fires, check
# what the scheduler actually granted against what the header asked for.
#
# Adjust the environment section below for your site: OSCAR needs a `module
# load` for Julia, and the Python side needs the numpy + POT virtualenv
# described in benchmark.md. Both are left as variables rather than hardcoded
# so this runs unmodified on a laptop too.

set -euo pipefail

# Locate the benchmark directory.
#
# Under sbatch this cannot use BASH_SOURCE: SLURM copies the batch script to a
# node-local spool directory and runs that copy, so BASH_SOURCE points at
# /var/spool/slurmd/... and every relative path below would resolve against the
# spool instead of the repository. SLURM_SUBMIT_DIR is the directory sbatch was
# invoked from, which is the reliable anchor on a batch node; fall back to
# BASH_SOURCE for interactive runs. Each candidate is confirmed by looking for
# files that only the real benchmark directory has.
find_benchmark_dir() {
    local candidates=() dir
    [[ -n "${BENCHMARK_DIR:-}" ]] && candidates+=("${BENCHMARK_DIR}")
    if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
        candidates+=("${SLURM_SUBMIT_DIR}/benchmark" "${SLURM_SUBMIT_DIR}")
    fi
    dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)"
    [[ -n "${dir}" ]] && candidates+=("${dir}")

    for dir in "${candidates[@]}"; do
        if [[ -f "${dir}/single_emd_benchmark.jl" && -f "${dir}/../Project.toml" ]]; then
            (cd "${dir}" && pwd)
            return 0
        fi
    done
    return 1
}

if ! BENCH_DIR="$(find_benchmark_dir)"; then
    echo "ERROR: could not locate the benchmark directory." >&2
    echo "  Submit from the repository root (sbatch benchmark/run_scaling_benchmarks.sh)," >&2
    echo "  or set BENCHMARK_DIR=/path/to/EnergyFlow.jl/benchmark." >&2
    exit 1
fi
cd "${BENCH_DIR}"
REPO_ROOT="$(cd .. && pwd)"

JULIA="${JULIA:-julia}"
# Remember whether the caller chose an interpreter, so activating the venv below
# does not silently override an explicit PYTHON=... .
PYTHON_SET_BY_CALLER=0
[[ -n "${PYTHON:-}" ]] && PYTHON_SET_BY_CALLER=1
PYTHON="${PYTHON:-python}"

# The venv may sit in the benchmark directory or at the repository root,
# depending on where it was created. Look in both rather than assuming.
if [[ -z "${VENV:-}" ]]; then
    for _candidate in ".venv-iso" "${REPO_ROOT}/.venv-iso"; do
        if [[ -d "${_candidate}" ]]; then
            VENV="${_candidate}"
            break
        fi
    done
    VENV="${VENV:-.venv-iso}"
fi

# How many CPUs this process may actually use. `nproc` reports the affinity
# mask, so under SLURM it reflects the cgroup the job was placed in rather than
# the size of the physical node — which is what we need, since running 48
# threads inside a 1-CPU allocation produces scaling numbers that mean nothing.
if command -v nproc >/dev/null 2>&1; then
    USABLE_CPUS="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
    USABLE_CPUS="$(sysctl -n hw.ncpu)"
else
    USABLE_CPUS=1
fi

echo "=== CPU allocation ==="
echo "nproc (usable by this process): ${USABLE_CPUS}"
for var in SLURM_CPUS_PER_TASK SLURM_CPUS_ON_NODE SLURM_NTASKS SLURM_JOB_CPUS_PER_NODE; do
    echo "${var}=${!var:-<unset>}"
done

# Trust the affinity mask over the requested value: a request that the scheduler
# silently reduced is exactly the failure this guard exists to catch.
MAX_THREADS="${USABLE_CPUS}"
if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" -ne "${USABLE_CPUS}" ]]; then
    echo "WARNING: requested ${SLURM_CPUS_PER_TASK} CPUs but only ${USABLE_CPUS} are usable."
    echo "         Using ${USABLE_CPUS}. Thread-scaling results above that are not meaningful."
fi
if [[ "${MAX_THREADS}" -lt 2 && -z "${THREAD_COUNTS:-}" ]]; then
    echo "ERROR: only ${MAX_THREADS} CPU available, so the thread-scaling sweep would be" >&2
    echo "  meaningless. Fix the allocation (on OSCAR, --cpus-per-task with a matching" >&2
    echo "  --ntasks=1, and check the job's cgroup), or set THREAD_COUNTS=1 to run the" >&2
    echo "  rest of the benchmarks single-threaded anyway." >&2
    exit 1
fi
echo

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

# Which groups of steps to run. Individual groups take hours, so re-running one
# after a fix should not mean redoing the rest: BENCH_ONLY="sinkhorn figure"
# runs those two groups and skips everything else. Default is everything.
ALL_GROUPS="single pairwise pairs large isotropy sinkhorn pivot tables figure"
BENCH_ONLY="${BENCH_ONLY:-${ALL_GROUPS}}"

for g in ${BENCH_ONLY}; do
    if [[ " ${ALL_GROUPS} " != *" ${g} "* ]]; then
        echo "ERROR: unknown group '${g}' in BENCH_ONLY." >&2
        echo "  Valid groups: ${ALL_GROUPS}" >&2
        exit 1
    fi
done

want() { [[ " ${BENCH_ONLY} " == *" $1 "* ]]; }

echo "=== Environment ==="
echo "Julia:          $(${JULIA} --version)"
echo "Benchmark dir:  ${BENCH_DIR}"
echo "Repo root:      ${REPO_ROOT}"
echo "Max threads:    ${MAX_THREADS}"
echo "Thread counts:  ${THREAD_COUNTS}"
echo "Groups:         ${BENCH_ONLY}"
echo

# Run a benchmark step without letting one failure abandon the rest of the job.
# The steps are largely independent — a broken Python dependency should not cost
# you the Julia thread sweep after it has already been queued for hours — so
# failures are recorded and reported at the end instead of aborting immediately.
FAILED_STEPS=()
run_step() {
    local label="$1"
    shift
    if "$@"; then
        return 0
    fi
    echo "WARNING: step failed and was skipped: ${label}"
    FAILED_STEPS+=("${label}")
    return 0
}

echo "=== Instantiating Julia benchmark environment ==="
# Which checkout of the package to benchmark. Defaults to this repository, but
# can point at another working tree — a branch or a pull request under test —
# so a fix can be measured with the current benchmark scripts without merging
# it first:
#
#   git worktree add ../EnergyFlow-pr72 pr72
#   ENERGYFLOW_PKG_PATH=../EnergyFlow-pr72 BENCH_ONLY=isotropy sbatch run_scaling_benchmarks.sh
#
# envinfo.jl records a "Solver source" row whenever this is not the script
# repository, so a result file can never misattribute a timing to the commit of
# the scripts that produced it.
PKG_PATH="$(cd "${ENERGYFLOW_PKG_PATH:-${REPO_ROOT}}" && pwd)"
if [[ ! -f "${PKG_PATH}/Project.toml" ]]; then
    echo "ERROR: no Project.toml at ENERGYFLOW_PKG_PATH=${PKG_PATH}." >&2
    exit 1
fi
[[ "${PKG_PATH}" != "${REPO_ROOT}" ]] && echo "Benchmarking package from ${PKG_PATH} (not this repo)"
# Absolute path, so this cannot depend on the working directory.
${JULIA} --project=. -e "using Pkg; Pkg.develop(path=raw\"${PKG_PATH}\"); Pkg.instantiate()"

if [[ -d "${VENV}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    if [[ "${PYTHON_SET_BY_CALLER}" -eq 0 ]]; then
        PYTHON=python
    fi
    echo "Activated Python environment ${VENV} (interpreter: ${PYTHON})"
else
    echo "No ${VENV} found; using ${PYTHON} as-is (needs numpy + pot + matplotlib)."
fi
echo

if want single; then
echo "=== Single-pair scaling (figure panel a) ==="
run_step "single-pair Julia"  ${JULIA} --project=. single_emd_benchmark.jl
run_step "single-pair Python" ${PYTHON} single_emd_benchmark_python.py
echo
fi

if want pairwise; then
echo "=== Pairwise EMD across thread counts (figure panel b) ==="
# Julia and wasserstein are run at each thread count so the scaling comparison
# is threaded-against-threaded. POT is measured once, single-threaded, because
# ot.lp.emd2 has no internal parallelism — it is the reference for the
# single-thread column only, not for the threaded ones.
for t in ${THREAD_COUNTS}; do
    echo "--- ${t} thread(s) ---"
    run_step "pairwise Julia (${t} threads)" ${JULIA} --project=. -t "${t}" emds_benchmark.jl
    run_step "pairwise wasserstein (${t} threads)" ${PYTHON} wass_pairwise_benchmark.py --threads "${t}"
done
run_step "pairwise POT (single-threaded)" ${PYTHON} emds_benchmark_python.py
echo
fi

if want pairs; then
echo "=== Pairwise scaling in matrix size (figure panel c) ==="
# Both halves sweep prefixes of one sample so each size solves byte-identical
# problems on the two sides, and both run at the full thread count — the panel
# varies the amount of work, holding parallelism fixed.
PAIRS_SIZES="${PAIRS_SIZES:-100,200,350,500,700,1000,1400,2000,2800}"
PAIRS_MAX="${PAIRS_SIZES##*,}"
run_step "pairs scaling sample (${PAIRS_MAX} events)" \
    ${JULIA} --project=. make_large_sample.jl "${PAIRS_MAX}" result/scaling_sample.csv
run_step "pairs scaling Julia (${MAX_THREADS} threads)" \
    env ENERGYFLOW_PAIRS_SIZES="${PAIRS_SIZES}" \
    ${JULIA} --project=. -t "${MAX_THREADS}" pairs_scaling_benchmark.jl
run_step "pairs scaling wasserstein (${MAX_THREADS} threads)" \
    ${PYTHON} wass_pairwise_benchmark.py --threads "${MAX_THREADS}" \
    --sample result/scaling_sample.csv --sizes "${PAIRS_SIZES}"
echo
fi

if want large; then
echo "=== Large distance matrix (${LARGE_NEVENTS:-1000} events, full width) ==="
run_step "large sample generation" ${JULIA} --project=. make_large_sample.jl "${LARGE_NEVENTS:-1000}"
run_step "large matrix Julia (${MAX_THREADS} threads)" \
    ${JULIA} --project=. -t "${MAX_THREADS}" large_pairwise_benchmark.jl
run_step "large matrix wasserstein (${MAX_THREADS} threads)" \
    ${PYTHON} wass_pairwise_benchmark.py --threads "${MAX_THREADS}" --sample result/large_sample.csv
echo
fi

if want isotropy; then
echo "=== Event isotropy (paper table) ==="
for t in 1 8; do
    if (( t <= MAX_THREADS )); then
        echo "--- ${t} thread(s) ---"
        run_step "isotropy Julia (${t} threads)" ${JULIA} --project=. -t "${t}" isotropy_benchmark.jl
    fi
done
run_step "isotropy Python" ${PYTHON} isotropy_benchmark_python.py
echo
fi

if want sinkhorn; then
echo "=== Sinkhorn accuracy vs cost (sinkhorn figure) ==="
# Single-threaded on both sides: this measures one solver against another over
# an ε grid, and thread count is not the variable under study. The POT half is
# the slow one — its log-domain Sinkhorn runs a few times longer than the Julia
# half — so it goes last and a failure there still leaves a usable Julia figure.
run_step "sinkhorn Julia"  ${JULIA} --project=. sinkhorn_benchmark.jl
run_step "sinkhorn Python" ${PYTHON} sinkhorn_benchmark_python.py
echo
fi

if want pivot; then
echo "=== Network-simplex pivot rules (pivot figure) ==="
# Julia fixes its thread count at startup, so the thread sweep is over separate
# invocations, one result file each. Unlike the pairwise groups this measures
# parallelism *inside* a single solve, so every thread count is meaningful even
# though the workload is one EMD at a time.
for t in ${THREAD_COUNTS}; do
    echo "--- ${t} thread(s) ---"
    run_step "pivot rules (${t} threads)" ${JULIA} --project=. -t "${t}" pivot_benchmark.jl
done
echo
fi

if want tables; then
echo "=== Comparison tables ==="
run_step "emds_compare"     ${JULIA} --project=. emds_compare.jl
run_step "isotropy_compare" ${JULIA} --project=. isotropy_compare.jl
echo
fi

if want figure; then
echo "=== Figures ==="
run_step "scaling_plot"  ${PYTHON} scaling_plot.py
run_step "sinkhorn_plot" ${PYTHON} sinkhorn_plot.py
run_step "isotropy_plot" ${PYTHON} isotropy_plot.py
run_step "pivot_plot"    ${PYTHON} pivot_plot.py
fi

echo
if (( ${#FAILED_STEPS[@]} > 0 )); then
    echo "Completed with ${#FAILED_STEPS[@]} failed step(s):"
    printf '  - %s\n' "${FAILED_STEPS[@]}"
    echo "Everything else finished; results are in benchmark/result/."
    exit 1
fi
if want figure; then
    echo "Done. Figures: paper/{scaling,sinkhorn,isotropy,pivot}.png; tables: benchmark/result/"
else
    # Do not advertise figures a BENCH_ONLY run never rebuilt: the ones on disk
    # are from an earlier run and may predate the results just written.
    echo "Done (groups: ${BENCH_ONLY}). Tables: benchmark/result/"
    echo "Figures not rebuilt; run with BENCH_ONLY=\"... figure\" or the plot scripts directly."
fi
