# Single-pair EMD scaling benchmark — Julia EnergyFlow.jl
# Usage: cd benchmark && julia --project=. single_emd_benchmark.jl
#
# Times one EMD between two events of equal multiplicity n, sweeping n across
# the data/event{0,1}_n*.csv samples, and writes result/single_emd_julia.md.
# This is the Julia half of the scaling figure in the JOSS paper; the Python
# half is single_emd_benchmark_python.py and scaling_plot.py merges the two.
#
# This is a script rather than a notebook so it can run unattended on a batch
# node. It differs from single_emd_benchmark.ipynb in two ways that matter for
# a published figure: every backend is warmed up individually before it is
# timed, and each point is the median of an adaptively chosen number of
# repetitions rather than a single measurement.

include("envinfo.jl")

using EnergyFlow
using DelimitedFiles
using Printf
using Statistics

# Multiplicities to sweep. Override for a quick smoke run, e.g.
#   ENERGYFLOW_BENCH_SIZES=2,10,50 julia --project=. single_emd_benchmark.jl
const SIZES = haskey(ENV, "ENERGYFLOW_BENCH_SIZES") ?
    parse.(Int, split(ENV["ENERGYFLOW_BENCH_SIZES"], ",")) :
    [2, 10, 50, 100, 200, 500, 1000, 2000, 3000]

const BACKENDS = [:ns64, :ot64]
const SETUPS   = [("Euclidean_norm",   EuclideanMetric(), true),
                  ("Euclidean_unnorm", EuclideanMetric(), false)]

# Timing budget per (size, setup, backend) point.
const TARGET_SECONDS = parse(Float64, get(ENV, "ENERGYFLOW_BENCH_TARGET", "2.0"))
const MIN_REPS       = 5
const MAX_REPS       = 50_000

# Accumulating results here keeps the compiler from treating the timed calls as
# dead code, which it would otherwise be free to do since we discard the EMD.
const SINK = Ref(0.0)

load_event(path) = readdlm(path, ',', Float64)

"""
    timed(f) -> (median_s, min_s, reps)

Time `f()` repeatedly and return the median and minimum per-call wall time.

`f` is called once before any measurement so that compilation is never folded
into a reported timing — the single largest source of error in the previous
version of this benchmark, where only the default backend was warmed up and the
others reported their compile time as though it were solve time.

The repetition count is then chosen from a pilot measurement so each point gets
roughly `TARGET_SECONDS` of timing. A fixed count cannot work across this
sweep: at n = 2 a single solve takes well under a microsecond and would be
dominated by timer granularity, while at n = 3000 it takes long enough that
many repetitions would be wasteful.
"""
function timed(f)
    SINK[] += f()                                   # warm up this exact combination
    t0 = time_ns()
    SINK[] += f()
    est = (time_ns() - t0) / 1e9
    reps = clamp(ceil(Int, TARGET_SECONDS / max(est, 1e-9)), MIN_REPS, MAX_REPS)

    times = Vector{Float64}(undef, reps)
    for i in 1:reps
        t0 = time_ns()
        SINK[] += f()
        times[i] = (time_ns() - t0) / 1e9
    end
    return median(times), minimum(times), reps
end

print_env()

results = NamedTuple[]
for n in SIZES
    ev0 = load_event(joinpath(@__DIR__, "..", "data", "event0_n$(n).csv"))
    ev1 = load_event(joinpath(@__DIR__, "..", "data", "event1_n$(n).csv"))

    for (setup_name, metric, norm) in SETUPS, backend in BACKENDS
        med, mn, reps = timed() do
            emd(ev0, ev1; R=1.0, beta=1.0, norm=norm, backend=backend, metric=metric)
        end
        push!(results, (n=n, setup=setup_name, backend=string(backend),
                        median_s=med, min_s=mn, reps=reps))
        @printf("n=%-5d %-18s %-6s median=%10.3f µs  min=%10.3f µs  (%d reps)\n",
                n, setup_name, backend, med * 1e6, mn * 1e6, reps)
    end
end

mkpath(joinpath(@__DIR__, "result"))
open(joinpath(@__DIR__, "result", "single_emd_julia.md"), "w") do io
    println(io, "# Single-Pair EMD Scaling — Julia EnergyFlow.jl\n")
    println(io, "Median wall time for one `emd` call between two events of equal")
    println(io, "multiplicity n. Each point is the median over `Reps` repetitions after a")
    println(io, "per-backend warmup; cost-matrix construction is included, as it is inside")
    println(io, "`emd`.\n")
    write_env_block(io)
    println(io, "| n | Setup | Backend | Median (s) | Min (s) | Reps |")
    println(io, "|---|---|---|---|---|---|")
    for r in results
        @printf(io, "| %d | %s | %s | %.9g | %.9g | %d |\n",
                r.n, r.setup, r.backend, r.median_s, r.min_s, r.reps)
    end
end
println("\nResults saved to result/single_emd_julia.md")
