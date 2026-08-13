# Pairwise EMD scaling in the number of pairs — Julia EnergyFlow.jl
# Usage: cd benchmark && julia --project=. -t 48 pairs_scaling_benchmark.jl
#
# The existing pairwise benchmarks answer "how does one matrix scale with thread
# count" (emds_benchmark.jl) and "is a 1000-event matrix tractable at all"
# (large_pairwise_benchmark.jl, a single point). Neither answers the question an
# analyst actually asks first: how does wall time grow with the size of the
# matrix, and can I extrapolate from a small run to the one I want?
#
# This sweeps self-pairwise matrices from ~5k to ~4M pairs at fixed thread
# count, so throughput can be read directly and the growth checked against
# linear. Pairwise EMD is embarrassingly parallel, so time *should* be linear in
# the pair count once the matrix is large enough to keep every thread fed; the
# useful content of the measurement is where that breaks down — small matrices,
# where per-call overhead and load imbalance across uneven multiplicities show
# up as throughput below the asymptote.
#
# Writes result/pairs_scaling_julia_t{N}.md, tagged by thread count.
# wass_pairwise_benchmark.py --sizes is the like-for-like counterpart, and
# scaling_plot.py turns both into panel (c) of the paper figure.
#
# Needs a sample at least as large as the biggest size:
#   julia --project=. make_large_sample.jl 2800 result/scaling_sample.csv

include("envinfo.jl")

using EnergyFlow
using DelimitedFiles
using Printf
using Statistics

const SAMPLE = get(ENV, "ENERGYFLOW_SCALING_SAMPLE",
                   joinpath(@__DIR__, "result", "scaling_sample.csv"))

# Event counts; the pair count is N(N-1)/2, so this spans 4,950 to 3,918,600.
const SIZES = haskey(ENV, "ENERGYFLOW_PAIRS_SIZES") ?
    parse.(Int, split(ENV["ENERGYFLOW_PAIRS_SIZES"], ",")) :
    [100, 200, 350, 500, 700, 1000, 1400, 2000, 2800]

const BACKENDS = [:ns64, :ot64]
const SETUP    = ("Euclidean_norm", EuclideanMetric(), true)

const MAX_REPS = 3
# Past this, repetitions cost minutes to reproduce a number that is already
# stable — a four-million-pair solve is not timer-noise limited.
const LONG_SOLVE_SECONDS = 30.0

"""
    load_sample(path) -> Vector{Matrix{Float64}}

Read the shared resampled sample, grouping rows by the leading event id.
Mirrors `large_pairwise_benchmark.jl`, which reads the same format.
"""
function load_sample(path)
    isfile(path) || error("Missing $path — run " *
        "`julia --project=. make_large_sample.jl $(maximum(SIZES)) result/scaling_sample.csv` first.")
    raw, _ = readdlm(path, ',', Float64; header = true)
    events = Matrix{Float64}[]
    start = 1
    for i in 2:(size(raw, 1) + 1)
        if i > size(raw, 1) || raw[i, 1] != raw[start, 1]
            push!(events, raw[start:(i - 1), 2:4])
            start = i
        end
    end
    return events
end

"""
    timed(f) -> (median_s, min_s, reps)

Time `f()` after an untimed warmup, returning median and minimum wall time.
A pilot decides whether repetitions are affordable: below `LONG_SOLVE_SECONDS`
the point is a median over `MAX_REPS`, above it the pilot is reported directly.
"""
function timed(f)
    f()                                             # warm up this backend
    t0 = time_ns()
    f()
    est = (time_ns() - t0) / 1e9
    if est > LONG_SOLVE_SECONDS
        return est, est, 1
    end
    times = Float64[est]
    for _ in 2:MAX_REPS
        t0 = time_ns()
        f()
        push!(times, (time_ns() - t0) / 1e9)
    end
    return median(times), minimum(times), length(times)
end

print_env()

all_events = load_sample(SAMPLE)
setup_name, metric, norm = SETUP
sizes = filter(n -> n <= length(all_events), SIZES)
if length(sizes) < length(SIZES)
    skipped = filter(n -> n > length(all_events), SIZES)
    println("Sample holds $(length(all_events)) events; skipping sizes $skipped.")
    println("Regenerate with `make_large_sample.jl $(maximum(SIZES)) result/scaling_sample.csv` "*
            "to cover the full sweep.\n")
end
isempty(sizes) && error("No requested size fits in the $(length(all_events))-event sample.")

nthr = Threads.nthreads()
println("Pairwise scaling sweep at $nthr thread(s), setup $setup_name")
println("Sizes: $sizes events → $(map(n -> n * (n - 1) ÷ 2, sizes)) pairs\n")

results = NamedTuple[]
for n in sizes
    events = all_events[1:n]
    npairs = n * (n - 1) ÷ 2
    for backend in BACKENDS
        med, mn, reps = timed() do
            emds(events; R=1.0, beta=1.0, norm=norm, backend=backend, metric=metric)
        end
        push!(results, (nevents=n, pairs=npairs, setup=setup_name, backend=string(backend),
                        time_s=med, min_s=mn, reps=reps, rate=npairs / med))
        @printf("%5dx%-5d %-6s %9.3f s  (min %9.3f s, %d reps, %9d pairs, %8.0f pairs/s)\n",
                n, n, backend, med, mn, reps, npairs, npairs / med)
        flush(stdout)
    end
end

mkpath(joinpath(@__DIR__, "result"))
outfile = "pairs_scaling_julia_t$(nthr).md"
open(joinpath(@__DIR__, "result", outfile), "w") do io
    println(io, "# Pairwise EMD Scaling in Pair Count — Julia EnergyFlow.jl\n")
    println(io, "Full self-pairwise matrices over prefixes of the resampled sample from")
    println(io, "`make_large_sample.jl`, at $nthr thread(s). Exact backends only.\n")
    println(io, "`Rate` is pairs per second — constant rate means time is linear in the pair")
    println(io, "count, which is what an embarrassingly parallel workload should give once the")
    println(io, "matrix is large enough to keep every thread busy. Points below `$(LONG_SOLVE_SECONDS)` s are")
    println(io, "the median of up to $(MAX_REPS) repetitions; longer ones report a single timed run after")
    println(io, "a warmup.\n")
    write_env_block(io)
    println(io, "| Events | Pairs | Setup | Backend | Time (s) | Min (s) | Reps | Rate (pairs/s) |")
    println(io, "|---|---|---|---|---|---|---|---|")
    for r in results
        @printf(io, "| %d | %d | %s | %s | %.4f | %.4f | %d | %.1f |\n",
                r.nevents, r.pairs, r.setup, r.backend, r.time_s, r.min_s, r.reps, r.rate)
    end
end
println("\nResults saved to result/$outfile")
