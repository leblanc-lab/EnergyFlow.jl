# Large-matrix pairwise EMD benchmark — Julia EnergyFlow.jl
# Usage: cd benchmark && julia --project=. -t 48 large_pairwise_benchmark.jl
#
# Computes the full self-pairwise distance matrix over the resampled sample
# written by make_large_sample.jl — 1000 events by default, so ~500k unique
# pairs, against the 2500 of the 50v50 split. The point is not a better ratio:
# pairwise EMD is embarrassingly parallel, so the per-pair advantage is the
# same as at 2500 pairs. The point is absolute tractability at the scale
# analyses actually use, and a threaded-vs-threaded comparison against
# wasserstein.PairwiseEMD at the same thread count
# (wass_pairwise_benchmark.py --sample).
#
# Writes result/large_pairwise_julia_t{N}.md.

include("envinfo.jl")

using EnergyFlow
using DelimitedFiles
using Printf
using Statistics

const SAMPLE = joinpath(@__DIR__, "result", "large_sample.csv")
const REPS   = 3

"""
    load_sample(path) -> Vector{Matrix{Float64}}

Read the shared resampled sample. Rows are grouped by the leading event id,
which is written in ascending order by make_large_sample.jl.
"""
function load_sample(path)
    isfile(path) || error("Missing $path — run `julia --project=. make_large_sample.jl` first.")
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

print_env()

events = load_sample(SAMPLE)
n = length(events)
npairs = n * (n - 1) ÷ 2
println("Loaded $n events ($npairs unique pairs) from $(basename(SAMPLE))\n")

results = NamedTuple[]
for (setup_name, metric, norm) in [("Euclidean_norm",   EuclideanMetric(), true),
                                   ("Euclidean_unnorm", EuclideanMetric(), false)]
    for backend in [:ns64, :ot64]
        # Warm up this backend on a small subset, not the full matrix.
        emds(events[1:4]; R=1.0, beta=1.0, norm=norm, backend=backend, metric=metric)

        times = Float64[]
        for _ in 1:REPS
            t0 = time_ns()
            emds(events; R=1.0, beta=1.0, norm=norm, backend=backend, metric=metric)
            push!(times, (time_ns() - t0) / 1e9)
        end
        push!(results, (split="$(n)x$(n)", setup=setup_name, backend=string(backend),
                        time_s=median(times), min_s=minimum(times), pairs=npairs))
        @printf("%s %-18s %-6s %8.2f s  (min %6.2f s, %d pairs, %d threads)\n",
                "$(n)x$(n)", setup_name, backend, median(times), minimum(times),
                npairs, Threads.nthreads())
    end
end

nthr = Threads.nthreads()
mkpath(joinpath(@__DIR__, "result"))
outfile = joinpath(@__DIR__, "result", "large_pairwise_julia_t$(nthr).md")
open(outfile, "w") do io
    println(io, "# Large Pairwise EMD — Julia EnergyFlow.jl\n")
    println(io, "Full self-pairwise matrix over the resampled sample from")
    println(io, "make_large_sample.jl. Time is the median over $(REPS) repetitions")
    println(io, "after a per-backend warmup.\n")
    write_env_block(io)
    println(io, "| Split | Setup | Backend | Time (s) | Pairs |")
    println(io, "|---|---|---|---|---|")
    for r in results
        @printf(io, "| %s | %s | %s | %.2f | %d |\n",
                r.split, r.setup, r.backend, r.time_s, r.pairs)
    end
end
println("\nResults saved to result/large_pairwise_julia_t$(nthr).md")
