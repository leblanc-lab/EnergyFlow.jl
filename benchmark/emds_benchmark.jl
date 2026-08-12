# Pairwise EMD Benchmark — Julia EnergyFlow.jl
# Usage: cd benchmark && julia --project=. -t 8 emds_benchmark.jl

include("envinfo.jl")
include("benchmarks.jl") # defines SUITE and loads the events

using Printf
using Statistics

print_env()
println("Loaded $(length(events)) events\n")

# Each pairwise solve here takes seconds, so a small number of repetitions is
# enough to suppress noise; the warmup matters more than the repeat count.
const REPS = 3

results = []
for (split_name, ra, rb) in [("10v90", 1:10, 11:100), ("50v50", 1:50, 51:100)]
    ea, eb = events[ra], events[rb]
    na, nb = length(ea), length(eb)

    for (setup_name, metric, norm) in [
        ("Euclidean_norm",   EuclideanMetric(), true),
        ("Euclidean_unnorm", EuclideanMetric(), false),
        ("EtaPhi_norm",      EtaPhiMetric(),    true),
    ]
        for backend in [:ns64, :ot64, :ns32, :ot32]
            # Warm up this backend/metric/norm combination specifically. Warming
            # up only the default backend, as this script previously did, made
            # every other backend report its own compilation as solve time.
            emds(ea[1:2], eb[1:2]; R=1.0, beta=1.0, norm=norm,
                 backend=backend, metric=metric)

            times = Float64[]
            for _ in 1:REPS
                t0 = time_ns()
                emds(ea, eb; R=1.0, beta=1.0, norm=norm, backend=backend, metric=metric)
                push!(times, (time_ns() - t0) / 1e9)
            end
            elapsed = median(times)
            push!(results, (split=split_name, setup=setup_name, backend=backend,
                            time_s=elapsed, min_s=minimum(times), pairs=na*nb))
            @printf("%-6s %-16s %-6s %8.2f s  (min %6.2f s, %d pairs)\n",
                    split_name, setup_name, backend, elapsed, minimum(times), na*nb)
        end
    end
end

# Save results. Two files are written: a thread-tagged one, so a sweep over
# thread counts accumulates rather than overwriting (this feeds the parallel
# scaling panel of the paper figure), and, for single-threaded runs, the
# untagged file that emds_compare.jl reads for the like-for-like POT table.
nthr = Threads.nthreads()
mkpath(joinpath(@__DIR__, "result"))

function write_table(path)
    open(path, "w") do io
        println(io, "# Pairwise EMD Benchmark — Julia EnergyFlow.jl\n")
        println(io, "Time is the median over $(REPS) repetitions, after a per-backend warmup.\n")
        write_env_block(io)
        println(io, "| Split | Setup | Backend | Time (s) | Pairs |")
        println(io, "|---|---|---|---|---|")
        for r in results
            @printf(io, "| %s | %s | %s | %.2f | %d |\n",
                    r.split, r.setup, r.backend, r.time_s, r.pairs)
        end
    end
end

tagged = joinpath(@__DIR__, "result", "emds_julia_t$(nthr).md")
write_table(tagged)
println("\nResults saved to result/emds_julia_t$(nthr).md")

if nthr == 1
    write_table(joinpath(@__DIR__, "result", "emds_julia.md"))
    println("Also saved to result/emds_julia.md (read by emds_compare.jl)")
end
