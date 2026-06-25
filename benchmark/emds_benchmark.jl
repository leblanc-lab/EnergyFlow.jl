# Pairwise EMD Benchmark — Julia EnergyFlow.jl
# Usage: cd benchmark && julia --project=. -t 8 emds_benchmark.jl

include("benchmarks.jl") # defines SUITE and loads the events

using Printf

println("Julia $(VERSION), $(Threads.nthreads()) threads")
println("Loaded $(length(events)) events\n")

results = []
for (split_name, ra, rb) in [("10v90", 1:10, 11:100), ("50v50", 1:50, 51:100)]
    ea, eb = events[ra], events[rb]
    na, nb = length(ea), length(eb)
    emds(ea[1:2], eb[1:2]; R=1.0, beta=1.0, norm=true, backend=:ns64)  # warmup

    for (setup_name, metric, norm) in [
        ("Euclidean_norm",   EuclideanMetric(), true),
        ("Euclidean_unnorm", EuclideanMetric(), false),
        ("EtaPhi_norm",      EtaPhiMetric(),    true),
    ]
        for backend in [:ns64, :ot64, :ns32, :ot32]
            t0 = time_ns()
            emds(ea, eb; R=1.0, beta=1.0, norm=norm, backend=backend, metric=metric)
            t1 = time_ns()
            elapsed = (t1 - t0) / 1e9
            push!(results, (split=split_name, setup=setup_name, backend=backend,
                            time_s=elapsed, pairs=na*nb))
            @printf("%-6s %-16s %-6s %8.2f s  (%d pairs)\n",
                    split_name, setup_name, backend, elapsed, na*nb)
        end
    end
end

# Save results
mkpath(joinpath(@__DIR__, "result"))
open(joinpath(@__DIR__, "result", "emds_julia.md"), "w") do io
    println(io, "# Pairwise EMD Benchmark — Julia EnergyFlow.jl")
    println(io, "\nJulia $(VERSION), $(Threads.nthreads()) thread(s)\n")
    println(io, "| Split | Setup | Backend | Time (s) | Pairs |")
    println(io, "|---|---|---|---|---|")
    for r in results
        @printf(io, "| %s | %s | %s | %.2f | %d |\n",
                r.split, r.setup, r.backend, r.time_s, r.pairs)
    end
end
println("\nResults saved to result/emds_julia.md")
