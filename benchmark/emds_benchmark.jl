# Pairwise EMD Benchmark — Julia EnergyFlow.jl
# Usage: cd benchmark && julia --project=. -t 8 emds_benchmark.jl

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, ".."))
Pkg.instantiate()

using EnergyFlow
using Printf

println("Julia $(VERSION), $(Threads.nthreads()) threads")

events = load_hepmc3_events(joinpath(@__DIR__, "..", "data", "sk_example_PU.hepmc"); maxevents=100)
println("Loaded $(length(events)) events\n")

backends = [:ns64, :ot64, :ns32, :ot32]
setups = [
    ("Euclidean_norm",   EuclideanMetric(), true),
    ("Euclidean_unnorm", EuclideanMetric(), false),
    ("EtaPhi_norm",      EtaPhiMetric(),    true),
]
splits = [("10v90", 1:10, 11:100), ("50v50", 1:50, 51:100)]

results = []
for (split_name, ra, rb) in splits
    ea, eb = events[ra], events[rb]
    na, nb = length(ea), length(eb)
    # warmup
    emds(ea[1:2], eb[1:2]; R=1.0, beta=1.0, norm=true, backend=:ns64)

    for (setup_name, metric, norm) in setups
        for backend in backends
            t0 = time_ns()
            D = emds(ea, eb; R=1.0, beta=1.0, norm=norm, backend=backend, metric=metric)
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
