#!/usr/bin/env julia
# Benchmark: load HepMC3 events, compute pairwise EMDs, save results to CSV.
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using EnergyFlow
using DelimitedFiles
using Printf

const DATADIR = joinpath(@__DIR__, "..", "data")
const HEPMC_FILE = joinpath(DATADIR, "sk_example_PU.hepmc")

# ── 1. Load events ──────────────────────────────────────────────────
println("Loading HepMC3 events...")
t_load = @elapsed events = load_hepmc3_events(HEPMC_FILE; maxevents=100)
@printf("Loaded %d events in %.3f s\n", length(events), t_load)
for i in 1:min(5, length(events))
    println("  Event $i: $(size(events[i], 1)) particles")
end

# ── 2. Define splits ────────────────────────────────────────────────
splits = [
    ("10v90",  events[1:10],  events[11:100]),
    ("50v50",  events[1:50],  events[51:100]),
]

backends = [
    ("ns64", :ns64),
    ("ot64", :ot64),
]

norms = [true, false]

# ── 3. Compute EMDs and save CSVs ──────────────────────────────────
println("\n=== Computing EMDs ===")
for (split_name, evA, evB) in splits
    for (backend_name, backend_sym) in backends
        for norm_val in norms
            norm_str = norm_val ? "norm" : "unnorm"
            label = "$(split_name)_$(backend_name)_$(norm_str)"
            println("\n--- $label ---")

            # Warmup (first call triggers compilation)
            _ = emds(evA[1:1], evB[1:1]; backend=backend_sym, R=1.0, beta=1.0, norm=norm_val)

            # Timed run
            t = @elapsed D = emds(evA, evB; backend=backend_sym, R=1.0, beta=1.0, norm=norm_val)
            @printf("  %s: %.4f s  |  shape (%d, %d)  |  mean=%.6e  max=%.6e\n",
                    label, t, size(D, 1), size(D, 2), sum(D)/length(D), maximum(D))

            # Save to CSV
            csv_path = joinpath(DATADIR, "julia_emds_$(label).csv")
            open(csv_path, "w") do io
                for i in 1:size(D, 1)
                    for j in 1:size(D, 2)
                        j > 1 && print(io, ",")
                        @printf(io, "%.15e", D[i, j])
                    end
                    println(io)
                end
            end
            println("  Saved → $csv_path")
        end
    end
end

# ── 4. Benchmark timing (3 trials each) ────────────────────────────
println("\n=== Benchmark Timing (3 trials) ===")
println(@sprintf("%-30s %10s %10s %10s", "Config", "Trial1", "Trial2", "Trial3"))
for (split_name, evA, evB) in splits
    for (backend_name, backend_sym) in backends
        for norm_val in norms
            norm_str = norm_val ? "norm" : "unnorm"
            label = "$(split_name)_$(backend_name)_$(norm_str)"

            times = Float64[]
            for trial in 1:3
                t = @elapsed emds(evA, evB; backend=backend_sym, R=1.0, beta=1.0, norm=norm_val)
                push!(times, t)
            end
            println(@sprintf("%-30s %10.4f %10.4f %10.4f", label, times...))
        end
    end
end

println("\nDone.")
