# Event Isotropy Benchmark — Julia EnergyFlow.jl
# Usage: cd benchmark && julia --project=. -t 8 isotropy_benchmark.jl
#
# Computes the event isotropy (arXiv:2004.06125) of each event against
# quasi-uniform ring, cylinder (hadron-collider) and sphere (lepton-collider)
# references, per solver backend. Mean isotropy values are reported alongside
# timings so results can be checked numerically against the POT implementation
# (isotropy_benchmark_python.py).

using EnergyFlow
using Printf
using Statistics

ring_event(ev) = ev[:, [1, 3]]

const events = load_hepmc3_events(joinpath(@__DIR__, "..", "data", "sk_example_PU.hepmc"); maxevents=100)
const ymax = 4.0
const selected = select_events(events, ymax)

# Sphere (lepton-collider) uses full 3-momenta with energy weighting. The
# forward pp sample here reads as near-pencil (isotropy ≈ 1); it is used as a
# solver stress-test / POT cross-check, not as physical e⁺e⁻ data.
const momenta = load_hepmc3_momenta(joinpath(@__DIR__, "..", "data", "sk_example_PU.hepmc"); maxevents=100)
const sphere_selected = select_sphere_events(momenta)

println("Julia $(VERSION), $(Threads.nthreads()) threads")
println("Loaded $(length(events)) events, $(length(selected)) after |y| ≤ $ymax selection\n")

setups = [
    ("ring128",   [ring_event(ev) for ev in selected], ring_reference(128),          ring_cos_metric()),
    ("ring2",     [ring_event(ev) for ev in selected], ring_reference(2),            ring_cos_metric()),
    ("cyl16",     selected,                            cylinder_reference(16, ymax), cylinder_metric(ymax)),
    ("sphere192", sphere_selected,                     sphere_reference(2),          sphere_cos_metric()),
    ("sphere48",  sphere_selected,                     sphere_reference(1),          sphere_cos_metric()),
]

results = []
for (setup_name, evs, ref, metric) in setups
    emds(evs[1:2], [ref]; R=1.0, beta=1.0, norm=true, metric=metric)  # warmup
    for backend in [:ns64, :ot64, :ns32, :ot32]
        t0 = time_ns()
        vals = emds(evs, [ref]; R=1.0, beta=1.0, norm=true, backend=backend, metric=metric)
        t1 = time_ns()
        elapsed = (t1 - t0) / 1e9
        push!(results, (setup=setup_name, backend=backend, time_s=elapsed,
                        events=length(evs), mean_iso=mean(vals)))
        @printf("%-8s %-6s %8.3f s  (%d events, mean isotropy %.6f)\n",
                setup_name, backend, elapsed, length(evs), mean(vals))
    end
end

# Save results — file is tagged with the thread count so isotropy_compare.jl can
# show single-thread (solver-fair) and multi-thread (workload) side by side. Run
# this benchmark once per thread count, e.g. `julia --project=.` and
# `julia --project=. -t 8`.
nthr = Threads.nthreads()
mkpath(joinpath(@__DIR__, "result"))
outfile = "isotropy_julia_t$(nthr).md"
open(joinpath(@__DIR__, "result", outfile), "w") do io
    println(io, "# Event Isotropy Benchmark — Julia EnergyFlow.jl")
    println(io, "\nJulia $(VERSION), $(nthr) thread(s)")
    println(io, "ring: N points in φ, (π/(π-2))·(1-cos Δφ); cylinder: 16×$(floor(Int, ymax*16/π)) grid, |y| ≤ $ymax")
    println(io, "sphere: HEALPix unit sphere (192/48 points), 2·(1-cos θ) on 3-momentum directions\n")
    println(io, "| Setup | Backend | Time (s) | Events | Mean isotropy |")
    println(io, "|---|---|---|---|---|")
    for r in results
        @printf(io, "| %s | %s | %.3f | %d | %.6f |\n",
                r.setup, r.backend, r.time_s, r.events, r.mean_iso)
    end
end
println("\nResults saved to result/$outfile")
