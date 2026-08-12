# Event Isotropy Benchmark — Julia EnergyFlow.jl
# Usage: cd benchmark && julia --project=. -t 8 isotropy_benchmark.jl
#
# Computes the event isotropy (arXiv:2004.06125) of each event against
# quasi-uniform ring, cylinder (hadron-collider) and sphere (lepton-collider)
# references, per solver backend. Mean isotropy values are reported alongside
# timings so results can be checked numerically against the POT implementation
# (isotropy_benchmark_python.py).

include("envinfo.jl")

using EnergyFlow
using Printf
using Statistics

# Timing budget per (setup, backend) point. The isotropy workloads are small —
# the fastest are milliseconds — so a single measurement is dominated by timer
# noise and thread-startup effects. Repeat each point and report the median.
const TARGET_SECONDS = 3.0
const MIN_REPS       = 3
const MAX_REPS       = 2_000
# Above this per-call cost, the minimum repetition count would dominate the
# runtime (sphere3072 takes tens of seconds per pass), and the measurement is
# long enough that timer noise is irrelevant anyway.
const LONG_SOLVE_SECONDS = 15.0
const LONG_SOLVE_REPS    = 2

"""
    timed(f) -> (median_s, min_s, reps)

Time `f()` repeatedly, returning the median and minimum wall time per call.

`f` is warmed up before any measurement. This matters more than it looks: the
previous version of this benchmark warmed up only the default backend and then
timed all four, so `:ot64`, `:ns32` and `:ot32` each reported their own
compilation as though it were solve time.
"""
function timed(f)
    f()
    t0 = time_ns(); f(); est = (time_ns() - t0) / 1e9
    reps = est > LONG_SOLVE_SECONDS ? LONG_SOLVE_REPS :
           clamp(ceil(Int, TARGET_SECONDS / max(est, 1e-9)), MIN_REPS, MAX_REPS)
    times = Vector{Float64}(undef, reps)
    for i in 1:reps
        t0 = time_ns()
        f()
        times[i] = (time_ns() - t0) / 1e9
    end
    return median(times), minimum(times), reps
end

const events = load_hepmc3_events(joinpath(@__DIR__, "..", "data", "sk_example_PU.hepmc"); maxevents=100)
const ymax = 4.0
const selected = select_rapidity(events, ymax)

# Sphere (lepton-collider) uses full 3-momenta with energy weighting. The
# forward pp sample here reads as near-pencil (isotropy ≈ 1); it is used as a
# solver stress-test / POT cross-check, not as physical e⁺e⁻ data.
const momenta = load_hepmc3_momenta(joinpath(@__DIR__, "..", "data", "sk_example_PU.hepmc"); maxevents=100)
const sphere_selected = select_sphere_events(momenta)

print_env()
println("Loaded $(length(events)) events, $(length(selected)) after |y| ≤ $ymax selection\n")

setups = [
    ("ring128",   [EnergyFlow._ring_event(ev) for ev in selected], ring_reference(128),          ring_cos_metric()),
    ("ring2",     [EnergyFlow._ring_event(ev) for ev in selected], ring_reference(2),            ring_cos_metric()),
    ("cyl16",     selected,                            cylinder_reference(16, ymax), cylinder_metric(ymax)),
    ("sphere192", sphere_selected,                     sphere_reference(2),          sphere_cos_metric()),
    ("sphere48",  sphere_selected,                     sphere_reference(1),          sphere_cos_metric()),
    # 12·(2^n)² points: n=3 -> 768, n=4 -> 3072. The finer references are the
    # ones that matter for physics — 3072 is the resolution at which the
    # spherical reference stops limiting the observable — and they are also
    # where the solver is actually stressed.
    ("sphere768",  sphere_selected,                    sphere_reference(3),          sphere_cos_metric()),
    ("sphere3072", sphere_selected,                    sphere_reference(4),          sphere_cos_metric()),
]

results = []
for (setup_name, evs, ref, metric) in setups
    for backend in [:ns64, :ot64, :ns32, :ot32]
        # Each backend is warmed up inside timed(), so no backend pays for its
        # own compilation in the reported number.
        med, mn, reps = timed() do
            emds(evs, [ref]; R=1.0, beta=1.0, norm=true, backend=backend, metric=metric)
        end
        vals = emds(evs, [ref]; R=1.0, beta=1.0, norm=true, backend=backend, metric=metric)
        push!(results, (setup=setup_name, backend=backend, time_s=med, min_s=mn,
                        reps=reps, events=length(evs), mean_iso=mean(vals)))
        @printf("%-8s %-6s %8.4f s  (min %8.4f s, %d reps, %d events, mean isotropy %.6f)\n",
                setup_name, backend, med, mn, reps, length(evs), mean(vals))
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
    println(io, "# Event Isotropy Benchmark — Julia EnergyFlow.jl\n")
    println(io, "ring: N points in φ, (π/(π-2))·(1-cos Δφ); cylinder: 16×$(floor(Int, ymax*16/π)) grid, |y| ≤ $ymax")
    println(io, "sphere: HEALPix unit sphere (192/48 points), 2·(1-cos θ) on 3-momentum directions\n")
    println(io, "Time is the median over `Reps` repetitions, after a per-backend warmup.\n")
    write_env_block(io)
    println(io, "| Setup | Backend | Time (s) | Min (s) | Reps | Events | Mean isotropy |")
    println(io, "|---|---|---|---|---|---|---|")
    for r in results
        @printf(io, "| %s | %s | %.4f | %.4f | %d | %d | %.6f |\n",
                r.setup, r.backend, r.time_s, r.min_s, r.reps, r.events, r.mean_iso)
    end
end
println("\nResults saved to result/$outfile")
