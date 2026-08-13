# Network-simplex pivot-rule benchmark — Julia EnergyFlow.jl
# Usage: cd benchmark && julia --project=. -t 8 pivot_benchmark.jl
#
# Compares the three entering-arc rules in NetworkSimplex.jl on single large
# solves:
#
#   :serial         — block search, one thread. The default, and the only rule
#                     any other benchmark in this directory exercises.
#   :parallel_block — block search with the block split across a persistent task
#                     group; the Kara & Ozturan scheme (arXiv/DOI in paper.bib).
#   :full_parallel  — every pivot scans all arcs in parallel and takes the
#                     globally most-negative reduced cost, i.e. a parallel
#                     Dantzig rule. Documented in the source as a
#                     correctness/performance baseline rather than a production
#                     mode; included so the comparison has both ends of the
#                     trade-off, not because it is expected to win.
#
# Single-pair, not pairwise, and that is the point rather than a limitation.
# Parallel pivoting parallelises *within* one solve, while `emds` already gets
# embarrassing parallelism *across* pairs; running both would oversubscribe the
# machine and measure scheduler contention rather than either scheme. So the
# regime where a parallel pivot rule can help at all is the one measured here:
# a single solve large enough that one entering-arc search is worth splitting.
#
# `pivot_mode` is not reachable through `emd`/`emds` — those construct their
# workspaces internally — so this sets `ws.ns.pivot_mode` on an explicit
# workspace, the same way test/test_emdsolver.jl does. That is a deliberate
# reach into the struct, and the only way to measure the mode without an API
# change.
#
# Writes result/pivot_julia_t{N}.md, tagged by thread count: Julia fixes its
# thread count at startup, so the thread sweep is over separate invocations.
# pivot_plot.py turns the collection into paper/pivot.png.

include("envinfo.jl")

using EnergyFlow
using DelimitedFiles
using Printf
using Statistics

const SIZES = haskey(ENV, "ENERGYFLOW_PIVOT_SIZES") ?
    parse.(Int, split(ENV["ENERGYFLOW_PIVOT_SIZES"], ",")) :
    [200, 500, 1000, 2000, 3000]

const MODES = [:serial, :parallel_block, :full_parallel]

# :full_parallel touches every arc on every pivot, so its cost grows as
# (arcs x pivots) with no block to bound the scan — at n = 1000 that is already
# 4.4 billion arc evaluations for one solve. It is a baseline, not a candidate,
# so by default it is not run at the sizes where it would dominate the job.
const FULL_PARALLEL_MAX_N = parse(Int, get(ENV, "ENERGYFLOW_PIVOT_FULLMAX", "1000"))

const TARGET_SECONDS = parse(Float64, get(ENV, "ENERGYFLOW_BENCH_TARGET", "2.0"))
const MIN_REPS       = 3
const MAX_REPS       = 5_000
const LONG_SOLVE_SECONDS = 10.0

const SINK = Ref(0.0)

load_event(path) = readdlm(path, ',', Float64)

"""
    timed(f) -> (median_s, min_s, reps)

Time `f()` after an untimed warmup. Same adaptive scheme as the other scripts
here: a pilot sizes the repetition count against `TARGET_SECONDS`, and a solve
slower than `LONG_SOLVE_SECONDS` reports the pilot rather than repeating a call
that is already far above timer noise.
"""
function timed(f)
    SINK[] += f()                                   # warm up this exact mode
    t0 = time_ns()
    SINK[] += f()
    est = (time_ns() - t0) / 1e9
    if est > LONG_SOLVE_SECONDS
        return est, est, 1
    end
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

nthr = Threads.nthreads()
println("Pivot-rule sweep at $nthr thread(s); sizes $SIZES")
println(":full_parallel capped at n <= $FULL_PARALLEL_MAX_N (baseline mode, cost grows as arcs x pivots)\n")

results = NamedTuple[]
for n in SIZES
    ev0 = load_event(joinpath(@__DIR__, "..", "data", "event0_n$(n).csv"))
    ev1 = load_event(joinpath(@__DIR__, "..", "data", "event1_n$(n).csv"))

    # Reference value from the default rule. Every mode solves the same problem,
    # so they must agree: a pivot rule changes which vertex of the polytope is
    # visited next, never the optimum. A disagreement beyond float noise is a
    # bug in the rule, not a tolerance, so it is recorded per row.
    reference = emd(ev0, ev1; R=1.0, beta=1.0, norm=true, backend=:ns64)

    for mode in MODES
        if mode === :full_parallel && n > FULL_PARALLEL_MAX_N
            continue
        end
        ws = EMDWorkspace(n, n; beta=1.0, R=1.0, norm=true)
        ws.ns.pivot_mode = mode

        med, mn, reps = timed() do
            emd!(ws, ev0, ev1)
        end

        # One more solve to capture the diagnostics for this configuration.
        val = emd!(ws, ev0, ev1)
        if ws.ns.status !== :optimal
            error("n=$n mode=$mode did not reach :optimal (status=$(ws.ns.status)); " *
                  "the timing would describe a truncated solve.")
        end

        push!(results, (n=n, mode=string(mode), median_s=med, min_s=mn, reps=reps,
                        value=val, delta=val - reference,
                        iters=ws.ns.n_iters, arc_scans=ws.ns.n_arc_scans))
        @printf("n=%-5d %-15s %9.4f s  (%d reps)  iters=%-7d arc_scans=%-13d Δ=%+.2e\n",
                n, mode, med, reps, ws.ns.n_iters, ws.ns.n_arc_scans, val - reference)
        flush(stdout)
    end
end

mkpath(joinpath(@__DIR__, "result"))
outfile = "pivot_julia_t$(nthr).md"
open(joinpath(@__DIR__, "result", outfile), "w") do io
    println(io, "# Network-Simplex Pivot Rules — Julia EnergyFlow.jl\n")
    println(io, "Single-pair `emd!` between two events of equal multiplicity n, at")
    println(io, "$nthr thread(s), for each entering-arc rule. `:parallel_block` is the")
    println(io, "Kara & Ozturan scheme; `:serial` is the package default and the rule every")
    println(io, "other benchmark here uses; `:full_parallel` is a parallel Dantzig baseline.\n")
    println(io, "`Iters` is the pivot count and `Delta` the difference from the `:ns64`")
    println(io, "reference value. A pivot rule changes the path taken to the optimum, not the")
    println(io, "optimum itself, so `Delta` is float noise and `Iters` is where the rules")
    println(io, "actually differ — a rule that reaches the same answer in fewer pivots is")
    println(io, "making better entering-arc choices, whether or not it is faster in wall time.\n")
    println(io, "`ArcScans` counts arc evaluations in the entering-arc search. It is not")
    println(io, "instrumented for `:parallel_block`, which reports 0 rather than a measurement.\n")
    println(io, "Every solve is asserted to reach `:optimal` before being recorded.\n")
    write_env_block(io)
    println(io, "| n | Mode | Median (s) | Min (s) | Reps | Value | Delta | Iters | ArcScans |")
    println(io, "|---|---|---|---|---|---|---|---|---|")
    for r in results
        @printf(io, "| %d | %s | %.9g | %.9g | %d | %.12g | %.3e | %d | %d |\n",
                r.n, r.mode, r.median_s, r.min_s, r.reps, r.value, r.delta,
                r.iters, r.arc_scans)
    end
end
println("\nResults saved to result/$outfile")
