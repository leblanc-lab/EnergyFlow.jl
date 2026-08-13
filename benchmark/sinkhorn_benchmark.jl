# Sinkhorn accuracy/cost benchmark — Julia EnergyFlow.jl
# Usage: cd benchmark && julia --project=. sinkhorn_benchmark.jl
#
# The `:sinkhorn` backend solves the entropy-regularised transport problem, so
# unlike the exact backends it cannot be benchmarked on time alone: a Sinkhorn
# timing is only meaningful next to the error of the value it produced. Every
# point here therefore records both, against the exact `:ns64` value of the same
# problem, and writes result/sinkhorn_julia.md. sinkhorn_plot.py turns that into
# paper/sinkhorn.png; sinkhorn_benchmark_python.py is the POT counterpart.
#
# Two things this measurement has to get right, both learned the hard way:
#
#   * Convergence must be recorded, not assumed. At the default max_iter the
#     small-epsilon points stop early, and an unconverged Sinkhorn run reports a
#     cost that is *too low* — the scaling iterates have not reached a feasible
#     plan, so <C,P> is summed over marginals that do not match the inputs. Read
#     naively that looks like accuracy improving and then mysteriously degrading
#     as epsilon falls, when it is only the iteration cap. Points are flagged so
#     the figure can draw them distinctly instead of asserting an error that is
#     really a truncation artifact.
#
#   * The reference must be the exact solve of the *same* problem, not a
#     published number: relative error at epsilon = 0.002 is ~1e-5, far below
#     the level at which a hardcoded reference would still be trustworthy.
#
# Sinkhorn supports EuclideanMetric only (see EMD.jl), and is run normalized —
# that is the balanced problem the algorithm natively solves, so the comparison
# is not confounded by the fictitious-particle construction.

include("envinfo.jl")

using EnergyFlow
using DelimitedFiles
using Printf
using Statistics

# Regularisation strengths. The upper end is where Sinkhorn is cheap and
# visibly biased; the lower end is where it approaches the exact value and the
# iteration count, not epsilon, becomes the binding constraint.
const EPSILONS = haskey(ENV, "ENERGYFLOW_SINKHORN_EPS") ?
    parse.(Float64, split(ENV["ENERGYFLOW_SINKHORN_EPS"], ",")) :
    [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]

# The measurement is a grid of (n, epsilon), built as the union of two sweeps so
# the cost stays bounded: the full epsilon sweep runs at a few multiplicities,
# and the multiplicity sweep runs at two epsilon values. Sinkhorn cost grows
# like n^2 per iteration *and* the iteration count grows as epsilon falls, so a
# full cross product at large n and small epsilon would dominate the job for
# points no figure needs.
const EPS_SWEEP_SIZES = haskey(ENV, "ENERGYFLOW_SINKHORN_EPS_SIZES") ?
    parse.(Int, split(ENV["ENERGYFLOW_SINKHORN_EPS_SIZES"], ",")) :
    [50, 100, 200]
const SCALING_SIZES = haskey(ENV, "ENERGYFLOW_SINKHORN_SIZES") ?
    parse.(Int, split(ENV["ENERGYFLOW_SINKHORN_SIZES"], ",")) :
    [10, 50, 100, 200, 500]
const SCALING_EPSILONS = [0.05, 0.01]

# Iterations per epsilon level. Deliberately far above the 5000 default: the
# point of the sweep is to see what epsilon costs when it is allowed to
# converge, and the default cap truncates every point below epsilon ~ 0.01.
const MAX_ITER = parse(Int, get(ENV, "ENERGYFLOW_SINKHORN_MAXITER", "100000"))
const TOL      = 1e-9

const TARGET_SECONDS = parse(Float64, get(ENV, "ENERGYFLOW_BENCH_TARGET", "2.0"))
const MIN_REPS       = 3
const MAX_REPS       = 10_000
# Past this per-call cost, repetitions buy nothing: the solve is long enough
# that timer noise is irrelevant, and MIN_REPS would triple the job's runtime.
const LONG_SOLVE_SECONDS = 10.0
const LONG_SOLVE_REPS    = 1

const SINK = Ref(0.0)

load_event(path) = readdlm(path, ',', Float64)

"""
    warmup()

Compile every entry point this script times, once, on a trivial problem.

The other benchmarks in this directory warm up per measured point because they
sweep several *backends*, which compile separately. This one sweeps a single
backend over ε and n, and neither ε nor n changes the types involved — so one
call per entry point compiles everything, and a per-point warmup would only
re-pay a cost that is already amortized. That matters here in a way it does not
elsewhere: a warmup at the expensive end of this sweep is a two-minute solve,
and repeating it at every point would roughly double the job for no accuracy.
"""
function warmup()
    ev0 = load_event(joinpath(@__DIR__, "..", "data", "event0_n10.csv"))
    ev1 = load_event(joinpath(@__DIR__, "..", "data", "event1_n10.csv"))
    SINK[] += emd(ev0, ev1; R=1.0, beta=1.0, norm=true, backend=:ns64)
    SINK[] += emd_sinkhorn(ev0, ev1; R=1.0, beta=1.0, norm=true, epsilon=0.5,
                           max_iter_sinkhorn=100, sinkhorn_tol=TOL, annealing=true)
    ws = SinkhornWorkspace(10, 10; beta=1.0, R=1.0, norm=true, epsilon=0.5,
                           max_iter=100, tol=TOL, annealing=true)
    SINK[] += emd_sinkhorn!(ws, ev0, ev1)
    return nothing
end

"""
    timed(f) -> (median_s, min_s, reps)

Time `f()` repeatedly, returning the median and minimum per-call wall time.

The repetition count is chosen from a pilot measurement so each point gets
roughly `TARGET_SECONDS` of timing, as in `single_emd_benchmark.jl`. Two
departures, both because Sinkhorn points here run from microseconds to minutes:

  * Compilation is handled once by `warmup()` above rather than per point.
  * A solve slower than `LONG_SOLVE_SECONDS` reports the pilot measurement
    itself instead of running further repetitions. The pilot is already a
    warmed, timed call, so an extra repetition would produce the same number at
    twice the cost — and at a hundred seconds per solve, timer noise is far
    below any difference the figure resolves.
"""
function timed(f)
    t0 = time_ns()
    SINK[] += f()
    est = (time_ns() - t0) / 1e9
    if est > LONG_SOLVE_SECONDS
        return est, est, LONG_SOLVE_REPS
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

# The (n, epsilon) points to measure, as the union described above.
function grid_points()
    points = Set{Tuple{Int,Float64}}()
    for n in EPS_SWEEP_SIZES, eps in EPSILONS
        push!(points, (n, eps))
    end
    for n in SCALING_SIZES, eps in SCALING_EPSILONS
        push!(points, (n, eps))
    end
    return sort!(collect(points))
end

print_env()
warmup()

const POINTS = grid_points()
const SIZES  = sort!(unique(first.(POINTS)))
println("Sinkhorn sweep: $(length(POINTS)) (n, ε) points over n ∈ $SIZES")
println("max_iter = $MAX_ITER per ε-level, tol = $TOL, annealing = true, norm = true\n")

results = NamedTuple[]
for n in SIZES
    ev0 = load_event(joinpath(@__DIR__, "..", "data", "event0_n$(n).csv"))
    ev1 = load_event(joinpath(@__DIR__, "..", "data", "event1_n$(n).csv"))

    # Exact reference for this n: both the value every Sinkhorn point is scored
    # against and the timing the figure compares against, so the two never come
    # from different events or different runs.
    exact = emd(ev0, ev1; R=1.0, beta=1.0, norm=true, backend=:ns64)
    med, mn, reps = timed() do
        emd(ev0, ev1; R=1.0, beta=1.0, norm=true, backend=:ns64)
    end
    push!(results, (n=n, epsilon=NaN, backend="ns64", median_s=med, min_s=mn,
                    reps=reps, value=exact, relerr=0.0, iters=0, converged=true))
    @printf("n=%-5d exact  :ns64            %10.3f ms  value=%.9g\n", n, med * 1e3, exact)

    for eps in sort([e for (m, e) in POINTS if m == n]; rev=true)
        # Timed through the allocating entry point, so the number includes
        # workspace allocation and cost-matrix construction exactly as the
        # exact-backend timings do.
        med, mn, reps = timed() do
            emd_sinkhorn(ev0, ev1; R=1.0, beta=1.0, norm=true, epsilon=eps,
                         max_iter_sinkhorn=MAX_ITER, sinkhorn_tol=TOL, annealing=true)
        end

        # One extra workspace solve purely for diagnostics. The workspace form
        # is the only way to see whether the run converged, and an unconverged
        # point's "error" is a truncation artifact rather than the entropic bias
        # the sweep is meant to measure.
        ws = SinkhornWorkspace(size(ev0, 1), size(ev1, 1); beta=1.0, R=1.0, norm=true,
                               epsilon=eps, max_iter=MAX_ITER, tol=TOL, annealing=true)
        val = emd_sinkhorn!(ws, ev0, ev1)
        relerr = (val - exact) / exact

        push!(results, (n=n, epsilon=eps, backend="sinkhorn", median_s=med, min_s=mn,
                        reps=reps, value=val, relerr=relerr, iters=ws.n_iters,
                        converged=ws.converged))
        @printf("n=%-5d ε=%-7g :sinkhorn       %10.3f ms  value=%.9g  relerr=%+.3e  iters=%-7d %s\n",
                n, eps, med * 1e3, val, relerr, ws.n_iters,
                ws.converged ? "converged" : "NOT CONVERGED")
    end
    println()
end

n_sinkhorn  = count(r -> r.backend == "sinkhorn", results)
unconverged = count(r -> r.backend == "sinkhorn" && !r.converged, results)
if unconverged > 0
    println("$unconverged of $n_sinkhorn Sinkhorn points hit the $MAX_ITER-iteration cap. Their values are")
    println("biased low (the scaling iterates never reached a feasible plan) and are flagged in the result")
    println("table; the figure draws them as open markers. Raise ENERGYFLOW_SINKHORN_MAXITER to push the")
    println("reachable ε lower.\n")
end

mkpath(joinpath(@__DIR__, "result"))
open(joinpath(@__DIR__, "result", "sinkhorn_julia.md"), "w") do io
    println(io, "# Sinkhorn Accuracy vs Cost — Julia EnergyFlow.jl\n")
    println(io, "Approximate (`:sinkhorn`) EMD against the exact (`:ns64`) value of the same")
    println(io, "problem, over entropic regularisation strength ε and multiplicity n.")
    println(io, "Euclidean metric, `norm=true`, ε-annealing on, `max_iter=$MAX_ITER` per")
    println(io, "ε-level, `tol=$TOL`.\n")
    println(io, "`RelErr` is (Sinkhorn − exact) / exact. `Converged` is the marginal-violation")
    println(io, "criterion: a `false` row stopped at the iteration cap, and its value is biased")
    println(io, "*low* because the iterates never reached a feasible transport plan — that is a")
    println(io, "truncation artifact, not the entropic bias. `ns64` rows are the exact reference")
    println(io, "(ε reported as `exact`), timed the same way.\n")
    println(io, "Time is the median over `Reps` repetitions after a per-configuration warmup,")
    println(io, "and includes workspace allocation and cost-matrix construction.\n")
    write_env_block(io)
    println(io, "| n | Epsilon | Backend | Median (s) | Min (s) | Reps | Value | RelErr | Iters | Converged |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|")
    for r in results
        eps_str = isnan(r.epsilon) ? "exact" : @sprintf("%.9g", r.epsilon)
        @printf(io, "| %d | %s | %s | %.9g | %.9g | %d | %.12g | %.6e | %d | %s |\n",
                r.n, eps_str, r.backend, r.median_s, r.min_s, r.reps,
                r.value, r.relerr, r.iters, r.converged)
    end
end
println("Results saved to result/sinkhorn_julia.md")
