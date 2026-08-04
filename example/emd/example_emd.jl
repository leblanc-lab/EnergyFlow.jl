# EnergyFlow.jl — EMD Example
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

using EnergyFlow
using Plots

# Load events from HepMC3
events = load_hepmc3_events(joinpath(@__DIR__, "..", "..", "data", "sk_example_PU.hepmc"); maxevents=20)
println("Loaded $(length(events)) events")

# ── Single EMD ──
# Default: ns64 backend, Euclidean distance
val = emd(events[1], events[2]; R=1.0, beta=1.0, norm=true)
println("\nEMD (ns64, Euclidean): $val")

# ot64 backend, EtaPhi distance
val2 = emd(events[1], events[2]; R=1.0, beta=1.0, norm=true, backend=:ot64, metric=EtaPhiMetric())
println("EMD (ot64, EtaPhi):    $val2")

# Other backends: :ns32, :ot32 (Float32), :sinkhorn (approximate)
# Other metrics:  SquaredEuclideanMetric(), PrecomputedMetric(matrix), CustomMetric(f)

# ── Pairwise EMD ──
# Cross-pairwise: 10 vs 10 → 10×10 matrix
D = emds(events[1:10], events[11:20]; R=1.0, beta=1.0, norm=true)
println("\nCross-pairwise (10×10): D[1,1]=$(D[1,1])")

# Self-pairwise: flat upper-triangular vector
dists = emds(events[1:10]; R=1.0, beta=1.0, norm=true)
println("Self-pairwise ($(length(dists)) pairs): dists[1]=$(dists[1])")

# With EtaPhi metric
D2 = emds(events[1:10], events[11:20]; R=1.0, beta=1.0, norm=true, metric=EtaPhiMetric())
println("Cross EtaPhi: D[1,1]=$(D2[1,1])")

# ── Workspace reuse ──
# For repeated single-pair calls, pre-allocate a workspace sized for the
# largest events and use emd! (parameters live in the workspace).
n = maximum(size(e, 1) for e in events)
ws = EMDWorkspace(n, n; beta=1.0, R=1.0, norm=true)
val3 = emd!(ws, events[1], events[2])
println("\nEMD via reusable workspace: $val3")

# ── Transport plan visualisation ──
# Ask for the plan so we can draw which particle in event 0 moves to which
# particle in event 1. Blue points are event 0, red points are event 1.
ev0 = events[1]
ev1 = events[2]
val4, plan = emd(ev0, ev1; R=1.0, beta=1.0, norm=true, return_flow=true)
println("EMD with transport plan: $val4")

function plot_transport_plan(ev0, ev1, plan)
	p = plot(; legend=false, aspect_ratio=:equal, xlabel="η", ylabel="φ",
			 title="EMD transport plan", background_color=:white)

	scatter!(p, ev0[:, 2], ev0[:, 3]; color=:dodgerblue, markersize=6,
			 markerstrokecolor=:dodgerblue)
	scatter!(p, ev1[:, 2], ev1[:, 3]; color=:tomato, markersize=6,
			 markerstrokecolor=:tomato)

	maxflow = maximum(plan)
	for i in 1:size(plan, 1), j in 1:size(plan, 2)
		flow = plan[i, j]
		flow <= 0 && continue
		strength = maxflow > 0 ? flow / maxflow : 0.0
		plot!(p, [ev0[i, 2], ev1[j, 2]], [ev0[i, 3], ev1[j, 3]];
			  color=:gray35, alpha=clamp(0.2 + 0.8 * strength, 0.2, 1.0),
			  linewidth=1 + 5 * strength)
	end

	return p
end

transport_plot = plot_transport_plan(ev0, ev1, plan)
savefig(transport_plot, joinpath(@__DIR__, "emd_transport_plan.png"))
println("Saved transport visualisation to emd_transport_plan.png")
