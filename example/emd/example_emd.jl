# EnergyFlow.jl — EMD Example
push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "src"))
using EnergyFlow

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
