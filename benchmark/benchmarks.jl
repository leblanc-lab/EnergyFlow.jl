using BenchmarkTools
using EnergyFlow

const events = load_hepmc3_events(joinpath(@__DIR__, "..", "data", "sk_example_PU.hepmc"); maxevents=100)

const SUITE = BenchmarkGroup()

for (split_name, ra, rb) in [("10v90", 1:10, 11:100), ("50v50", 1:50, 51:100)]
    ea, eb = events[ra], events[rb]
    g = SUITE[split_name] = BenchmarkGroup()
    for (setup_name, metric, norm) in [
        ("Euclidean_norm",   EuclideanMetric(), true),
        ("Euclidean_unnorm", EuclideanMetric(), false),
        ("EtaPhi_norm",      EtaPhiMetric(),    true),
    ]
        h = g[setup_name] = BenchmarkGroup()
        for backend in [:ns64, :ot64, :ns32, :ot32]
            h[string(backend)] = @benchmarkable emds($ea, $eb; R=1.0, beta=1.0,
                                                     norm=$norm, backend=$backend,
                                                     metric=$metric)
        end
    end
end

# ── Event isotropy (arXiv:2004.06125, arXiv:2305.16930) ──
# EMD of each event against a quasi-uniform ring/cylinder (hadron collider) or
# sphere (lepton collider) reference.
let
    ymax = 4.0
    selected = select_rapidity(events, ymax)
    sphere_selected = select_sphere_events(
        load_hepmc3_momenta(joinpath(@__DIR__, "..", "data", "sk_example_PU.hepmc"); maxevents=100))
    setups = [
        ("ring128",   [EnergyFlow._ring_event(ev) for ev in selected], ring_reference(128),          ring_cos_metric()),
        ("ring2",     [EnergyFlow._ring_event(ev) for ev in selected], ring_reference(2),            ring_cos_metric()),
        ("cyl16",     selected,                            cylinder_reference(16, ymax), cylinder_metric(ymax)),
        ("sphere192", sphere_selected,                     sphere_reference(2),          sphere_cos_metric()),
    ]
    g = SUITE["isotropy"] = BenchmarkGroup()
    for (setup_name, evs, ref, metric) in setups
        h = g[setup_name] = BenchmarkGroup()
        for backend in [:ns64, :ot64, :ns32, :ot32]
            h[string(backend)] = @benchmarkable emds($evs, $([ref]); R=1.0, beta=1.0,
                                                     norm=true, backend=$backend,
                                                     metric=$metric)
        end
    end
end

