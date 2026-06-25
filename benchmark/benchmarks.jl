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

