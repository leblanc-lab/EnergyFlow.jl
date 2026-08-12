# Build a large event sample for the large-matrix pairwise benchmark.
# Usage: cd benchmark && julia --project=. make_large_sample.jl [nevents]
#
# data/sk_example_PU.hepmc holds 100 events, which caps a pairwise matrix at
# 100x100. Analyses routinely want 1000x1000 and larger, so this script builds
# an N-event sample by drawing the 100 real events with replacement and giving
# each draw an independent random azimuthal rotation.
#
# That is legitimate for a *timing* benchmark and nothing more: a rotation in
# phi is a symmetry of the detector geometry, so each drawn event remains a
# physically valid event, and the multiplicity distribution — which is what
# actually sets the cost of a transport solve — is preserved exactly. The
# resulting distance matrix is not a physics measurement and should not be
# interpreted as one.
#
# The sample is written to a CSV that both the Julia and the Python benchmark
# read, so the two sides solve byte-identical problems. Generating it
# separately in each language would not guarantee that.

include("envinfo.jl")

using EnergyFlow
using Printf
using Random

const NEVENTS = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000
const SEED    = 20260812
const OUTFILE = joinpath(@__DIR__, "result", "large_sample.csv")

print_env()

events = load_hepmc3_events(joinpath(@__DIR__, "..", "data", "sk_example_PU.hepmc");
                            maxevents = 100)
println("Loaded $(length(events)) source events")

"""
    write_sample(path, events, nevents, seed) -> Int

Write the resampled sample and return the particle count. This is a function
rather than top-level code so the row counter is an ordinary local: assigning
to a global from inside the `open(...) do` closure hits Julia's soft-scope
rules and fails.
"""
function write_sample(path, events, nevents, seed)
    rng = MersenneTwister(seed)
    mkpath(dirname(path))
    total_rows = 0
    open(path, "w") do io
        println(io, "event_id,weight,y,phi")
        for i in 1:nevents
            src = events[rand(rng, 1:length(events))]
            dphi = 2π * rand(rng)
            for r in 1:size(src, 1)
                w, y, phi = src[r, 1], src[r, 2], src[r, 3]
                # Wrap into (-pi, pi] so both readers see the same convention.
                rotated = mod(phi + dphi + π, 2π) - π
                @printf(io, "%d,%.17g,%.17g,%.17g\n", i, w, y, rotated)
                total_rows += 1
            end
        end
    end
    return total_rows
end

total_rows = write_sample(OUTFILE, events, NEVENTS, SEED)

npairs = NEVENTS * (NEVENTS - 1) ÷ 2
@printf("Wrote %d events (%d particles, %d unique pairs) to %s\n",
        NEVENTS, total_rows, npairs, OUTFILE)
