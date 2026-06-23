using Test
using EnergyFlow
include("test_helpers.jl")

test_log("="^70)
test_log("HEPMC3 TESTS")
test_log("="^70)

const _HEPMC3_CONTENT = """
HepMC::Version 3.03.00
HepMC::Asciiv3-START_EVENT_LISTING
E 0 2 3
U GEV MM
P 1 0 211  1.0  0.5  2.0  2.291  0.14 1
P 2 0 -211 -1.0 -0.5 -2.0  2.291  0.14 1
P 3 0 22   0.1  0.2  0.3  0.374  0.0  2
E 1 2 3
U GEV MM
P 4 0 2212  0.0  3.0  4.0  5.0  0.938  1
HepMC::Asciiv3-END_EVENT_LISTING
"""

@testset "HepMC3.read_events - basic" begin
    test_log("  read_events basic: reading 2-event synthetic stream")
    events = Vector{Vector{EnergyFlow.HepMC3.Particle{Float64}}}()
    EnergyFlow.HepMC3.read_events(IOBuffer(_HEPMC3_CONTENT)) do particles
        push!(events, copy(particles))
    end

    test_log("  got $(length(events)) events")
    @test length(events) == 2

    ev0 = events[1]
    test_log("  event 0: $(length(ev0)) particles")
    @test length(ev0) == 3

    # pion+, check all Particle fields
    p1 = ev0[1]
    test_log("  event 0 p1: pdgid=$(p1.pdgid) status=$(p1.status) barcode=$(p1.barcode) vertex=$(p1.vertex)")
    @test p1.pdgid   == 211
    @test p1.status  == 1
    @test p1.barcode == 1
    @test p1.vertex  == 0

    # photon is non-final-state (status=2)
    p3 = ev0[3]
    test_log("  event 0 p3: pdgid=$(p3.pdgid) status=$(p3.status)")
    @test p3.pdgid  == 22
    @test p3.status == 2

    ev1 = events[2]
    test_log("  event 1: $(length(ev1)) particles")
    @test length(ev1) == 1
    @test ev1[1].pdgid  == 2212
    @test ev1[1].status == 1
end

@testset "HepMC3.read_events - maxevents" begin
    test_log("  read_events maxevents=1")
    events = Vector{Vector{EnergyFlow.HepMC3.Particle{Float64}}}()
    EnergyFlow.HepMC3.read_events(IOBuffer(_HEPMC3_CONTENT); maxevents=1) do particles
        push!(events, copy(particles))
    end

    test_log("  maxevents=1: got $(length(events)) event(s)")
    @test length(events) == 1
    @test length(events[1]) == 3   # event 0 has 3 particles
end

@testset "HepMC3.read_events - skipevents" begin
    test_log("  read_events skipevents=1")
    events = Vector{Vector{EnergyFlow.HepMC3.Particle{Float64}}}()
    EnergyFlow.HepMC3.read_events(IOBuffer(_HEPMC3_CONTENT); skipevents=1) do particles
        push!(events, copy(particles))
    end

    test_log("  skipevents=1: got $(length(events)) event(s)")
    @test length(events) == 1
    @test events[1][1].pdgid == 2212  # only event 1's proton
end

@testset "HepMC3.read_events - END_EVENT_LISTING sentinel" begin
    content_with_extra = _HEPMC3_CONTENT * "E 99 0 0\nP 5 0 999 0.0 0.0 0.0 1.0 0.0 1\n"
    events = Vector{Vector{EnergyFlow.HepMC3.Particle{Float64}}}()
    EnergyFlow.HepMC3.read_events(IOBuffer(content_with_extra)) do particles
        push!(events, copy(particles))
    end

    test_log("  sentinel stop: got $(length(events)) event(s)")
    @test length(events) == 2
end

@testset "load_hepmc3_events - sample file" begin
    fpath = joinpath(@__DIR__, "..", "data", "sk_example_PU.hepmc")
    events = load_hepmc3_events(fpath)

    test_log("  sample file: $(length(events)) event(s) loaded")
    @test length(events) >= 1

    for ev in events
        @test size(ev, 2) == 3          # [pT, eta, phi]
        @test all(ev[:, 1] .> 0.0)      # beam remnants with pT=0 are always dropped
    end

    ev1 = events[1]
    test_log("  event 1: $(size(ev1, 1)) particles, " *
             "pT in [$(round(minimum(ev1[:,1]), digits=4)), $(round(maximum(ev1[:,1]), digits=4))]")
end

@testset "load_hepmc3_events - maxevents" begin
    fpath   = joinpath(@__DIR__, "..", "data", "sk_example_PU.hepmc")
    all_evs = load_hepmc3_events(fpath)
    one_ev  = load_hepmc3_events(fpath; maxevents=1)

    test_log("  maxevents=1: $(length(one_ev)) of $(length(all_evs)) total")
    @test length(one_ev) == min(1, length(all_evs))
end

@testset "load_hepmc3_events - skipevents" begin
    fpath   = joinpath(@__DIR__, "..", "data", "sk_example_PU.hepmc")
    all_evs = load_hepmc3_events(fpath)
    if length(all_evs) >= 2
        skipped = load_hepmc3_events(fpath; skipevents=1)
        test_log("  skipevents=1: $(length(skipped)) vs $(length(all_evs)) total")
        @test length(skipped) == length(all_evs) - 1
    end
end

@testset "load_hepmc3_events - min_pt filter" begin
    fpath      = joinpath(@__DIR__, "..", "data", "sk_example_PU.hepmc")
    evs_nopt   = load_hepmc3_events(fpath; maxevents=1, min_pt=0.0)
    evs_highpt = load_hepmc3_events(fpath; maxevents=1, min_pt=100.0)

    test_log("  min_pt=0:   $(size(evs_nopt[1], 1)) particles")
    test_log("  min_pt=100: $(size(evs_highpt[1], 1)) particles")
    @test size(evs_nopt[1], 1) >= size(evs_highpt[1], 1)
    if size(evs_highpt[1], 1) > 0
        @test all(evs_highpt[1][:, 1] .> 100.0)
    end
end

test_log("\nAll HepMC3 tests completed!")
