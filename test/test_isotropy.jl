using Test
using EnergyFlow

@testset "Event isotropy" begin
    @testset "reference geometries" begin
        ring = ring_reference(4)
        @test size(ring) == (4, 2)
        @test sum(ring[:, 1]) ≈ 1.0
        @test ring[:, 2] ≈ [π / 4, 3π / 4, 5π / 4, 7π / 4]

        cyl = cylinder_reference(4, 2.0)
        @test size(cyl, 2) == 3
        @test sum(cyl[:, 1]) ≈ 1.0

        sph = sphere_reference(1)
        @test size(sph) == (48, 4)
        @test sum(sph[:, 1]) ≈ 1.0

        vecs = healpix_pix2vec_ring(1)
        @test length(vecs) == 12
        @test all(isapprox(sum(abs2, v), 1.0; atol=1e-12) for v in vecs)
    end

    @testset "ground metrics" begin
        @test emd([1.0 0.0], [1.0 π]; R=1.0, beta=1.0, norm=true, metric=ring_phi_metric()) ≈ 4.0 atol=1e-10
        @test emd([1.0 0.0 0.0], [1.0 1.0 0.0]; R=1.0, beta=1.0, norm=true, metric=cylinder_metric(1.0)) ≈ (12 / (π^2 + 16)) atol=1e-10
        @test emd([1.0 1.0 0.0 0.0], [1.0 0.0 1.0 0.0]; R=1.0, beta=1.0, norm=true, metric=sphere_cos_metric()) ≈ 2.0 atol=1e-10
        @test emd([1.0 1.0 0.0 0.0], [1.0 0.0 1.0 0.0]; R=1.0, beta=1.0, norm=true, metric=sphere_angular_metric()) ≈ π / 2 atol=1e-10
    end

    @testset "selection helpers" begin
        events = [
            [1.0 0.1 0.0; 1.0 -0.2 0.1; 1.0 2.5 -0.3],
            [1.0 3.0 0.0],
        ]
        selected = select_events(events, 1.0)
        @test length(selected) == 1
        @test size(selected[1], 1) == 2

        sphere_events = [
            [1.0 1.0 0.0 0.0; 2.0 0.0 0.0 0.0; 1.0 0.0 1.0 0.0],
            [1.0 0.0 0.0 0.0],
        ]
        sphere_selected = select_sphere_events(sphere_events)
        @test length(sphere_selected) == 1
        @test size(sphere_selected[1], 1) == 2
    end

    @testset "front door" begin
        ring = ring_reference(4)
        ring_event = hcat(ring[:, 1], zeros(4), ring[:, 2])
        @test event_isotropy(ring_event; geometry=:ring, n=4) ≈ 0.0 atol=1e-10

        cyl = cylinder_reference(4, 2.0)
        @test event_isotropy(cyl; geometry=:cylinder, nphi=4, ymax=2.0) ≈ 0.0 atol=1e-10

        sph = sphere_reference(1)
        @test event_isotropy(sph; geometry=:sphere, nval=1) ≈ 0.0 atol=1e-10
    end

    @testset "hepmc3 momenta loader" begin
        mktempdir() do dir
            path = joinpath(dir, "mini.hepmc")
            open(path, "w") do io
                println(io, "HepMC::Version 3.0")
                println(io, "E 1 0 0 0 0 0 0 0 0 0")
                println(io, "P 1 0 11 1.0 0.0 0.0 1.0 0.0 1")
                println(io, "P 2 0 13 0.0 1.0 1.0 1.5 0.0 2")
                println(io, "E 2 0 0 0 0 0 0 0 0 0")
                println(io, "P 3 0 22 0.0 2.0 0.0 2.0 0.0 1")
                println(io, "HepMC::END_EVENT_LISTING")
            end

            events = load_hepmc3_momenta(path)
            @test length(events) == 2
            @test size(events[1]) == (1, 4)
            @test size(events[2]) == (1, 4)
            @test events[1][1, 1] ≈ 1.0 atol=1e-10
            @test events[1][1, 2] ≈ 1.0 atol=1e-10
            @test events[1][1, 4] ≈ 0.0 atol=1e-10
        end
    end
end