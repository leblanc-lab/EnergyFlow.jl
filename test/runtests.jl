using EnergyFlow
using Test
using LinearAlgebra
using Random

@testset "EnergyFlow.jl Tests" begin
    include("test_emd.jl")
    include("test_utils.jl")
    include("test_distances.jl")
end