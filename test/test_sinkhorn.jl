using Test
using Random

# Load EnergyFlow
pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))
using EnergyFlow

Random.seed!(42)

# Helper: generate random event with n particles in d dimensions
function random_event(n::Int, d::Int=2; weight_scale=1.0)
    ev = Matrix{Float64}(undef, n, 1 + d)
    for i in 1:n
        ev[i, 1] = rand() * weight_scale  # weight
        for k in 1:d
            ev[i, k+1] = randn()           # coordinates
        end
    end
    return ev
end

test_log("="^70)
test_log("SINKHORN BACKEND TESTS")
test_log("="^70)

# ─────────────────────────────────────────────────────────────────────
# Test 1: Basic functionality — does it run without error?
# ─────────────────────────────────────────────────────────────────────
@testset "Sinkhorn basic functionality" begin
    ev0 = [1.0 0.0 0.0; 1.0 1.0 0.0]
    ev1 = [1.0 0.5 0.0; 1.0 0.5 1.0]

    val = emd_sinkhorn(ev0, ev1; R=1.0, beta=1.0, norm=true)
    @test isfinite(val)
    @test val >= 0.0
    test_log("  Basic test: emd_sinkhorn = $val")

    # Via backend dispatch
    val2 = emd(ev0, ev1; backend=:sinkhorn, R=1.0, beta=1.0, norm=true)
    @test val2 ≈ val
    test_log("  Backend dispatch: emd(:sinkhorn) = $val2")
end

# ─────────────────────────────────────────────────────────────────────
# Test 2: Workspace reuse
# ─────────────────────────────────────────────────────────────────────
@testset "Sinkhorn workspace reuse" begin
    ev0 = random_event(5)
    ev1 = random_event(5)

    ws = SinkhornWorkspace{Float64}(10, 10; beta=1.0, R=1.0, norm=true,
                                     epsilon=0.01, annealing=true)
    val1 = emd_sinkhorn!(ws, ev0, ev1)
    val2 = emd_sinkhorn!(ws, ev0, ev1)
    @test val1 ≈ val2
    test_log("  Workspace reuse: $val1 ≈ $val2 ✓")
end

# ─────────────────────────────────────────────────────────────────────
# Test 3: Comparison with ns64 for various parameters
# ─────────────────────────────────────────────────────────────────────
test_log("\n" * "="^70)
test_log("SINKHORN vs NS64 ACCURACY")
test_log("="^70)

@testset "Sinkhorn vs ns64 accuracy" begin
    for n in [2, 10, 50, 100]
        for (R, beta, norm_flag) in [
            (1.0, 1.0, true),
            (0.5, 1.0, true),
            (2.0, 1.0, true),
            (1.0, 2.0, true),
            (1.0, 0.5, true),
            (1.0, 1.0, false),
            (0.5, 2.0, false),
        ]
            ev0 = random_event(n)
            ev1 = random_event(n)

            ns_val = emd_ns64(ev0, ev1; R=R, beta=beta, norm=norm_flag)
            sk_val = emd_sinkhorn(ev0, ev1; R=R, beta=beta, norm=norm_flag,
                                   epsilon=0.001, max_iter_sinkhorn=10000,
                                   sinkhorn_tol=1e-12, annealing=true)

            if ns_val > 1e-10
                rel_err = abs(sk_val - ns_val) / ns_val
            else
                rel_err = abs(sk_val - ns_val)
            end

            status = rel_err < 0.01 ? "✓" : (rel_err < 0.05 ? "~" : "✗")
                test_log("  n=$n R=$R β=$beta norm=$norm_flag: ns64=$(round(ns_val, digits=6)) " *
                         "sinkhorn=$(round(sk_val, digits=6)) rel_err=$(round(rel_err, digits=6)) $status")

            # Sinkhorn entropic bias grows with cost scale: R=0.5, β=2 produces
            # very large costs where ε=0.001 is still significant.
            # Allow 15% for extreme cases (small R + large β), 10% otherwise.
            threshold = (R < 1.0 && beta > 1.0) ? 0.15 : 0.10
            @test rel_err < threshold || ns_val < 1e-10
        end
    end
end

# ─────────────────────────────────────────────────────────────────────
# Test 4: Epsilon sensitivity study
# ─────────────────────────────────────────────────────────────────────
test_log("\n" * "="^70)
test_log("EPSILON SENSITIVITY")
test_log("="^70)

@testset "Epsilon sensitivity" begin
    ev0 = random_event(20)
    ev1 = random_event(20)
    ns_val = emd_ns64(ev0, ev1; R=1.0, beta=1.0, norm=true)

    test_log("  ns64 reference: $ns_val")
    test_log("  ε         sinkhorn     rel_err    iters  converged")
    test_log("  " * "-"^60)

    for eps_val in [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        ws = SinkhornWorkspace{Float64}(20, 20; beta=1.0, R=1.0, norm=true,
                                         epsilon=eps_val, max_iter=10000,
                                         tol=1e-12, annealing=true)
        sk_val = emd_sinkhorn!(ws, ev0, ev1)
        rel_err = ns_val > 1e-10 ? abs(sk_val - ns_val) / ns_val : abs(sk_val - ns_val)
        test_log("  $(lpad(eps_val, 8))  $(lpad(round(sk_val, digits=8), 12))  " *
             "$(lpad(round(rel_err, digits=8), 10))  $(lpad(ws.n_iters, 5))  $(ws.converged)")
    end
end

# ─────────────────────────────────────────────────────────────────────
# Test 5: Annealing vs no-annealing
# ─────────────────────────────────────────────────────────────────────
test_log("\n" * "="^70)
test_log("ANNEALING vs NO-ANNEALING")
test_log("="^70)

@testset "Annealing comparison" begin
    ev0 = random_event(30)
    ev1 = random_event(30)
    ns_val = emd_ns64(ev0, ev1; R=1.0, beta=1.0, norm=true)

    for eps_val in [0.01, 0.001]
        sk_anneal = emd_sinkhorn(ev0, ev1; R=1.0, beta=1.0, norm=true,
                                  epsilon=eps_val, annealing=true, max_iter_sinkhorn=10000)
        sk_plain = emd_sinkhorn(ev0, ev1; R=1.0, beta=1.0, norm=true,
                                 epsilon=eps_val, annealing=false, max_iter_sinkhorn=10000)

        err_anneal = abs(sk_anneal - ns_val) / max(ns_val, 1e-10)
        err_plain = abs(sk_plain - ns_val) / max(ns_val, 1e-10)

        test_log("  ε=$eps_val: anneal=$(round(sk_anneal, digits=6)) (err=$(round(err_anneal, digits=6))) " *
             "plain=$(round(sk_plain, digits=6)) (err=$(round(err_plain, digits=6)))")
    end
end

# ─────────────────────────────────────────────────────────────────────
# Test 6: Pairwise EMDs
# ─────────────────────────────────────────────────────────────────────
@testset "Sinkhorn pairwise" begin
    events = [random_event(10) for _ in 1:5]

    ns_results = emds(events; backend=:ns64, R=1.0, beta=1.0, norm=true)
    sk_results = emds_sinkhorn(events; R=1.0, beta=1.0, norm=true,
                                epsilon=0.001, annealing=true)

    max_rel_err = 0.0
    for i in eachindex(ns_results)
        if ns_results[i] > 1e-10
            err = abs(sk_results[i] - ns_results[i]) / ns_results[i]
            max_rel_err = max(max_rel_err, err)
        end
    end
    test_log("\n  Pairwise (5 events): max relative error = $(round(max_rel_err, digits=6))")
    @test max_rel_err < 0.10
end

test_log("\nAll Sinkhorn tests completed!")
