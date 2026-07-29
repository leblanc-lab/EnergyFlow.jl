#=
    Precompile.jl

    PrecompileTools workload. Running representative `emd`/`emds` calls during
    package precompilation caches native code for the hot paths, so the first
    real call in a fresh session is fast (see the TTFX numbers in the 0.2 PR).

    Everything here is LOCAL to the workload: no workspace, Task, or solver is
    stored in a module global, so nothing sized by `Threads.nthreads()` at
    precompile time (when nthreads is typically 1) leaks into runtime state.
    Runtime constructs its own workspaces with the runtime thread count.
=#

using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # Tiny events in [pT, y, φ] format (M×3). e4 is deliberately imbalanced
    # (total pT ≠ that of the others) so norm=false traces the fictitious-particle
    # branches. Nothing here escapes the workload block.
    e1 = [0.6 0.1 0.2; 0.4 -0.3 0.5]
    e2 = [0.5 0.2 -0.1; 0.5 0.4 0.3]
    e3 = reshape([1.0, 0.0, 0.0], 1, 3)
    e4 = [0.3 0.5 0.5; 0.9 -0.2 0.1]      # total pT = 1.2 (imbalanced vs. 1.0)
    events = [e1, e2, e3, e4]

    e1_32 = Float32.(e1)
    e2_32 = Float32.(e2)

    all_backends   = (:ns64, :ot64, :ns32, :ot32, :sinkhorn)
    dispatch_metrics = (EuclideanMetric(), SquaredEuclideanMetric(), EtaPhiMetric())

    @compile_workload begin
        # ── Single-pair EMD: all 5 backends, Float64 pair, norm=true/false ──
        for be in all_backends, nrm in (true, false)
            emd(e1, e2; backend=be, R=1.0, beta=1.0, norm=nrm)
        end
        # Float32 pair through the Float32 backends
        for be in (:ns32, :ot32), nrm in (true, false)
            emd(e1_32, e2_32; backend=be, R=1.0, beta=1.0, norm=nrm)
        end

        # ── Each concrete GroundMetric, norm=true/false, Float64 + Float32 ──
        for m in dispatch_metrics, nrm in (true, false)
            emd(e1, e2; backend=:ns64, R=1.0, beta=1.0, norm=nrm, metric=m)
            emd(e1_32, e2_32; backend=:ns32, R=1.0, beta=1.0, norm=nrm, metric=m)
        end
        # PrecomputedMetric (single pair, 2×2 cost, via a reused workspace)
        let pc = PrecomputedMetric([0.0 1.0; 1.0 0.0]),
            ws = EMDWorkspace(2, 2; norm=true, metric=pc)
            emd!(ws, e1, e2; backend=:ns64)
        end
        # CustomMetric (closure ground distance)
        let cm = CustomMetric((a, b) -> sqrt(sum(abs2, a .- b)))
            emd(e1, e2; backend=:ns64, norm=true, metric=cm)
        end

        # ── Workspace-reuse single-pair path (Float64 + Float32) ──
        let ws64 = EMDWorkspace(2, 2; norm=true)
            emd!(ws64, e1, e2)
        end
        let ws32 = EMDWorkspace{Float32}(2, 2; norm=true)
            emd_ns32!(ws32, e1_32, e2_32)
            emd_ot32!(ws32, e1_32, e2_32)
        end

        # ── Pairwise self + cross on 4 tiny events, all backends, norm=true/false ──
        for be in all_backends, nrm in (true, false)
            emds(events; backend=be, R=1.0, beta=1.0, norm=nrm)                 # self
            emds(events[1:2], events[3:4]; backend=be, R=1.0, beta=1.0, norm=nrm)  # cross
        end
        # In-place pairwise (exact backends only; :sinkhorn is unsupported)
        let r = Vector{Float64}(undef, length(events) * (length(events) - 1) ÷ 2)
            emds!(r, events; backend=:ns64, norm=true)
        end
    end
end
