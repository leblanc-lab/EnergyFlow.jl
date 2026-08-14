# Harness for #2: backend dispatch — type-stability + contagion into caller code.
#   julia --project=. demo_backend_dispatch.jl
#
# ONE line to edit as we go — `ns64_handle` below:
#   * BEFORE the refactor:                 const ns64_handle = :ns64
#   * AFTER you define NS64 <: EMDBackend: const ns64_handle = NS64()

using EnergyFlow
using Test
using Random
Random.seed!(1)

# >>> EDIT THIS LINE (see header) <<<
# const ns64_handle = :ns64
const ns64_handle = EnergyFlow.NS64()

ev0 = rand(6, 3); ev1 = rand(6, 3)
events = [rand(6, 3) for _ in 1:6]

infer1(f, types) = first(Base.return_types(f, types))

println("explicit backend handle = ", ns64_handle, " :: ", typeof(ns64_handle))

# ---- 1. Inferred return types (want: concrete, not Union / bare Array) ----
call_default(a, b)  = emd(a, b)
call_explicit(a, b) = emd(a, b; backend = ns64_handle)
println("\nINFERRED RETURN TYPES")
println("  emd(a, b)               -> ", infer1(call_default,  (Matrix{Float64}, Matrix{Float64})))
println("  emd(a, b; backend=<h>)  -> ", infer1(call_explicit, (Matrix{Float64}, Matrix{Float64})))
println("  emds(events)            -> ", infer1(emds, (Vector{Matrix{Float64}},)))

# ---- 2. THE EFFECT: does the instability leak into YOUR code? ----
user_pipeline(events) = sum(emds(events))     # a plausible bit of downstream analysis
println("\nCONTAGION (caller code built on the API)")
println("  sum(emds(events))       -> ", infer1(user_pipeline, (Vector{Matrix{Float64}},)))

# ---- 3. @inferred gates ----
emd(ev0, ev1); emd(ev0, ev1; backend = ns64_handle); emds(events)  # warm up
function gate(f)
    try
        f(); "PASS (concrete)"
    catch e
        (e isa ErrorException && occursin("does not match", e.msg)) ? "FAIL (unstable)" : rethrow()
    end
end
println("\n@inferred GATES")
println("  emd(ev0, ev1)           = ", gate(() -> @inferred emd(ev0, ev1)))
println("  emd(ev0, ev1; <h>)      = ", gate(() -> @inferred emd(ev0, ev1; backend = ns64_handle)))

# ---- 4. correctness: same value however the backend is spelled ----
println("\nCORRECTNESS")
println("  value(:ns64) == value(<h>) : ",
        emd(ev0, ev1; backend = :ns64) == emd(ev0, ev1; backend = ns64_handle))