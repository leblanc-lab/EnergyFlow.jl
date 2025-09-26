# Exact EMD solver using JuMP and HiGHS

using JuMP
using HiGHS

"""
    solve_emd_exact(a::AbstractVector{Float64}, b::AbstractVector{Float64}, C::AbstractMatrix{Float64})

Solve EMD using exact linear programming with JuMP and HiGHS.

# Arguments
- `a`: Source weights (supply)
- `b`: Target weights (demand)
- `C`: Cost matrix

# Returns
- `G`: Optimal transport plan (flow matrix)
- `cost`: Total transport cost
"""
function solve_emd_exact(a::AbstractVector{Float64}, b::AbstractVector{Float64}, C::AbstractMatrix{Float64})
    n, m = length(a), length(b)
    
    # Create JuMP model with HiGHS optimizer
    model = Model(HiGHS.Optimizer)
    set_silent(model)
    
    # Decision variables: flow matrix G[i,j] >= 0
    @variable(model, G[1:n, 1:m] >= 0)
    
    # Objective: minimize sum of costs
    @objective(model, Min, sum(G[i,j] * C[i,j] for i in 1:n, j in 1:m))
    
    # Constraints: supply constraints (sum over j)
    for i in 1:n
        @constraint(model, sum(G[i,j] for j in 1:m) == a[i])
    end
    
    # Constraints: demand constraints (sum over i)
    for j in 1:m
        @constraint(model, sum(G[i,j] for i in 1:n) == b[j])
    end
    
    # Solve the linear program
    optimize!(model)
    
    # Check solver status
    status = termination_status(model)
    if status != OPTIMAL
        if status == INFEASIBLE
            error("EMD problem is infeasible. Check that sum(a) ≈ sum(b)")
        else
            @warn "LP solver status: $status"
        end
    end
    
    # Extract solution
    G_val = value.(G)
    cost_val = objective_value(model)
    
    return G_val, cost_val
end