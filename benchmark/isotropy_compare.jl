# Event Isotropy — side-by-side Julia vs POT comparison
# Usage (after running both benchmarks):
#   cd benchmark
#   julia --project=. isotropy_benchmark.jl          # writes result/isotropy_julia.md
#   python isotropy_benchmark_python.py              # writes result/isotropy_python.md
#   julia --project=. isotropy_compare.jl            # writes result/isotropy_compare.md
#
# Merges the two result tables into one, pairing Julia's default exact backend
# (:ns64) against POT per setup, with speedup and mean-isotropy difference.

using Printf

const RESULT = joinpath(@__DIR__, "result")

# Parse a result markdown table into rows keyed by (setup, backend).
function parse_results(path)
    rows = Dict{Tuple{String,String},NamedTuple}()
    for line in eachline(path)
        startswith(line, "|") || continue
        cells = strip.(split(line, "|"))
        cells = cells[2:end-1]                       # drop leading/trailing empties
        length(cells) == 5 || continue
        cells[1] in ("Setup", "---", ":---") && continue
        occursin("-", cells[1]) && all(c -> c in "-: ", cells[1]) && continue
        setup, backend = cells[1], cells[2]
        t = tryparse(Float64, cells[3]); t === nothing && continue
        mean_iso = tryparse(Float64, cells[5]); mean_iso === nothing && continue
        rows[(setup, backend)] = (time_s=t, mean=mean_iso)
    end
    return rows
end

jl_path = joinpath(RESULT, "isotropy_julia.md")
py_path = joinpath(RESULT, "isotropy_python.md")
for p in (jl_path, py_path)
    isfile(p) || error("Missing $p — run both isotropy benchmarks first (see isotropy_compare.jl header).")
end

jl = parse_results(jl_path)
py = parse_results(py_path)

# Setup order, preserved from the Julia file.
setups = String[]
for (setup, backend) in keys(jl)
    backend == "ns64" && push!(setups, setup)
end
# stable order: ring128, ring2, cyl16 if present, else sorted
const ORDER = ["ring128", "ring2", "cyl16", "sphere192", "sphere48"]
sort!(setups, by = s -> (i = findfirst(==(s), ORDER); i === nothing ? length(ORDER) + 1 : i))

println("Event Isotropy — Julia (:ns64) vs POT\n")
@printf("%-8s %10s %10s %8s %12s %12s %10s\n",
        "Setup", "Jl (s)", "POT (s)", "Speedup", "Jl mean", "POT mean", "|Δ mean|")
rows = String[]
for s in setups
    j = get(jl, (s, "ns64"), nothing)
    p = get(py, (s, "POT"), nothing)
    (j === nothing || p === nothing) && continue
    speedup = p.time_s / j.time_s
    dmean = abs(j.mean - p.mean)
    @printf("%-8s %10.3f %10.3f %7.2f× %12.6f %12.6f %10.2e\n",
            s, j.time_s, p.time_s, speedup, j.mean, p.mean, dmean)
    push!(rows, @sprintf("| %s | %.3f | %.3f | %.2f× | %.6f | %.6f | %.2e |",
                         s, j.time_s, p.time_s, speedup, j.mean, p.mean, dmean))
end

mkpath(RESULT)
open(joinpath(RESULT, "isotropy_compare.md"), "w") do io
    println(io, "# Event Isotropy — Julia (:ns64) vs POT\n")
    println(io, "Same reference events, ground distances, and event selection on both sides;")
    println(io, "Julia uses the default exact network-simplex backend, POT uses `ot.lp.emd2`.\n")
    println(io, "| Setup | Julia (s) | POT (s) | Speedup | Julia mean | POT mean | \\|Δ mean\\| |")
    println(io, "|---|---|---|---|---|---|---|")
    foreach(r -> println(io, r), rows)
end
println("\nResults saved to result/isotropy_compare.md")
