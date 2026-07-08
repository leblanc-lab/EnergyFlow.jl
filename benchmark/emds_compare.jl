# Pairwise EMD — side-by-side Julia vs POT comparison
# Usage (after running both benchmarks):
#   cd benchmark
#   julia --project=. emds_benchmark.jl              # writes result/emds_julia.md
#   python emds_benchmark_python.py                  # writes result/emds_python.md
#   julia --project=. emds_compare.jl                # writes result/emds_compare.md
#
# Merges the two result tables into one, pairing Julia's default exact backend
# (:ns64) against POT (ot.lp.emd2) per (split, setup), with speedup. Both sides
# solve the identical pairwise EMD problems (same ground distances, R/beta/norm
# handling), so this is a pure timing comparison — the numeric agreement is
# verified separately.

using Printf

const RESULT = joinpath(@__DIR__, "result")

# Parse an emds result markdown table into rows keyed by (split, setup, backend).
function parse_results(path)
    rows = Dict{Tuple{String,String,String},NamedTuple}()
    for line in eachline(path)
        startswith(line, "|") || continue
        cells = strip.(split(line, "|"))
        cells = cells[2:end-1]                       # drop leading/trailing empties
        length(cells) == 5 || continue
        cells[1] == "Split" && continue
        all(c -> c in "-: ", cells[1]) && continue   # separator row
        split_name, setup, backend = cells[1], cells[2], cells[3]
        t = tryparse(Float64, cells[4]); t === nothing && continue
        pairs = tryparse(Int, cells[5]); pairs === nothing && continue
        rows[(split_name, setup, backend)] = (time_s=t, pairs=pairs)
    end
    return rows
end

jl_path = joinpath(RESULT, "emds_julia.md")
py_path = joinpath(RESULT, "emds_python.md")
for p in (jl_path, py_path)
    isfile(p) || error("Missing $p — run both pairwise EMD benchmarks first (see emds_compare.jl header).")
end

jl = parse_results(jl_path)
py = parse_results(py_path)

# (split, setup) pairs present on both sides. Both exact Float64 backends are
# reported: :ns64 (plain network simplex) and :ot64 (arc mixing). POT's emd2 is
# itself an arc-mixing network simplex, so :ot64 is its closest analog — best
# for the unbalanced (norm=false) problems, while :ns64 tends to win the
# balanced ones.
keys_both = Tuple{String,String}[]
for (split_name, setup, backend) in keys(jl)
    backend == "ns64" || continue
    haskey(py, (split_name, setup, "POT")) &&
        haskey(jl, (split_name, setup, "ot64")) &&
        push!(keys_both, (split_name, setup))
end
const SPLIT_ORDER = ["10v90", "50v50"]
const SETUP_ORDER = ["Euclidean_norm", "Euclidean_unnorm", "EtaPhi_norm"]
sort!(keys_both, by = k -> (
    (i = findfirst(==(k[1]), SPLIT_ORDER); i === nothing ? length(SPLIT_ORDER) + 1 : i),
    (j = findfirst(==(k[2]), SETUP_ORDER); j === nothing ? length(SETUP_ORDER) + 1 : j),
))

println("Pairwise EMD — Julia (:ns64, :ot64) vs POT  (speedup = POT / Julia, >1 ⇒ Julia faster)\n")
@printf("%-6s %-18s %9s %9s %9s %8s %8s %8s\n",
        "Split", "Setup", "ns64 (s)", "ot64 (s)", "POT (s)", "ns64×", "ot64×", "Pairs")
rows = String[]
for (split_name, setup) in keys_both
    ns = jl[(split_name, setup, "ns64")]
    ot = jl[(split_name, setup, "ot64")]
    p  = py[(split_name, setup, "POT")]
    su_ns = p.time_s / ns.time_s
    su_ot = p.time_s / ot.time_s
    @printf("%-6s %-18s %9.2f %9.2f %9.2f %7.2f× %7.2f× %8d\n",
            split_name, setup, ns.time_s, ot.time_s, p.time_s, su_ns, su_ot, p.pairs)
    push!(rows, @sprintf("| %s | %s | %.2f | %.2f | %.2f | %.2f× | %.2f× | %d |",
                         split_name, setup, ns.time_s, ot.time_s, p.time_s, su_ns, su_ot, p.pairs))
end

mkpath(RESULT)
open(joinpath(RESULT, "emds_compare.md"), "w") do io
    println(io, "# Pairwise EMD — Julia (:ns64, :ot64) vs POT\n")
    println(io, "Identical pairwise EMD problems on both sides (same ground distances,")
    println(io, "R/beta/norm handling). Both exact Float64 backends are shown: `:ns64`")
    println(io, "(plain network simplex, best for balanced/normalized) and `:ot64` (arc")
    println(io, "mixing, best for unbalanced). POT (`ot.lp.emd2`) is itself an arc-mixing")
    println(io, "network simplex, so `:ot64` is its closest analog. Speedup = POT / Julia")
    println(io, "(> 1 ⇒ Julia faster).\n")
    println(io, "> Compare at matched thread counts: POT is single-threaded, so run the")
    println(io, "> Julia benchmark single-threaded (no `-t`) for a like-for-like table.\n")
    println(io, "| Split | Setup | ns64 (s) | ot64 (s) | POT (s) | ns64× | ot64× | Pairs |")
    println(io, "|---|---|---|---|---|---|---|---|")
    foreach(r -> println(io, r), rows)
end
println("\nResults saved to result/emds_compare.md")
