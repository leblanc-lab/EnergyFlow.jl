# Event Isotropy — side-by-side Julia vs POT comparison
# Usage (after running the benchmarks):
#   cd benchmark
#   julia --project=. isotropy_benchmark.jl          # writes result/isotropy_julia_t1.md
#   julia --project=. -t 8 isotropy_benchmark.jl     # writes result/isotropy_julia_t8.md
#   python isotropy_benchmark_python.py              # writes result/isotropy_python.md
#   julia --project=. isotropy_compare.jl            # writes result/isotropy_compare.md
#
# Merges the tables into one, pairing Julia's exact backends (:ns64, :ot64)
# against POT per setup, with speedup and mean-isotropy difference. Julia is
# reported at every thread count found (isotropy_julia_t{N}.md) — POT is always
# single-threaded, so the 1-thread rows are the like-for-like solver comparison
# and the multi-thread rows show the realistic threaded workload.

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

# Locate thread-tagged Julia result files: isotropy_julia_t{N}.md -> N.
jl_files = Tuple{Int,String}[]
for f in readdir(RESULT)
    m = match(r"^isotropy_julia_t(\d+)\.md$", f)
    m === nothing && continue
    push!(jl_files, (parse(Int, m.captures[1]), joinpath(RESULT, f)))
end
sort!(jl_files, by = first)

py_path = joinpath(RESULT, "isotropy_python.md")
isempty(jl_files) &&
    error("No isotropy_julia_t*.md in $RESULT — run isotropy_benchmark.jl first (see header).")
isfile(py_path) ||
    error("Missing $py_path — run isotropy_benchmark_python.py first (see header).")

jl = Dict(n => parse_results(path) for (n, path) in jl_files)   # thread count -> rows
py = parse_results(py_path)
threads = first.(jl_files)

# Setup order, from the highest-thread Julia file (all files share setups).
setups = String[]
for (setup, backend) in keys(jl[threads[end]])
    backend == "ns64" && push!(setups, setup)
end
const ORDER = ["ring128", "ring2", "cyl16", "sphere192", "sphere48"]
sort!(setups, by = s -> (i = findfirst(==(s), ORDER); i === nothing ? length(ORDER) + 1 : i))

println("Event Isotropy — Julia (:ns64, :ot64) vs POT  (speedup = POT / Julia, >1 ⇒ Julia faster)")
println("POT is single-threaded; Julia shown at threads = $(join(threads, ", "))\n")
@printf("%-8s %4s %9s %9s %9s %8s %8s %9s\n",
        "Setup", "thr", "ns64 (s)", "ot64 (s)", "POT (s)", "ns64×", "ot64×", "|Δ mean|")
rows = String[]
for s in setups
    p = get(py, (s, "POT"), nothing)
    p === nothing && continue
    for n in threads
        ns = get(jl[n], (s, "ns64"), nothing)
        ot = get(jl[n], (s, "ot64"), nothing)
        (ns === nothing || ot === nothing) && continue
        su_ns = p.time_s / ns.time_s
        su_ot = p.time_s / ot.time_s
        dmean = abs(ns.mean - p.mean)
        @printf("%-8s %4d %9.3f %9.3f %9.3f %7.2f× %7.2f× %9.2e\n",
                s, n, ns.time_s, ot.time_s, p.time_s, su_ns, su_ot, dmean)
        push!(rows, @sprintf("| %s | %d | %.3f | %.3f | %.3f | %.2f× | %.2f× | %.2e |",
                             s, n, ns.time_s, ot.time_s, p.time_s, su_ns, su_ot, dmean))
    end
end

mkpath(RESULT)
open(joinpath(RESULT, "isotropy_compare.md"), "w") do io
    println(io, "# Event Isotropy — Julia (:ns64, :ot64) vs POT\n")
    println(io, "Same reference events, ground distances, and event selection on both sides.")
    println(io, "Both exact Float64 backends are shown: `:ns64` (plain network simplex) and")
    println(io, "`:ot64` (arc mixing); POT (`ot.lp.emd2`) is an arc-mixing network simplex,")
    println(io, "so `:ot64` is its closest analog. Speedup = POT / Julia (> 1 ⇒ Julia faster).")
    println(io, "Mean isotropy is backend-independent, so one `|Δ mean|` column (:ns64 vs POT)")
    println(io, "documents the numeric agreement.\n")
    println(io, "POT is single-threaded, so the **thr = 1** rows are the like-for-like solver")
    println(io, "comparison; higher-thread rows show Julia's realistic threaded workload")
    println(io, "(`emds` parallelizes across events).\n")
    println(io, "| Setup | Julia threads | ns64 (s) | ot64 (s) | POT (s) | ns64× | ot64× | \\|Δ mean\\| |")
    println(io, "|---|---|---|---|---|---|---|---|")
    foreach(r -> println(io, r), rows)
end
println("\nResults saved to result/isotropy_compare.md")
