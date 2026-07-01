using Documenter
using EnergyFlow

DocMeta.setdocmeta!(EnergyFlow, :DocTestSetup, :(using EnergyFlow); recursive=true)

makedocs(
    modules=[EnergyFlow],
    sitename="EnergyFlow.jl",
    checkdocs=:none,
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://leblanc-lab.github.io/EnergyFlow.jl/stable",
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Getting Started" => "man/getting_started.md",
            "Backends and Metrics" => "man/backends_and_metrics.md",
        ],
        "API" => "api.md",
    ],
)

deploydocs(
    repo="github.com/leblanc-lab/EnergyFlow.jl.git",
    devbranch="main",
)