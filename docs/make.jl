using Documenter, ReverseDiff

makedocs(;
    modules=[ReverseDiff],
    sitename="ReverseDiff.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
    ),
    pages=[
        "Home" => "index.md",
        "Limitation of ReverseDiff" => "limits.md",
        "API" => "api.md",
    ],
    checkdocs=:exports,
)

deploydocs(;
    repo="github.com/JuliaDiff/ReverseDiff.jl.git", push_preview=true,
)
