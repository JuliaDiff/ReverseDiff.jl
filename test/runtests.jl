testprintln(f) = println("  testing `$(f)`...")

println("running ArrayTests...")
include(joinpath(dirname(@__FILE__), "ArrayTests.jl"))
println("running MiscTests...")
include(joinpath(dirname(@__FILE__), "MiscTests.jl"))
println("done")
