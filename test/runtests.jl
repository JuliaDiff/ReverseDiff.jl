const TESTDIR = dirname(@__FILE__)

testprintln(kind, f, pad = "  ") = println(pad, "testing $(kind): `$(f)`...")

include(joinpath(TESTDIR, "GradientTests.jl"))
include(joinpath(TESTDIR, "JacobianTests.jl"))
include(joinpath(TESTDIR, "HessianTests.jl"))
