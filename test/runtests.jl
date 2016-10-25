const TESTDIR = dirname(@__FILE__)

testprintln(kind, f, pad = "  ") = println(pad, "testing $(kind): `$(f)`...")

include(joinpath(TESTDIR, "TapeTests.jl"))
include(joinpath(TESTDIR, "TrackedTests.jl"))
include(joinpath(TESTDIR, "UtilsTests.jl"))
include(joinpath(TESTDIR, "api/GradientTests.jl"))
include(joinpath(TESTDIR, "api/JacobianTests.jl"))
include(joinpath(TESTDIR, "api/HessianTests.jl"))
