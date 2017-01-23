const TESTDIR = dirname(@__FILE__)

test_println(kind, f, pad = "  ") = println(pad, "testing $(kind): `$(f)`...")

include(joinpath(TESTDIR, "TapeTests.jl"))
include(joinpath(TESTDIR, "TrackedTests.jl"))
include(joinpath(TESTDIR, "MacrosTests.jl"))
include(joinpath(TESTDIR, "derivatives/ScalarTests.jl"))
include(joinpath(TESTDIR, "derivatives/LinAlgTests.jl"))
include(joinpath(TESTDIR, "derivatives/ElementwiseTests.jl"))
include(joinpath(TESTDIR, "api/ConfigTests.jl"))
include(joinpath(TESTDIR, "api/GradientTests.jl"))
include(joinpath(TESTDIR, "api/JacobianTests.jl"))
include(joinpath(TESTDIR, "api/HessianTests.jl"))
