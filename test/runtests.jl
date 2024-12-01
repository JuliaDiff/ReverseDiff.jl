using Test

const TESTDIR = dirname(@__FILE__)

test_println(kind, f, pad="  ") = println(pad, "testing $(kind): `$(f)`...")

@testset "ReverseDiff" begin

    @testset "TapeTests" begin
        println("running TapeTests...")
        t = @elapsed include(joinpath(TESTDIR, "TapeTests.jl"))
        println("done (took $t seconds).")
    end

    @testset "TrackedTests" begin
        println("running TrackedTests...")
        t = @elapsed include(joinpath(TESTDIR, "TrackedTests.jl"))
        println("done (took $t seconds).")
    end

    @testset "MacrosTests" begin
        println("running MacrosTests...")
        t = @elapsed include(joinpath(TESTDIR, "MacrosTests.jl"))
        println("done (took $t seconds).")
    end

    @testset "ChainRulesTests" begin
        println("running ChainRulesTests...")
        t = @elapsed include(joinpath(TESTDIR, "ChainRulesTests.jl"))
        println("done (took $t seconds).")
    end

    @testset "ScalarTests" begin
        println("running ScalarTests...")
        t = @elapsed include(joinpath(TESTDIR, "derivatives/ScalarTests.jl"))
        println("done (took $t seconds).")
    end

    @testset "LinAlgTests" begin
        println("running LinAlgTests...")
        t = @elapsed include(joinpath(TESTDIR, "derivatives/LinAlgTests.jl"))
        println("done (took $t seconds).")
    end

    @testset "ElementWiseTests" begin
        println("running ElementwiseTests...")
        t = @elapsed include(joinpath(TESTDIR, "derivatives/ElementwiseTests.jl"))
        println("done (took $t seconds).")
    end

    @testset "ArrayFunctionTests" begin
        println("running ArrayFunctionTests...")
        t = @elapsed include(joinpath(TESTDIR, "derivatives/ArrayFunctionTests.jl"))
        println("done (took $t seconds).")
    end

    @testset "GradientTests" begin
        println("running GradientTests...")
        t = @elapsed include(joinpath(TESTDIR, "api/GradientTests.jl"))
        println("done (took $t seconds).")
    end

    @testset "JacobianTests" begin
        println("running JacobianTests...")
        t = @elapsed include(joinpath(TESTDIR, "api/JacobianTests.jl"))
        println("done (took $t seconds).")
    end

    @testset "HessianTests" begin
        println("running HessianTests...")
        t = @elapsed include(joinpath(TESTDIR, "api/HessianTests.jl"))
        println("done (took $t seconds).")
    end

    @testset "ConfigTests" begin
        println("running ConfigTests...")
        t = @elapsed include(joinpath(TESTDIR, "api/ConfigTests.jl"))
        println("done (took $t seconds).")
    end

    @testset "CompatTests" begin
        println("running CompatTests...")
        t = @elapsed include(joinpath(TESTDIR, "compat/CompatTests.jl"))
        println("done (took $t seconds).")
    end
end
