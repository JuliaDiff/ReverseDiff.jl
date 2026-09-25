using Test

test_println(kind, f, pad="  ") = println(pad, "testing $(kind): `$(f)`...")

@testset "ReverseDiff" begin

    @testset "TapeTests" begin
        println("running TapeTests...")
        t = @elapsed include("TapeTests.jl")
        println("done (took $t seconds).")
    end

    @testset "TrackedTests" begin
        println("running TrackedTests...")
        t = @elapsed include("TrackedTests.jl")
        println("done (took $t seconds).")
    end

    @testset "MacrosTests" begin
        println("running MacrosTests...")
        t = @elapsed include("MacrosTests.jl")
        println("done (took $t seconds).")
    end

    @testset "ChainRulesTests" begin
        println("running ChainRulesTests...")
        t = @elapsed include("ChainRulesTests.jl")
        println("done (took $t seconds).")
    end

    @testset "ScalarTests" begin
        println("running ScalarTests...")
        t = @elapsed include("derivatives/ScalarTests.jl")
        println("done (took $t seconds).")
    end

    @testset "LinAlgTests" begin
        println("running LinAlgTests...")
        t = @elapsed include("derivatives/LinAlgTests.jl")
        println("done (took $t seconds).")
    end

    @testset "PrecisionTests" begin
        println("running PrecisionTests...")
        t = @elapsed include("PrecisionTests.jl")
        println("done (took $t seconds).")
    end

    @testset "ElementWiseTests" begin
        println("running ElementwiseTests...")
        t = @elapsed include("derivatives/ElementwiseTests.jl")
        println("done (took $t seconds).")
    end

    @testset "BroadcastTests" begin
        println("running BroadcastTests...")
        t = @elapsed include("derivatives/BroadcastTests.jl")
        println("done (took $t seconds).")
    end

    @testset "ArrayFunctionTests" begin
        println("running ArrayFunctionTests...")
        t = @elapsed include("derivatives/ArrayFunctionTests.jl")
        println("done (took $t seconds).")
    end

    @testset "GradientTests" begin
        println("running GradientTests...")
        t = @elapsed include("api/GradientTests.jl")
        println("done (took $t seconds).")
    end

    @testset "JacobianTests" begin
        println("running JacobianTests...")
        t = @elapsed include("api/JacobianTests.jl")
        println("done (took $t seconds).")
    end

    @testset "HessianTests" begin
        println("running HessianTests...")
        t = @elapsed include("api/HessianTests.jl")
        println("done (took $t seconds).")
    end

    @testset "ConfigTests" begin
        println("running ConfigTests...")
        t = @elapsed include("api/ConfigTests.jl")
        println("done (took $t seconds).")
    end

    @testset "CompatTests" begin
        println("running CompatTests...")
        t = @elapsed include("compat/CompatTests.jl")
        println("done (took $t seconds).")
    end
end
