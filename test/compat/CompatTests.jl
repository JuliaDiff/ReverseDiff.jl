module CompatTests

using DiffResults, FillArrays, StaticArrays, ReverseDiff, Test

@testset "FillArrays" begin
    @test ReverseDiff.gradient(fill(2.0, 3)) do x
        sum(abs2.(x .- Zeros(3)))
    end == fill(4.0, 3)

    @test ReverseDiff.gradient(fill(2.0, 3)) do x
        sum(abs2.(x .- (1:3)))
    end == [2, 0, -2]
end

sumabs2(x) = sum(abs2, x)

@testset "StaticArrays" begin
    @testset "Gradient" begin
        x = MVector{2}(3.0, 4.0)
        result = ReverseDiff.gradient!(DiffResults.GradientResult(x), sumabs2, x)
        @test_broken x == [3.0, 4.0]
        @test_broken DiffResults.value(result) == 25.0
    end

    @testset "Hessian" begin
        x = MVector{2}(3.0, 4.0)
        result = ReverseDiff.hessian!(DiffResults.HessianResult(x), sumabs2, x)
        @test_broken x == [3.0, 4.0]
        @test_broken DiffResults.value(result) == 25.0
    end
end

end
