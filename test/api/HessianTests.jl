module HessianTests

using DiffTests, ForwardDiff, ReverseDiff, StaticArrays, Test

include(joinpath(dirname(@__FILE__), "../utils.jl"))

# Circumvent type inference bug where an erroneous `Any` eltype derails the computation
# during tests, but not outside of the tests. This is really hacky, but I couldn't figure
# out enough of an MRE for the bug to report it...
ReverseDiff.hessian(DiffTests.mat2num_1, rand(3, 3))

hess_test_approx(a, b) = test_approx(a, b, 1e-4)

function test_unary_hessian(f, x)
    test = DiffResults.HessianResult(x)
    ForwardDiff.hessian!(test, f, x, ForwardDiff.HessianConfig(f, test, x, ForwardDiff.Chunk{1}()))

    # without HessianConfig

    hess_test_approx(ReverseDiff.hessian(f, x), DiffResults.hessian(test))

    out = similar(DiffResults.hessian(test))
    ReverseDiff.hessian!(out, f, x)
    hess_test_approx(out, DiffResults.hessian(test))

    result = DiffResults.HessianResult(x)
    ReverseDiff.hessian!(result, f, x)
    hess_test_approx(DiffResults.value(result), DiffResults.value(test))
    hess_test_approx(DiffResults.gradient(result), DiffResults.gradient(test))
    hess_test_approx(DiffResults.hessian(result), DiffResults.hessian(test))

    # with HessianConfig

    cfg = ReverseDiff.HessianConfig(x)

    hess_test_approx(ReverseDiff.hessian(f, x, cfg), DiffResults.hessian(test))

    out = similar(DiffResults.hessian(test))
    ReverseDiff.hessian!(out, f, x, cfg)
    hess_test_approx(out, DiffResults.hessian(test))

    result = DiffResults.HessianResult(x)
    cfg = ReverseDiff.HessianConfig(result, x)
    ReverseDiff.hessian!(result, f, x, cfg)
    hess_test_approx(DiffResults.value(result), DiffResults.value(test))
    hess_test_approx(DiffResults.gradient(result), DiffResults.gradient(test))
    hess_test_approx(DiffResults.hessian(result), DiffResults.hessian(test))

    # with HessianTape

    seedx = rand(eltype(x), size(x))
    tp = ReverseDiff.HessianTape(f, seedx)

    hess_test_approx(ReverseDiff.hessian!(tp, x), DiffResults.hessian(test))

    out = similar(DiffResults.hessian(test))
    ReverseDiff.hessian!(out, tp, x)
    hess_test_approx(out, DiffResults.hessian(test))

    result = DiffResults.HessianResult(x)
    ReverseDiff.hessian!(result, tp, x)
    hess_test_approx(DiffResults.value(result), DiffResults.value(test))
    hess_test_approx(DiffResults.gradient(result), DiffResults.gradient(test))
    hess_test_approx(DiffResults.hessian(result), DiffResults.hessian(test))

    # with compiled HessianTape

    if length(tp.tape) <= 10000 # otherwise compile time can be crazy
        ctp = ReverseDiff.compile(tp)

        hess_test_approx(ReverseDiff.hessian!(ctp, x), DiffResults.hessian(test))

        out = similar(DiffResults.hessian(test))
        ReverseDiff.hessian!(out, ctp, x)
        hess_test_approx(out, DiffResults.hessian(test))

        result = DiffResults.HessianResult(x)
        ReverseDiff.hessian!(result, ctp, x)
        hess_test_approx(DiffResults.value(result), DiffResults.value(test))
        hess_test_approx(DiffResults.gradient(result), DiffResults.gradient(test))
        hess_test_approx(DiffResults.hessian(result), DiffResults.hessian(test))
    end
end

for f in DiffTests.MATRIX_TO_NUMBER_FUNCS
    test_println("MATRIX_TO_NUMBER_FUNCS", f)
    test_unary_hessian(f, rand(5, 5))
end

for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
    test_println("VECTOR_TO_NUMBER_FUNCS", f)
    test_unary_hessian(f, rand(5))
end

############################################################

@testset "`float` keeps the tape (#107, #276)" begin
    x = [2.0, 3.0]

    ff(x) = float(x[1]^3 * x[2] + x[2]^2)
    @test ReverseDiff.gradient(ff, x) == [36.0, 14.0]
    @test ReverseDiff.hessian(ff, x) == [36.0 12.0; 12.0 2.0]

    fa(x) = sum((float(x)::ReverseDiff.TrackedArray) .^ 3)
    @test ReverseDiff.gradient(fa, x) == [12.0, 27.0]
    @test ReverseDiff.hessian(fa, x) == [12.0 0.0; 0.0 18.0]
end

############################################################################################

f251(x) = sum(abs2, x)

@testset "primal value of an immutable result (#251)" begin
    x = MVector{2}(3.0, 5.0)
    value, grad, hess = 34.0, [6.0, 10.0], [2.0 0.0; 0.0 2.0]

    result = DiffResults.HessianResult(MVector(x))
    result = ReverseDiff.hessian!(result, f251, x, ReverseDiff.HessianConfig(result, x))
    @test result isa DiffResults.ImmutableDiffResult
    @test DiffResults.value(result) == value
    @test DiffResults.gradient(result) == grad
    @test DiffResults.hessian(result) == hess

    tape = ReverseDiff.HessianTape(f251, x)
    for t in (tape, ReverseDiff.compile(tape))
        result = ReverseDiff.hessian!(DiffResults.HessianResult(MVector(x)), t, x)
        @test DiffResults.value(result) == value
        @test DiffResults.gradient(result) == grad
        @test DiffResults.hessian(result) == hess
    end
end

end # module
