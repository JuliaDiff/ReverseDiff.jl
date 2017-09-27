module HessianTests

using DiffTests, ForwardDiff, ReverseDiff, Base.Test

include(joinpath(dirname(@__FILE__), "../utils.jl"))

println("testing hessian/hessian!...")
tic()

# Circumvent type inference bug where an erroneous `Any` eltype derails the computation
# during tests, but not outside of the tests. This is really hacky, but I couldn't figure
# out enough of an MRE for the bug to report it...
ReverseDiff.hessian(DiffTests.mat2num_1, rand(3, 3))

############################################################################################

function test_unary_hessian(f, x)
    test = DiffResults.HessianResult(x)
    ForwardDiff.hessian!(test, f, x, ForwardDiff.HessianConfig(f, test, x, ForwardDiff.Chunk{1}()))

    # without HessianConfig

    test_approx(ReverseDiff.hessian(f, x), DiffResults.hessian(test))

    out = similar(DiffResults.hessian(test))
    ReverseDiff.hessian!(out, f, x)
    test_approx(out, DiffResults.hessian(test))

    result = DiffResults.HessianResult(x)
    ReverseDiff.hessian!(result, f, x)
    test_approx(DiffResults.value(result), DiffResults.value(test))
    test_approx(DiffResults.gradient(result), DiffResults.gradient(test))
    test_approx(DiffResults.hessian(result), DiffResults.hessian(test))

    # with HessianConfig

    cfg = ReverseDiff.HessianConfig(x)

    test_approx(ReverseDiff.hessian(f, x, cfg), DiffResults.hessian(test))

    out = similar(DiffResults.hessian(test))
    ReverseDiff.hessian!(out, f, x, cfg)
    test_approx(out, DiffResults.hessian(test))

    result = DiffResults.HessianResult(x)
    cfg = ReverseDiff.HessianConfig(result, x)
    ReverseDiff.hessian!(result, f, x, cfg)
    test_approx(DiffResults.value(result), DiffResults.value(test))
    test_approx(DiffResults.gradient(result), DiffResults.gradient(test))
    test_approx(DiffResults.hessian(result), DiffResults.hessian(test))

    # with HessianTape

    seedx = rand(size(x))
    tp = ReverseDiff.HessianTape(f, seedx)

    test_approx(ReverseDiff.hessian!(tp, x), DiffResults.hessian(test))

    out = similar(DiffResults.hessian(test))
    ReverseDiff.hessian!(out, tp, x)
    test_approx(out, DiffResults.hessian(test))

    result = DiffResults.HessianResult(x)
    ReverseDiff.hessian!(result, tp, x)
    test_approx(DiffResults.value(result), DiffResults.value(test))
    test_approx(DiffResults.gradient(result), DiffResults.gradient(test))
    test_approx(DiffResults.hessian(result), DiffResults.hessian(test))

    # with compiled HessianTape

    if length(tp.tape) <= 10000 # otherwise compile time can be crazy
        ctp = ReverseDiff.compile(tp)

        test_approx(ReverseDiff.hessian!(ctp, x), DiffResults.hessian(test))

        out = similar(DiffResults.hessian(test))
        ReverseDiff.hessian!(out, ctp, x)
        test_approx(out, DiffResults.hessian(test))

        result = DiffResults.HessianResult(x)
        ReverseDiff.hessian!(result, ctp, x)
        test_approx(DiffResults.value(result), DiffResults.value(test))
        test_approx(DiffResults.gradient(result), DiffResults.gradient(test))
        test_approx(DiffResults.hessian(result), DiffResults.hessian(test))
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

############################################################################################

println("done (took $(toq()) seconds)")

end # module
