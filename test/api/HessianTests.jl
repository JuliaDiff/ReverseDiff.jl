module HessianTests

using DiffBase, ForwardDiff, ReverseDiff, Base.Test

include(joinpath(dirname(@__FILE__), "../utils.jl"))

println("testing hessian/hessian!...")
tic()

############################################################################################

function test_unary_hessian(f, x)
    test = DiffBase.HessianResult(x)
    ForwardDiff.hessian!(test, f, x, ForwardDiff.HessianConfig{1}(test, x))

    # without HessianConfig

    test_approx(ReverseDiff.hessian(f, x), DiffBase.hessian(test))

    out = similar(DiffBase.hessian(test))
    ReverseDiff.hessian!(out, f, x)
    test_approx(out, DiffBase.hessian(test))

    result = DiffBase.HessianResult(x)
    ReverseDiff.hessian!(result, f, x)
    test_approx(DiffBase.value(result), DiffBase.value(test))
    test_approx(DiffBase.gradient(result), DiffBase.gradient(test))
    test_approx(DiffBase.hessian(result), DiffBase.hessian(test))

    # with HessianConfig

    cfg = ReverseDiff.HessianConfig(x)

    test_approx(ReverseDiff.hessian(f, x, cfg), DiffBase.hessian(test))

    out = similar(DiffBase.hessian(test))
    ReverseDiff.hessian!(out, f, x, cfg)
    test_approx(out, DiffBase.hessian(test))

    result = DiffBase.HessianResult(x)
    cfg = ReverseDiff.HessianConfig(result, x)
    ReverseDiff.hessian!(result, f, x, cfg)
    test_approx(DiffBase.value(result), DiffBase.value(test))
    test_approx(DiffBase.gradient(result), DiffBase.gradient(test))
    test_approx(DiffBase.hessian(result), DiffBase.hessian(test))

    # with HessianTape

    seedx = rand(size(x))
    tp = ReverseDiff.HessianTape(f, seedx)

    test_approx(ReverseDiff.hessian!(tp, x), DiffBase.hessian(test))

    out = similar(DiffBase.hessian(test))
    ReverseDiff.hessian!(out, tp, x)
    test_approx(out, DiffBase.hessian(test))

    result = DiffBase.HessianResult(x)
    ReverseDiff.hessian!(result, tp, x)
    test_approx(DiffBase.value(result), DiffBase.value(test))
    test_approx(DiffBase.gradient(result), DiffBase.gradient(test))
    test_approx(DiffBase.hessian(result), DiffBase.hessian(test))

    # with compiled HessianTape

    if length(tp.tape) <= 10000 # otherwise compile time can be crazy
        Hf! = ReverseDiff.compile_hessian(f, seedx)
        ctp = ReverseDiff.compile(tp)

        # circumvent world-age problems (`ctp` and `Hf!` have a future world age)
        @eval begin
            test, x,) = $test, $x, $EPS
            ctp, Hf! = $ctp, $Hf!

            test_approx(ReverseDiff.hessian!(ctp, x), DiffBase.hessian(test))

            out = similar(DiffBase.hessian(test))
            ReverseDiff.hessian!(out, ctp, x)
            test_approx(out, DiffBase.hessian(test))

            out = similar(DiffBase.hessian(test))
            Hf!(out, x)
            test_approx(out, DiffBase.hessian(test))

            result = DiffBase.HessianResult(x)
            ReverseDiff.hessian!(result, ctp, x)
            test_approx(DiffBase.value(result), DiffBase.value(test))
            test_approx(DiffBase.gradient(result), DiffBase.gradient(test))
            test_approx(DiffBase.hessian(result), DiffBase.hessian(test))

            result = DiffBase.HessianResult(x)
            Hf!(result, x)
            test_approx(DiffBase.value(result), DiffBase.value(test))
            test_approx(DiffBase.gradient(result), DiffBase.gradient(test))
            test_approx(DiffBase.hessian(result), DiffBase.hessian(test))
        end
    end
end

for f in DiffBase.MATRIX_TO_NUMBER_FUNCS
    test_println("MATRIX_TO_NUMBER_FUNCS", f)
    test_unary_hessian(f, rand(5, 5))
end

for f in DiffBase.VECTOR_TO_NUMBER_FUNCS
    test_println("VECTOR_TO_NUMBER_FUNCS", f)
    test_unary_hessian(f, rand(5))
end

############################################################################################

println("done (took $(toq()) seconds)")

end # module
