module HessianTests

using DiffBase, ForwardDiff, ReverseDiffPrototype, Base.Test

const RDP = ReverseDiffPrototype

include("../utils.jl")

println("testing hessian/hessian!...")
tic()

############################################################################################

function test_unary_hessian(f, x)
    test = DiffBase.HessianResult(x)
    ForwardDiff.hessian!(test, f, x, ForwardDiff.HessianOptions{1}(test, x))

    # without HessianOptions

    @test_approx_eq_eps RDP.hessian(f, x) DiffBase.hessian(test) EPS

    out = similar(DiffBase.hessian(test))
    RDP.hessian!(out, f, x)
    @test_approx_eq_eps out DiffBase.hessian(test) EPS

    result = DiffBase.HessianResult(x)
    RDP.hessian!(result, f, x)
    # @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
    # @test_approx_eq_eps DiffBase.gradient(result) DiffBase.gradient(test) EPS
    # @test_approx_eq_eps DiffBase.hessian(result) DiffBase.hessian(test) EPS

    # with HessianOptions

    opts = RDP.HessianOptions(x)

    @test_approx_eq_eps RDP.hessian(f, x, opts) DiffBase.hessian(test) EPS

    out = similar(DiffBase.hessian(test))
    RDP.hessian!(out, f, x, opts)
    @test_approx_eq_eps out DiffBase.hessian(test) EPS

    # result = DiffBase.HessianResult(x)
    # opts = RDP.HessianOptions(result, x)
    # RDP.hessian!(result, f, x, opts)
    # @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
    # @test_approx_eq_eps DiffBase.gradient(result) DiffBase.gradient(test) EPS
    # @test_approx_eq_eps DiffBase.hessian(result) DiffBase.hessian(test) EPS
end

for f in DiffBase.MATRIX_TO_NUMBER_FUNCS
    testprintln("MATRIX_TO_NUMBER_FUNCS", f)
    test_unary_hessian(f, rand(5, 5))
end

for f in DiffBase.VECTOR_TO_NUMBER_FUNCS
    testprintln("VECTOR_TO_NUMBER_FUNCS", f)
    test_unary_hessian(f, rand(5))
end

############################################################################################

println("done (took $(toq()) seconds)")

end # module
