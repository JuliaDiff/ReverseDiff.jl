using ReverseDiffPrototype
using Base.Test
using ForwardDiff

const RDP = ReverseDiffPrototype
const EPS = 1e-6

# make RNG deterministic, and thus make result inaccuracies
# deterministic so we don't have to retune EPS for arbitrary inputs
srand(2)

include("testfuncs.jl")

for f in UNARY_ARR2NUM_FUNCS

    x = rand(3, 3)
    test = ForwardDiff.hessian!(HessianResult(x), f, x, Chunk{1}())

    testprintln("gradient", f)

    out = similar(x)
    result = GradientResult(out)
    @test_approx_eq_eps RDP.gradient(f, x) test.gradient EPS
    @test_approx_eq_eps RDP.gradient!(out, f, x) test.gradient EPS
    RDP.gradient!(result, f, x)
    @test_approx_eq_eps result.value test.value EPS
    @test_approx_eq_eps result.gradient test.gradient EPS

    testprintln("hessian", f)

    out = similar(x, length(x), length(x))
    result = HessianResult(zero(test.value), similar(test.gradient), similar(test.hessian))
    @test_approx_eq_eps RDP.hessian(f, x) test.hessian EPS
    @test_approx_eq_eps RDP.hessian!(out, f, x) test.hessian EPS
    # RDP.hessian!(result, f, x)
    # @test_approx_eq_eps result.value test.value EPS
    # @test_approx_eq_eps result.gradient test.gradient EPS
    # @test_approx_eq_eps result.hessian test.hessian EPS
end

println("done")
