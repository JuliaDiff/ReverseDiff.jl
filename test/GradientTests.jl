module GradientTests

using DiffBase, ForwardDiff, ReverseDiffPrototype, Base.Test

const RDP = ReverseDiffPrototype

include("utils.jl")

println("testing gradient/gradient!...")
tic()

############################################################################################
function test_unary_gradient(f, x)
    test = ForwardDiff.gradient!(DiffBase.GradientResult(x), f, x)
    out = similar(x)

    @test_approx_eq_eps RDP.gradient(f, x)       DiffBase.gradient(test) EPS
    @test_approx_eq_eps RDP.gradient!(out, f, x) DiffBase.gradient(test) EPS

    result = RDP.gradient!(DiffBase.GradientResult(x), f, x)
    @test_approx_eq_eps DiffBase.value(result)    DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.gradient(result) DiffBase.gradient(test) EPS

    opts = RDP.Options(x)
    @test_approx_eq_eps RDP.gradient(f, x, opts)       DiffBase.gradient(test) EPS
    @test_approx_eq_eps RDP.gradient!(out, f, x, opts) DiffBase.gradient(test) EPS

    result = RDP.gradient!(DiffBase.GradientResult(x), f, x, opts)
    @test_approx_eq_eps DiffBase.value(result)    DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.gradient(result) DiffBase.gradient(test) EPS
end

for f in DiffBase.MATRIX_TO_NUMBER_FUNCS
    testprintln("MATRIX_TO_NUMBER_FUNCS", f)
    test_unary_gradient(f, rand(5, 5))
end

for f in DiffBase.VECTOR_TO_NUMBER_FUNCS
    testprintln("VECTOR_TO_NUMBER_FUNCS", f)
    test_unary_gradient(f, rand(5))
end

for f in DiffBase.TERNARY_MATRIX_TO_NUMBER_FUNCS
    testprintln("TERNARY_MATRIX_TO_NUMBER_FUNCS", f)

    a, b, c = rand(5, 5), rand(5, 5), rand(5, 5)
    opts = RDP.Options((a, b, c))

    test_val = f(a, b, c)
    test_a = ForwardDiff.gradient(x -> f(x, b, c), a)
    test_b = ForwardDiff.gradient(x -> f(a, x, c), b)
    test_c = ForwardDiff.gradient(x -> f(a, b, x), c)

    ∇a, ∇b, ∇c = RDP.gradient(f, (a, b, c), opts)
    @test_approx_eq_eps ∇a test_a EPS
    @test_approx_eq_eps ∇b test_b EPS
    @test_approx_eq_eps ∇c test_c EPS

    ∇a, ∇b, ∇c = map(similar, (a, b, c))
    RDP.gradient!((∇a, ∇b, ∇c), f, (a, b, c), opts)
    @test_approx_eq_eps ∇a test_a EPS
    @test_approx_eq_eps ∇b test_b EPS
    @test_approx_eq_eps ∇c test_c EPS

    ∇a, ∇b, ∇c = map(DiffBase.GradientResult, (a, b, c))
    RDP.gradient!((∇a, ∇b, ∇c), f, (a, b, c), opts)
    @test_approx_eq_eps DiffBase.value(∇a) test_val EPS
    @test_approx_eq_eps DiffBase.value(∇b) test_val EPS
    @test_approx_eq_eps DiffBase.value(∇c) test_val EPS
    @test_approx_eq_eps DiffBase.gradient(∇a) test_a EPS
    @test_approx_eq_eps DiffBase.gradient(∇b) test_b EPS
    @test_approx_eq_eps DiffBase.gradient(∇c) test_c EPS
end
############################################################################################

println("done (took $(toq()) seconds)")

end # module
