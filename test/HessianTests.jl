module HessianTests

using DiffBase, ForwardDiff, ReverseDiffPrototype, Base.Test

const RDP = ReverseDiffPrototype

include("utils.jl")

println("testing hessian/hessian!...")
tic()

############################################################################################

function test_unary_hessian(f, x)
    test = DiffBase.HessianResult(x)
    ForwardDiff.hessian!(test, f, x, ForwardDiff.HessianOptions{1}(test, x))

    @test_approx_eq_eps RDP.hessian(f, x) DiffBase.hessian(test) EPS

    # out = similar(DiffBase.hessian(test))
    # RDP.hessian!(out, f, x)
    #
    # @test_approx_eq_eps out DiffBase.hessian(test) EPS

    # result = RDP.hessian!(DiffBase.HessianResult(x), f, x)
    # @test_approx_eq_eps DiffBase.value(result)    DiffBase.value(test)    EPS
    # @test_approx_eq_eps DiffBase.gradient(result) DiffBase.gradient(test) EPS
    # @test_approx_eq_eps DiffBase.hessian(result)  DiffBase.hessian(test)  EPS
end

for f in DiffBase.MATRIX_TO_NUMBER_FUNCS
    testprintln("MATRIX_TO_NUMBER_FUNCS", f)
    test_unary_hessian(f, rand(5, 5))
end

for f in DiffBase.VECTOR_TO_NUMBER_FUNCS
    testprintln("VECTOR_TO_NUMBER_FUNCS", f)
    test_unary_hessian(f, rand(5))
end

# for f in DiffBase.TERNARY_MATRIX_TO_NUMBER_FUNCS
#     # testprintln("TERNARY_ARR2NUM_FUNCS", f)
#
#     # a, b, c = rand(5, 5), rand(5, 5), rand(5, 5)
#     #
#     # test_a = ForwardDiff.hessian!(DiffBase.HessianResult(a), x -> f(x, b, c), a)
#     # test_b = ForwardDiff.hessian!(DiffBase.HessianResult(b), x -> f(a, x, c), b)
#     # test_c = ForwardDiff.hessian!(DiffBase.HessianResult(c), x -> f(a, b, x), c)
#
#     # Ha, Hb, Hc = RDP.hessian(f, (a, b, c))
#     # @test_approx_eq_eps Ha test_a.hessian EPS
#     # @test_approx_eq_eps Hb test_b.hessian EPS
#     # @test_approx_eq_eps Hc test_c.hessian EPS
#     #
#     # Ha, Hb, Hc = zeros(9, 9), zeros(9, 9), zeros(9, 9)
#     # RDP.hessian!((Ha, Hb, Hc), f, (a, b, c))
#     # @test_approx_eq_eps Ha test_a.hessian EPS
#     # @test_approx_eq_eps Hb test_b.hessian EPS
#     # @test_approx_eq_eps Hc test_c.hessian EPS
#     #
#     # Ha, Hb, Hc = map(DiffBase.HessianResult, (a, b, c))
#     # RDP.hessian!((Ha, Hb, Hc), f, (a, b, c))
#     # @test_approx_eq_eps ∇a.value    test_a.value    EPS
#     # @test_approx_eq_eps ∇b.value    test_b.value    EPS
#     # @test_approx_eq_eps ∇c.value    test_c.value    EPS
#     # @test_approx_eq_eps ∇a.gradient test_a.gradient EPS
#     # @test_approx_eq_eps ∇b.gradient test_b.gradient EPS
#     # @test_approx_eq_eps ∇c.gradient test_c.gradient EPS
#     # @test_approx_eq_eps ∇a.hessian  test_a.hessian  EPS
#     # @test_approx_eq_eps ∇b.hessian  test_b.hessian  EPS
#     # @test_approx_eq_eps ∇c.hessian  test_c.hessian  EPS
# end

############################################################################################

println("done (took $(toq()) seconds)")

end # module
