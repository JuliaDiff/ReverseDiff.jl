using ReverseDiffPrototype
using Base.Test
using ForwardDiff

const RDP = ReverseDiffPrototype
const EPS = 1e-6

# make RNG deterministic, and thus make result inaccuracies
# deterministic so we don't have to retune EPS for arbitrary inputs
srand(1)

include("testfuncs.jl")

println("testing gradient/gradient!...")
tic()

for f in UNARY_ARR2NUM_FUNCS
    testprintln("UNARY_ARR2NUM_FUNCS", f)

    x = rand(3, 3)
    test = ForwardDiff.gradient!(GradientResult(x), f, x)

    @test_approx_eq_eps RDP.gradient(f, x) test.gradient EPS
    @test_approx_eq_eps RDP.gradient!(similar(x), f, x) test.gradient EPS

    result = RDP.gradient!(GradientResult(x), f, x)
    @test_approx_eq_eps result.value test.value EPS
    @test_approx_eq_eps result.gradient test.gradient EPS
end

for f in TERNARY_ARR2NUM_FUNCS
    testprintln("TERNARY_ARR2NUM_FUNCS", f)

    a, b, c = rand(3), rand(3, 3), rand(3)

    test_val = f(a, b, c)
    test_a = ForwardDiff.gradient(x -> f(x, b, c), a)
    test_b = ForwardDiff.gradient(x -> f(a, x, c), b)
    test_c = ForwardDiff.gradient(x -> f(a, b, x), c)

    ∇a, ∇b, ∇c = RDP.gradient(f, (a, b, c))
    @test_approx_eq_eps ∇a test_a EPS
    @test_approx_eq_eps ∇b test_b EPS
    @test_approx_eq_eps ∇c test_c EPS

    ∇a, ∇b, ∇c = map(similar, (a, b, c))
    RDP.gradient!((∇a, ∇b, ∇c), f, (a, b, c))
    @test_approx_eq_eps ∇a test_a EPS
    @test_approx_eq_eps ∇b test_b EPS
    @test_approx_eq_eps ∇c test_c EPS

    ∇a, ∇b, ∇c = map(GradientResult, (a, b, c))
    RDP.gradient!((∇a, ∇b, ∇c), f, (a, b, c))
    @test_approx_eq_eps ∇a.value test_val EPS
    @test_approx_eq_eps ∇b.value test_val EPS
    @test_approx_eq_eps ∇c.value test_val EPS
    @test_approx_eq_eps ∇a.gradient test_a EPS
    @test_approx_eq_eps ∇b.gradient test_b EPS
    @test_approx_eq_eps ∇c.gradient test_c EPS
end

println("done (took $(toq()) seconds)")


println("testing jacobian/jacobian!...")
tic()

for f in UNARY_ARR2ARR_FUNCS
    testprintln("UNARY_ARR2ARR_FUNCS", f)

    x = rand(3, 3)
    test = ForwardDiff.jacobian!(JacobianResult(x), (y, x) -> copy!(y, f(x)), x)

    @test_approx_eq_eps RDP.jacobian(f, x) test.jacobian EPS
    @test_approx_eq_eps RDP.jacobian!(similar(test.jacobian), f, x) test.jacobian EPS

    result = RDP.jacobian!(JacobianResult(x), f, x)
    @test_approx_eq_eps result.value test.value EPS
    @test_approx_eq_eps result.jacobian test.jacobian EPS
end

for f in BINARY_ARR2ARR_FUNCS
    testprintln("BINARY_ARR2ARR_FUNCS", f)

    a, b = rand(3, 3), rand(3, 3)

    test_val = f(a, b)
    test_a = ForwardDiff.jacobian(x -> f(x, b), a)
    test_b = ForwardDiff.jacobian(x -> f(a, x), b)

    Ja, Jb = RDP.jacobian(f, (a, b))
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    Ja, Jb = map(x -> similar(x, length(x), length(x)), (a, b))
    RDP.jacobian!((Ja, Jb), f, (a, b))
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    Ja, Jb = map(JacobianResult, (a, b))
    RDP.jacobian!((Ja, Jb), f, (a, b))
    @test_approx_eq_eps Ja.value test_val EPS
    @test_approx_eq_eps Jb.value test_val EPS
    @test_approx_eq_eps Ja.jacobian test_a EPS
    @test_approx_eq_eps Jb.jacobian test_b EPS
end

println("done (took $(toq()) seconds)")
