module ArrayTests

using ReverseDiffPrototype
using Base.Test
using ForwardDiff

const RDP = ReverseDiffPrototype
const EPS = 1e-6

for f in (det,)
    Main.testprintln(f)
    x = rand(3, 3)
    @test_approx_eq_eps RDP.gradient(f, x) ForwardDiff.gradient(f, x) EPS
    # @test_approx_eq_eps RDP.hessian(f, x) ForwardDiff.hessian(f, x) EPS
end

for f in (-, inv)
    Main.testprintln(f)
    x = rand(3, 3)
    # f2 = y -> RDP.jacobian(f, y)
    @test_approx_eq_eps RDP.jacobian(f, x) ForwardDiff.jacobian(f, x) EPS
    # @test_approx_eq_eps RDP.jacobian(f2, x) ForwardDiff.jacobian(f2, x) EPS
end

for f in (+, .+, -, .-, *, .*, ./, .^,
          A_mul_Bt, At_mul_B, At_mul_Bt,
          A_mul_Bc, Ac_mul_B, Ac_mul_Bc)
    Main.testprintln(f)

    A, B = rand(3, 3), rand(3, 3)

    f1 = X -> f(X, B)
    dA = RDP.jacobian(f1, A)
    @test_approx_eq_eps dA ForwardDiff.jacobian(f1, A) EPS

    f2 = X -> f(A, X)
    dB = RDP.jacobian(f2, B)
    @test_approx_eq_eps dB ForwardDiff.jacobian(f2, B) EPS

    dA2, dB2 = RDP.jacobian(f, (A, B))

    @test_approx_eq_eps dA dA2 EPS
    @test_approx_eq_eps dB dB2 EPS
end

end # module
