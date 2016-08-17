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
end

for f in (-, inv)
    Main.testprintln(f)
    x = rand(3, 3)
    @test_approx_eq_eps RDP.jacobian(f, x) ForwardDiff.jacobian(f, x) EPS
end

for f in (+, .+, -, .-, *, .*, ./, .^,
          A_mul_Bt, At_mul_B, At_mul_Bt,
          A_mul_Bc, Ac_mul_B, Ac_mul_Bc)
    Main.testprintln(f)

    A, B = rand(3, 3), rand(3, 3)

    dfA = X -> f(X, B)
    @test_approx_eq_eps RDP.jacobian(dfA, A) ForwardDiff.jacobian(dfA, A) EPS

    dfB = X -> f(A, X)
    @test_approx_eq_eps RDP.jacobian(dfB, B) ForwardDiff.jacobian(dfB, B) EPS
end

end # module
