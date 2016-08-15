module ArrayTests

using ReverseDiffPrototype
using Base.Test
using ForwardDiff

const RDP = ReverseDiffPrototype
const EPS = 1e-6

for f in (-, det, inv)
    Main.testprintln(f)

    freduce = (r, x) -> begin
        N = isqrt(length(x))
        return r(f(reshape(x, N, N)))
    end

    y = rand(4)

    fprod = x -> freduce(prod, x)
    @test_approx_eq_eps RDP.gradient(fprod, y) ForwardDiff.gradient(fprod, y) EPS

    fsum = x -> freduce(sum, x)
    @test_approx_eq_eps RDP.gradient(fsum, y) ForwardDiff.gradient(fsum, y) EPS
end

for f in (+, .+, -, .-, *, .*, ./, .^,
          A_mul_Bt, At_mul_B, At_mul_Bt,
          A_mul_Bc, Ac_mul_B, Ac_mul_Bc)
    Main.testprintln(f)

    freduce = (r, x, s) -> begin
        N = isqrt(length(x))
        A = reshape(x, N, N)
        B = s * A
        return r(f(A, B))
    end

    y, n = rand(4), rand()

    fprod = x -> freduce(prod, x, n)
    @test_approx_eq_eps RDP.gradient(fprod, y) ForwardDiff.gradient(fprod, y) EPS

    fsum = x -> freduce(sum, x, n)
    @test_approx_eq_eps RDP.gradient(fsum, y) ForwardDiff.gradient(fsum, y) EPS
end

end # module
