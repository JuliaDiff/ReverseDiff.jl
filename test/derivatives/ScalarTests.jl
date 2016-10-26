module ScalarTests

using ReverseDiffPrototype, ForwardDiff, Base.Test

include("../utils.jl")

println("testing scalar derivatives (both forward and reverse)")
tic()

############################################################################################
x, a, b = rand(3)
tp = Tape()

function test_forward(f, x, tp)
    tx = track(x, tp)
    y = f(x)
    ty = f(tx)
    @test ty == y
    RDP.seed!(ty)
    RDP.reverse_pass!(tp)
    @test adjoint(tx) == ForwardDiff.derivative(f, x)
    empty!(tp)
end

function test_forward(f, a, b, tp)
    ta, tb = track(a, tp), track(b, tp)
    c = f(a, b)
    tc = f(ta, tb)
    @test tc == c
    RDP.seed!(tc)
    RDP.reverse_pass!(tp)
    @test_approx_eq_eps adjoint(ta) ForwardDiff.derivative(x -> f(x, b), a) EPS
    @test_approx_eq_eps adjoint(tb) ForwardDiff.derivative(x -> f(a, x), b) EPS
    empty!(tp)
end

function test_skip(f, x, tp)
    tx = track(x, tp)
    y = f(x)
    ty = f(tx)
    @test ty == y
    @test isempty(tp)
end

function test_skip(f, a, b, tp)
    ta, tb = track(a, tp), track(b, tp)
    c = f(a, b)
    tc = f(ta, tb)
    @test tc == c
    @test isempty(tp)
end

DOMAIN_ERR_FUNCS = (:asec, :acsc, :asecd, :acscd, :acoth, :acosh)

testprintln("FORWARD_UNARY_SCALAR_FUNCS", "(too many to print)")
for f in RDP.FORWARD_UNARY_SCALAR_FUNCS
    n = in(f, DOMAIN_ERR_FUNCS) ? x + 1 : x
    test_forward(eval(f), n, tp)
end

testprintln("FORWARD_BINARY_SCALAR_FUNCS", "(too many to print)")
for f in RDP.FORWARD_BINARY_SCALAR_FUNCS
    test_forward(eval(f), a, b, tp)
end

INT_ONLY_FUNCS = (:iseven, :isodd)

testprintln("SKIPPED_UNARY_SCALAR_FUNCS", "(too many to print)")
for f in RDP.SKIPPED_UNARY_SCALAR_FUNCS
    n = in(f, DOMAIN_ERR_FUNCS) ? x + 1 : x
    n = in(f, INT_ONLY_FUNCS) ? ceil(Int, n) : n
    test_skip(eval(f), n, tp)
end

testprintln("SKIPPED_BINARY_SCALAR_FUNCS", "(too many to print)")
for f in RDP.SKIPPED_BINARY_SCALAR_FUNCS
    test_skip(eval(f), a, b, tp)
end

############################################################################################

println("done (took $(toq()) seconds)")

end # module
