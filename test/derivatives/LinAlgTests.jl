module LinAlgTests

using ReverseDiffPrototype, ForwardDiff, Base.Test

include("../utils.jl")

println("testing linear algebra derivatives (both forward and reverse)")
tic()

############################################################################################
x, a, b = rand(3, 3), rand(3, 3), rand(3, 3)
tp = Tape()

function test_arr2num(f, x, tp)
    xt = track(x, tp)
    y = f(x)
    yt = f(xt)
    @test yt == y
    @test length(tp) == 1
    RDP.seed!(yt)
    RDP.reverse_pass!(tp)
    @test_approx_eq_eps adjoint(xt) ForwardDiff.gradient(f, x) EPS
    empty!(tp)
end

function test_arr2arr(f, x, tp)
    xt = track(x, tp)
    y = f(x)
    out = similar(y, (length(y), length(x)))
    yt = f(xt)
    @test yt == y
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out, yt, xt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(f, x) EPS
    empty!(tp)
end

function test_arr2arr(f, a, b, tp)
    at, bt = track(a, tp), track(b, tp)
    c = f(a, b)

    out = similar(c, (length(c), length(a)))
    ct = f(at, b)
    @test ct == c
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out, ct, at, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> f(x, b), a) EPS
    RDP.unseed!(tp)
    empty!(tp)

    out = similar(c, (length(c), length(b)))
    ct = f(a, bt)
    @test ct == c
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out, ct, bt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> f(a, x), b) EPS
    RDP.unseed!(tp)
    empty!(tp)

    out_a = similar(c, (length(c), length(a)))
    out_b = similar(c, (length(c), length(b)))
    ct = f(at, bt)
    @test ct == c
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out_a, ct, at, tp)
    RDP.jacobian_reverse_pass!(out_b, ct, bt, tp)
    @test_approx_eq_eps out_a ForwardDiff.jacobian(x -> f(x, b), a) EPS
    @test_approx_eq_eps out_b ForwardDiff.jacobian(x -> f(a, x), b) EPS
    RDP.unseed!(tp)
    empty!(tp)
end

function test_arr2arr_inplace(f!, f, c, a, b, tp)
    at, bt = track(a, tp), track(b, tp)
    f!(c, a, b)

    out = similar(c, (length(c), length(a)))
    ct = track(c, eltype(c), Nullable{Tape}())
    f!(ct, at, b)
    @test ct == c
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out, ct, at, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> f(x, b), a) EPS
    RDP.unseed!(tp)
    empty!(tp)

    out = similar(c, (length(c), length(b)))
    ct = track(c, eltype(c), Nullable{Tape}())
    f!(ct, a, bt)
    @test ct == c
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out, ct, bt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> f(a, x), b) EPS
    RDP.unseed!(tp)
    empty!(tp)

    out_a = similar(c, (length(c), length(a)))
    out_b = similar(c, (length(c), length(b)))
    ct = track(c, eltype(c), Nullable{Tape}())
    f!(ct, at, bt)
    @test ct == c
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out_a, ct, at, tp)
    RDP.jacobian_reverse_pass!(out_b, ct, bt, tp)
    @test_approx_eq_eps out_a ForwardDiff.jacobian(x -> f(x, b), a) EPS
    @test_approx_eq_eps out_b ForwardDiff.jacobian(x -> f(a, x), b) EPS
    RDP.unseed!(tp)
    empty!(tp)
end

for f in (sum, det)
    testprintln("Array -> Number functions", f)
    test_arr2num(f, x, tp)
end

for f in (-, inv)
    testprintln("Array -> Array functions", f)
    test_arr2arr(f, x, tp)
end

for f in (+, -)
    testprintln("(Array, Array) -> Array functions", f)
    test_arr2arr(f, a, b, tp)
end

for f in RDP.A_MUL_B_FUNCS
    testprintln("A_mul_B functions", f)
    test_arr2arr(eval(f), a, b, tp)
end

for (f!, f) in RDP.A_MUL_B!_FUNCS
    testprintln("A_mul_B! functions", f!)
    test_arr2arr_inplace(eval(f!), eval(f), x, a, b, tp)
end

############################################################################################

println("done (took $(toq()) seconds)")

end # module
