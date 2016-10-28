module LinAlgTests

using ReverseDiff, ForwardDiff, Base.Test

include("../utils.jl")

println("testing linear algebra derivatives (both forward and reverse passes)")
tic()

############################################################################################
x, a, b = rand(3, 3), rand(3, 3), rand(3, 3)
tp = Tape()

function test_arr2num(f, x, tp)
    xt = track(x, tp)
    y = f(x)

    # record
    yt = f(xt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    ReverseDiff.seed!(yt)
    ReverseDiff.reverse_pass!(tp)
    @test_approx_eq_eps adjoint(xt) ForwardDiff.gradient(f, x) EPS

    # forward
    x2 = rand(size(x))
    ReverseDiff.setvalue!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test value(yt) == f(x2)
    ReverseDiff.setvalue!(xt, x)

    empty!(tp)
end

function test_arr2arr(f, x, tp)
    xt = track(x, tp)
    y = f(x)

    # record
    yt = f(xt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out = similar(y, (length(y), length(x)))
    ReverseDiff.jacobian_reverse_pass!(out, yt, xt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(f, x) EPS

    # forward
    x2 = rand(size(x))
    ReverseDiff.setvalue!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test value(yt) == f(x2)
    ReverseDiff.setvalue!(xt, x)

    empty!(tp)
end

function test_arr2arr(f, a, b, tp)
    at, bt = track(a, tp), track(b, tp)
    c = f(a, b)

    ########################################

    # record
    ct = f(at, b)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(c), length(a)))
    ReverseDiff.jacobian_reverse_pass!(out, ct, at, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> f(x, b), a) EPS
    ReverseDiff.unseed!(tp)

    # forward
    a2 = rand(size(a))
    ReverseDiff.setvalue!(at, a2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == f(a2, b)
    ReverseDiff.setvalue!(at, a)

    empty!(tp)

    ########################################

    # record
    ct = f(a, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(c), length(b)))
    ReverseDiff.jacobian_reverse_pass!(out, ct, bt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> f(a, x), b) EPS
    ReverseDiff.unseed!(tp)

    # forward
    b2 = rand(size(b))
    ReverseDiff.setvalue!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == f(a, b2)
    ReverseDiff.setvalue!(bt, b)

    empty!(tp)

    ########################################

    # record
    ct = f(at, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out_a = similar(c, (length(c), length(a)))
    out_b = similar(c, (length(c), length(b)))
    ReverseDiff.jacobian_reverse_pass!(out_a, ct, at, tp)
    ReverseDiff.jacobian_reverse_pass!(out_b, ct, bt, tp)
    @test_approx_eq_eps out_a ForwardDiff.jacobian(x -> f(x, b), a) EPS
    @test_approx_eq_eps out_b ForwardDiff.jacobian(x -> f(a, x), b) EPS
    ReverseDiff.unseed!(tp)

    # forward
    a2, b2 = rand(size(a)), rand(size(b))
    ReverseDiff.setvalue!(at, a2)
    ReverseDiff.setvalue!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == f(a2, b2)
    ReverseDiff.setvalue!(at, a)
    ReverseDiff.setvalue!(bt, b)

    empty!(tp)
end

function test_arr2arr_inplace(f!, f, c, a, b, tp)
    at, bt = track(a, tp), track(b, tp)
    f!(c, a, b)

    ########################################

    # record
    ct = track(c, eltype(c), Nullable{Tape}())
    f!(ct, at, b)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(c), length(a)))
    ReverseDiff.jacobian_reverse_pass!(out, ct, at, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> f(x, b), a) EPS
    ReverseDiff.unseed!(tp)

    # forward
    a2 = rand(size(a))
    ReverseDiff.setvalue!(at, a2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == f(a2, b)
    ReverseDiff.setvalue!(at, a)

    empty!(tp)

    ########################################

    # record
    ct = track(c, eltype(c), Nullable{Tape}())
    f!(ct, a, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(c), length(b)))
    ReverseDiff.jacobian_reverse_pass!(out, ct, bt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> f(a, x), b) EPS
    ReverseDiff.unseed!(tp)

    # forward
    b2 = rand(size(b))
    ReverseDiff.setvalue!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == f(a, b2)
    ReverseDiff.setvalue!(bt, b)

    empty!(tp)

    ########################################

    # record
    ct = track(c, eltype(c), Nullable{Tape}())
    f!(ct, at, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out_a = similar(c, (length(c), length(a)))
    out_b = similar(c, (length(c), length(b)))
    ReverseDiff.jacobian_reverse_pass!(out_a, ct, at, tp)
    ReverseDiff.jacobian_reverse_pass!(out_b, ct, bt, tp)
    @test_approx_eq_eps out_a ForwardDiff.jacobian(x -> f(x, b), a) EPS
    @test_approx_eq_eps out_b ForwardDiff.jacobian(x -> f(a, x), b) EPS
    ReverseDiff.unseed!(tp)

    # forward
    a2, b2 = rand(size(a)), rand(size(b))
    ReverseDiff.setvalue!(at, a2)
    ReverseDiff.setvalue!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == f(a2, b2)
    ReverseDiff.setvalue!(at, a)
    ReverseDiff.setvalue!(bt, b)

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

for (f!, f) in ReverseDiff.A_MUL_B_FUNCS
    testprintln("A_mul_B functions", f)
    test_arr2arr(eval(f), a, b, tp)
    test_arr2arr_inplace(eval(f!), eval(f), x, a, b, tp)
end

############################################################################################

println("done (took $(toq()) seconds)")

end # module
