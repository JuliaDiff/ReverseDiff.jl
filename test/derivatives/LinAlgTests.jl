module LinAlgTests

using ReverseDiff, ForwardDiff, Base.Test

include(joinpath(dirname(@__FILE__), "../utils.jl"))

println("testing linear algebra derivatives (both forward and reverse passes)")
tic()

############################################################################################
x, a, b = rand(3, 3), rand(3, 3), rand(3, 3)
tp = Tape()

function test_arr2num(f, x, tp)
    xt = track(copy(x), tp)
    y = f(x)

    # record
    yt = f(xt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    ReverseDiff.seed!(yt)
    ReverseDiff.reverse_pass!(tp)
    @test_approx_eq_eps deriv(xt) ForwardDiff.gradient(f, x) EPS

    # forward
    x2 = rand(size(x))
    ReverseDiff.value!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test value(yt) == f(x2)
    ReverseDiff.value!(xt, x)

    empty!(tp)
end

function test_arr2arr(f, x, tp)
    xt = track(copy(x), tp)
    y = f(x)

    # record
    yt = f(xt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out = similar(y, (length(y), length(x)))
    ReverseDiff.seeded_reverse_pass!(out, yt, xt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(f, x) EPS

    # forward
    x2 = rand(size(x))
    ReverseDiff.value!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test value(yt) == f(x2)
    ReverseDiff.value!(xt, x)

    empty!(tp)
end

function test_arr2arr(f, a, b, tp)
    at, bt = track(copy(a), tp), track(copy(b), tp)
    c = f(a, b)

    ########################################

    # record
    ct = f(at, b)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(c), length(a)))
    ReverseDiff.seeded_reverse_pass!(out, ct, at, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> f(x, b), a) EPS

    # forward
    a2 = rand(size(a))
    ReverseDiff.value!(at, a2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == f(a2, b)
    ReverseDiff.value!(at, a)

    empty!(tp)

    ########################################

    # record
    ct = f(a, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(c), length(b)))
    ReverseDiff.seeded_reverse_pass!(out, ct, bt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> f(a, x), b) EPS

    # forward
    b2 = rand(size(b))
    ReverseDiff.value!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == f(a, b2)
    ReverseDiff.value!(bt, b)

    empty!(tp)

    ########################################

    # record
    ct = f(at, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out_a = similar(c, (length(c), length(a)))
    out_b = similar(c, (length(c), length(b)))
    ReverseDiff.seeded_reverse_pass!(out_a, ct, at, tp)
    ReverseDiff.seeded_reverse_pass!(out_b, ct, bt, tp)
    @test_approx_eq_eps out_a ForwardDiff.jacobian(x -> f(x, b), a) EPS
    @test_approx_eq_eps out_b ForwardDiff.jacobian(x -> f(a, x), b) EPS

    # forward
    a2, b2 = rand(size(a)), rand(size(b))
    ReverseDiff.value!(at, a2)
    ReverseDiff.value!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == f(a2, b2)
    ReverseDiff.value!(at, a)
    ReverseDiff.value!(bt, b)

    empty!(tp)
end

function test_arr2arr_inplace(f!, f, c, a, b, tp)
    at, bt = track(copy(a), tp), track(copy(b), tp)
    f!(c, a, b)

    ########################################

    # record
    ct = track(c, eltype(c), ReverseDiff.NULL_TAPE)
    f!(ct, at, b)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(c), length(a)))
    ReverseDiff.seeded_reverse_pass!(out, ct, at, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> f(x, b), a) EPS

    # forward
    a2 = rand(size(a))
    ReverseDiff.value!(at, a2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == f(a2, b)
    ReverseDiff.value!(at, a)

    empty!(tp)

    ########################################

    # record
    ct = track(c, eltype(c), ReverseDiff.NULL_TAPE)
    f!(ct, a, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(c), length(b)))
    ReverseDiff.seeded_reverse_pass!(out, ct, bt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> f(a, x), b) EPS

    # forward
    b2 = rand(size(b))
    ReverseDiff.value!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == f(a, b2)
    ReverseDiff.value!(bt, b)

    empty!(tp)

    ########################################

    # record
    ct = track(c, eltype(c), ReverseDiff.NULL_TAPE)
    f!(ct, at, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out_a = similar(c, (length(c), length(a)))
    out_b = similar(c, (length(c), length(b)))
    ReverseDiff.seeded_reverse_pass!(out_a, ct, at, tp)
    ReverseDiff.seeded_reverse_pass!(out_b, ct, bt, tp)
    @test_approx_eq_eps out_a ForwardDiff.jacobian(x -> f(x, b), a) EPS
    @test_approx_eq_eps out_b ForwardDiff.jacobian(x -> f(a, x), b) EPS

    # forward
    a2, b2 = rand(size(a)), rand(size(b))
    ReverseDiff.value!(at, a2)
    ReverseDiff.value!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == f(a2, b2)
    ReverseDiff.value!(at, a)
    ReverseDiff.value!(bt, b)

    empty!(tp)
end

for f in (sum, det, y -> dot(vec(y), vec(y)))
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
