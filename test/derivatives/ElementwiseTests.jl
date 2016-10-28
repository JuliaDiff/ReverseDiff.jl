module ElementwiseTests

using ReverseDiff, ForwardDiff, Base.Test

include("../utils.jl")

println("testing elementwise derivatives (both forward and reverse passes)")
tic()

############################################################################################
x, y = rand(3, 3), rand(3, 3)
a, b = rand(3), rand(3)
n = rand()
tp = Tape()

function test_elementwise(f, x, tp)
    xt = track(x, tp)

    ########################################

    y = map(f, x)

    # record
    yt = map(ReverseDiff.@forward(f), xt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out = similar(y, (length(x), length(x)))
    ReverseDiff.jacobian_reverse_pass!(out, yt, xt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(z -> map(f, z), x) EPS

    # forward
    x2 = rand(size(x))
    ReverseDiff.setvalue!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test value(yt) == map(f, x2)
    ReverseDiff.setvalue!(xt, x)

    empty!(tp)

    ########################################

    y = broadcast(f, x)

    # record
    yt = broadcast(ReverseDiff.@forward(f), xt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out = similar(y, (length(x), length(x)))
    ReverseDiff.jacobian_reverse_pass!(out, yt, xt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(z -> broadcast(f, z), x) EPS

    # forward
    x2 = rand(size(x))
    ReverseDiff.setvalue!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test value(yt) == broadcast(f, x2)
    ReverseDiff.setvalue!(xt, x)

    empty!(tp)

end

function test_map(f, a, b, tp)
    at, bt = track(a, tp), track(b, tp)
    c = map(f, a, b)

    ########################################

    # record
    ct = map(ReverseDiff.@forward(f), at, b)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(a), length(a)))
    ReverseDiff.jacobian_reverse_pass!(out, ct, at, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> map(f, x, b), a) EPS
    ReverseDiff.unseed!(tp)

    # forward
    a2 = rand(size(a))
    ReverseDiff.setvalue!(at, a2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == map(f, a2, b)
    ReverseDiff.setvalue!(at, a)

    empty!(tp)

    ########################################

    # record
    ct = map(ReverseDiff.@forward(f), a, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(a), length(a)))
    ReverseDiff.jacobian_reverse_pass!(out, ct, bt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> map(f, a, x), b) EPS
    ReverseDiff.unseed!(tp)

    # forward
    b2 = rand(size(b))
    ReverseDiff.setvalue!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == map(f, a, b2)
    ReverseDiff.setvalue!(bt, b)

    empty!(tp)

    ########################################

    # record
    ct = map(ReverseDiff.@forward(f), at, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out_a = similar(c, (length(a), length(a)))
    out_b = similar(c, (length(a), length(a)))
    ReverseDiff.jacobian_reverse_pass!(out_a, ct, at, tp)
    ReverseDiff.jacobian_reverse_pass!(out_b, ct, bt, tp)
    @test_approx_eq_eps out_a ForwardDiff.jacobian(x -> map(f, x, b), a) EPS
    @test_approx_eq_eps out_b ForwardDiff.jacobian(x -> map(f, a, x), b) EPS
    ReverseDiff.unseed!(tp)

    # forward
    a2, b2 = rand(size(a)), rand(size(b))
    ReverseDiff.setvalue!(at, a2)
    ReverseDiff.setvalue!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == map(f, a2, b2)
    ReverseDiff.setvalue!(at, a)
    ReverseDiff.setvalue!(bt, b)

    empty!(tp)
end

function test_broadcast(f, a::AbstractArray, b::AbstractArray, tp, builtin = false)
    at, bt = track(a, tp), track(b, tp)

    if builtin
        g = ReverseDiff.@forward(f)
    else
        g = (x, y) -> broadcast(ReverseDiff.@forward(f), x, y)
    end

    c = g(a, b)

    ########################################

    # record
    ct = g(at, b)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(c), length(a)))
    ReverseDiff.jacobian_reverse_pass!(out, ct, at, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> g(x, b), a) EPS
    ReverseDiff.unseed!(tp)

    # forward
    a2 = rand(size(a))
    ReverseDiff.setvalue!(at, a2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == g(a2, b)
    ReverseDiff.setvalue!(at, a)

    empty!(tp)

    ########################################

    # record
    ct = g(a, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(c), length(b)))
    ReverseDiff.jacobian_reverse_pass!(out, ct, bt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> g(a, x), b) EPS
    ReverseDiff.unseed!(tp)

    # forward
    b2 = rand(size(b))
    ReverseDiff.setvalue!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == g(a, b2)
    ReverseDiff.setvalue!(bt, b)

    empty!(tp)

    ########################################

    # record
    ct = g(at, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out_a = similar(c, (length(c), length(a)))
    out_b = similar(c, (length(c), length(b)))
    ReverseDiff.jacobian_reverse_pass!(out_a, ct, at, tp)
    ReverseDiff.jacobian_reverse_pass!(out_b, ct, bt, tp)
    @test_approx_eq_eps out_a ForwardDiff.jacobian(x -> g(x, b), a) EPS
    @test_approx_eq_eps out_b ForwardDiff.jacobian(x -> g(a, x), b) EPS
    ReverseDiff.unseed!(tp)

    # forward
    a2, b2 = rand(size(a)), rand(size(b))
    ReverseDiff.setvalue!(at, a2)
    ReverseDiff.setvalue!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == g(a2, b2)
    ReverseDiff.setvalue!(at, a)
    ReverseDiff.setvalue!(bt, b)

    empty!(tp)
end

function test_broadcast(f, n::Number, x::AbstractArray, tp, builtin = false)
    nt, xt = track(n, tp), track(x, tp)

    if builtin
        g = ReverseDiff.@forward(f)
    else
        g = (x, y) -> broadcast(ReverseDiff.@forward(f), x, y)
    end

    y = g(n, x)

    ########################################

    # record
    yt = g(nt, x)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out = similar(y)
    ReverseDiff.jacobian_reverse_pass!(out, yt, [nt], tp)
    @test_approx_eq_eps out ForwardDiff.derivative(z -> g(z, x), n) EPS
    ReverseDiff.unseed!(tp)

    # forward
    n2 = rand()
    ReverseDiff.setvalue!(nt, n2)
    ReverseDiff.forward_pass!(tp)
    @test value(yt) == g(n2, x)
    ReverseDiff.setvalue!(nt, n)

    empty!(tp)

    ########################################

    # record
    yt = g(n, xt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out = similar(y, (length(y), length(x)))
    ReverseDiff.jacobian_reverse_pass!(out, yt, xt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(z -> g(n, z), x) EPS
    ReverseDiff.unseed!(tp)

    # forward
    x2 = rand(size(x))
    ReverseDiff.setvalue!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test value(yt) == g(n, x2)
    ReverseDiff.setvalue!(xt, x)

    empty!(tp)

    ########################################

    # record
    yt = g(nt, xt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out_n = similar(y)
    out_x = similar(y, (length(y), length(x)))
    ReverseDiff.jacobian_reverse_pass!(out_n, yt, [nt], tp)
    ReverseDiff.jacobian_reverse_pass!(out_x, yt, xt, tp)
    @test_approx_eq_eps out_n ForwardDiff.derivative(z -> g(z, x), n) EPS
    @test_approx_eq_eps out_x ForwardDiff.jacobian(z -> g(n, z), x) EPS
    ReverseDiff.unseed!(tp)

    # forward
    n2, x2 = rand(), rand(size(x))
    ReverseDiff.setvalue!(nt, n2)
    ReverseDiff.setvalue!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test value(yt) == g(n2, x2)
    ReverseDiff.setvalue!(nt, n)
    ReverseDiff.setvalue!(xt, x)

    empty!(tp)
end

function test_broadcast(f, x::AbstractArray, n::Number, tp, builtin = false)
    xt, nt = track(x, tp), track(n, tp)

    if builtin
        g = ReverseDiff.@forward(f)
    else
        g = (x, y) -> broadcast(ReverseDiff.@forward(f), x, y)
    end

    y = g(x, n)

    ########################################

    # record
    yt = g(x, nt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out = similar(y)
    ReverseDiff.jacobian_reverse_pass!(out, yt, [nt], tp)
    @test_approx_eq_eps out ForwardDiff.derivative(z -> g(x, z), n) EPS
    ReverseDiff.unseed!(tp)

    # forward
    n2 = rand()
    ReverseDiff.setvalue!(nt, n2)
    ReverseDiff.forward_pass!(tp)
    @test value(yt) == g(x, n2)
    ReverseDiff.setvalue!(nt, n)

    empty!(tp)

    ########################################

    # record
    yt = g(xt, n)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out = similar(y, (length(y), length(x)))
    ReverseDiff.jacobian_reverse_pass!(out, yt, xt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(z -> g(z, n), x) EPS
    ReverseDiff.unseed!(tp)

    # forward
    x2 = rand(size(x))
    ReverseDiff.setvalue!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test value(yt) == g(x2, n)
    ReverseDiff.setvalue!(xt, x)

    empty!(tp)

    ########################################

    # record
    yt = g(xt, nt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out_n = similar(y)
    out_x = similar(y, (length(y), length(x)))
    ReverseDiff.jacobian_reverse_pass!(out_n, yt, [nt], tp)
    ReverseDiff.jacobian_reverse_pass!(out_x, yt, xt, tp)
    @test_approx_eq_eps out_n ForwardDiff.derivative(z -> g(x, z), n) EPS
    @test_approx_eq_eps out_x ForwardDiff.jacobian(z -> g(z, n), x) EPS
    ReverseDiff.unseed!(tp)

    # forward
    x2, n2 = rand(size(x)), rand()
    ReverseDiff.setvalue!(xt, x2)
    ReverseDiff.setvalue!(nt, n2)
    ReverseDiff.forward_pass!(tp)
    @test value(yt) == g(x2, n2)
    ReverseDiff.setvalue!(xt, x)
    ReverseDiff.setvalue!(nt, n)

    empty!(tp)
end

for f in (sin, cos, tan, exp, x -> 1. / (1. + exp(-x)))
    testprintln("unary scalar functions", f)
    test_elementwise(f, x, tp)
    test_elementwise(f, a, tp)
end

for fsym in ReverseDiff.FORWARD_BINARY_SCALAR_FUNCS
    f = eval(fsym)
    testprintln("binary scalar functions", f)
    test_map(f, x, y, tp)
    test_map(f, a, b, tp)
    test_broadcast(f, x, y, tp)
    test_broadcast(f, a, b, tp)
    test_broadcast(f, x, a, tp)
    test_broadcast(f, a, x, tp)
    test_broadcast(f, n, x, tp)
    test_broadcast(f, x, n, tp)
    test_broadcast(f, n, a, tp)
    test_broadcast(f, a, n, tp)
end

for f in (.+, .-, .*, ./, .\, .^)
    testprintln("built-in broadcast functions", f)
    test_broadcast(f, x, y, tp, true)
    test_broadcast(f, a, b, tp, true)
    test_broadcast(f, x, a, tp, true)
    test_broadcast(f, a, x, tp, true)
    test_broadcast(f, n, x, tp, true)
    test_broadcast(f, x, n, tp, true)
    test_broadcast(f, n, a, tp, true)
    test_broadcast(f, a, n, tp, true)
end

############################################################################################

println("done (took $(toq()) seconds)")

end # module
