# module ElementwiseTests

using ReverseDiffPrototype, ForwardDiff, Base.Test

include("../utils.jl")

println("testing elementwise derivatives (both forward and reverse passes)")
tic()

############################################################################################
x, a, b, n = rand(3, 3), rand(3, 3), rand(3, 3), rand()
tp = Tape()

function test_elementwise(f, x, tp)
    xt = track(x, tp)
    y = map(f, x)

    out = similar(y, (length(x), length(x)))
    yt = map(RDP.@forward(f), xt)
    @test yt == y
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out, yt, xt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(z -> map(f, z), x) EPS
    empty!(tp)

    y = broadcast(RDP.@forward(f), x)
    out = similar(y, (length(x), length(x)))
    yt = broadcast(RDP.@forward(f), xt)
    @test yt == y
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out, yt, xt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(z -> broadcast(f, z), x) EPS
    empty!(tp)
end

function test_map(f, a, b, tp)
    at, bt = track(a, tp), track(b, tp)
    c = map(f, a, b)

    out = similar(c, (length(a), length(a)))
    ct = map(RDP.@forward(f), at, b)
    @test ct == c
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out, ct, at, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> map(f, x, b), a) EPS
    RDP.unseed!(tp)
    empty!(tp)

    out = similar(c, (length(a), length(a)))
    ct = map(RDP.@forward(f), a, bt)
    @test ct == c
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out, ct, bt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> map(f, a, x), b) EPS
    RDP.unseed!(tp)
    empty!(tp)

    out_a = similar(c, (length(a), length(a)))
    out_b = similar(c, (length(a), length(a)))
    ct = map(RDP.@forward(f), at, bt)
    @test ct == c
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out_a, ct, at, tp)
    RDP.jacobian_reverse_pass!(out_b, ct, bt, tp)
    @test_approx_eq_eps out_a ForwardDiff.jacobian(x -> map(f, x, b), a) EPS
    @test_approx_eq_eps out_b ForwardDiff.jacobian(x -> map(f, a, x), b) EPS
    RDP.unseed!(tp)
    empty!(tp)
end

function test_broadcast(f, a::AbstractArray, b::AbstractArray, tp, builtin = false)
    at, bt = track(a, tp), track(b, tp)

    if builtin
        g = RDP.@forward(f)
    else
        g = (x, y) -> broadcast(RDP.@forward(f), x, y)
    end

    c = g(a, b)

    out = similar(c, (length(c), length(a)))
    ct = g(at, b)
    @test ct == c
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out, ct, at, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> g(x, b), a) EPS
    RDP.unseed!(tp)
    empty!(tp)

    out = similar(c, (length(c), length(b)))
    ct = g(a, bt)
    @test ct == c
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out, ct, bt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(x -> g(a, x), b) EPS
    RDP.unseed!(tp)
    empty!(tp)

    out_a = similar(c, (length(c), length(a)))
    out_b = similar(c, (length(c), length(b)))
    ct = g(at, bt)
    @test ct == c
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out_a, ct, at, tp)
    RDP.jacobian_reverse_pass!(out_b, ct, bt, tp)
    @test_approx_eq_eps out_a ForwardDiff.jacobian(x -> g(x, b), a) EPS
    @test_approx_eq_eps out_b ForwardDiff.jacobian(x -> g(a, x), b) EPS
    RDP.unseed!(tp)
    empty!(tp)
end

function test_broadcast(f, n::Number, x::AbstractArray, tp, builtin = false)
    nt, xt = track(n, tp), track(x, tp)

    if builtin
        g = RDP.@forward(f)
    else
        g = (x, y) -> broadcast(RDP.@forward(f), x, y)
    end

    y = g(n, x)

    out = similar(y)
    yt = g(nt, x)
    @test yt == y
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out, yt, [nt], tp)
    @test_approx_eq_eps out ForwardDiff.derivative(z -> g(z, x), n) EPS
    RDP.unseed!(tp)
    empty!(tp)

    out = similar(y, (length(y), length(x)))
    yt = g(n, xt)
    @test yt == y
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out, yt, xt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(z -> g(n, z), x) EPS
    RDP.unseed!(tp)
    empty!(tp)

    out_n = similar(y)
    out_x = similar(y, (length(y), length(x)))
    yt = g(nt, xt)
    @test yt == y
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out_n, yt, [nt], tp)
    RDP.jacobian_reverse_pass!(out_x, yt, xt, tp)
    @test_approx_eq_eps out_n ForwardDiff.derivative(z -> g(z, x), n) EPS
    @test_approx_eq_eps out_x ForwardDiff.jacobian(z -> g(n, z), x) EPS
    RDP.unseed!(tp)
    empty!(tp)
end

function test_broadcast(f, x::AbstractArray, n::Number, tp, builtin = false)
    xt, nt = track(x, tp), track(n, tp)

    if builtin
        g = RDP.@forward(f)
    else
        g = (x, y) -> broadcast(RDP.@forward(f), x, y)
    end

    y = g(x, n)

    out = similar(y)
    yt = g(x, nt)
    @test yt == y
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out, yt, [nt], tp)
    @test_approx_eq_eps out ForwardDiff.derivative(z -> g(x, z), n) EPS
    RDP.unseed!(tp)
    empty!(tp)

    out = similar(y, (length(y), length(x)))
    yt = g(xt, n)
    @test yt == y
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out, yt, xt, tp)
    @test_approx_eq_eps out ForwardDiff.jacobian(z -> g(z, n), x) EPS
    RDP.unseed!(tp)
    empty!(tp)

    out_n = similar(y)
    out_x = similar(y, (length(y), length(x)))
    yt = g(xt, nt)
    @test yt == y
    @test length(tp) == 1
    RDP.jacobian_reverse_pass!(out_n, yt, [nt], tp)
    RDP.jacobian_reverse_pass!(out_x, yt, xt, tp)
    @test_approx_eq_eps out_n ForwardDiff.derivative(z -> g(x, z), n) EPS
    @test_approx_eq_eps out_x ForwardDiff.jacobian(z -> g(z, n), x) EPS
    RDP.unseed!(tp)
    empty!(tp)
end

for f in (sin, cos, tan, exp, x -> 1. / (1. + exp(-x)))
    testprintln("unary scalar functions", f)
    test_elementwise(f, x, tp)
end

for fsym in RDP.FORWARD_BINARY_SCALAR_FUNCS
    f = eval(fsym)
    testprintln("binary scalar functions", f)
    test_map(f, a, b, tp)
    test_broadcast(f, a, b, tp)
    test_broadcast(f, n, x, tp)
    test_broadcast(f, x, n, tp)
end

for f in (.+, .-, .*, ./, .\, .^)
    testprintln("built-in broadcast functions", f)
    test_broadcast(f, a, b, tp, true)
    test_broadcast(f, n, x, tp, true)
    test_broadcast(f, x, n, tp, true)
end

############################################################################################

println("done (took $(toq()) seconds)")

# end # module
