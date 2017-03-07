module ElementwiseTests

using ReverseDiff, ForwardDiff, Base.Test

include(joinpath(dirname(@__FILE__), "../utils.jl"))

println("testing elementwise derivatives (both forward and reverse passes)")
tic()

############################################################################################
x, y = rand(3, 3), rand(3, 3)
a, b = rand(3), rand(3)
n = rand()
tp = InstructionTape()
offset = 0.00001

function test_elementwise(f, fopt, x, tp)
    xt = track(copy(x), tp)

    ########################################

    y = map(f, x)

    # record
    yt = map(fopt, xt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out = similar(y, (length(x), length(x)))
    ReverseDiff.seeded_reverse_pass!(out, yt, xt, tp)
    test_approx(out, ForwardDiff.jacobian(z -> map(f, z), x))

    # forward
    x2 = x .- offset
    ReverseDiff.value!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test yt == map(f, x2)
    ReverseDiff.value!(xt, x)

    empty!(tp)

    ########################################

    y = broadcast(f, x)

    # record
    yt = broadcast(fopt, xt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out = similar(y, (length(x), length(x)))
    ReverseDiff.seeded_reverse_pass!(out, yt, xt, tp)
    test_approx(out, ForwardDiff.jacobian(z -> broadcast(f, z), x))

    # forward
    x2 = x .- offset
    ReverseDiff.value!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test yt == broadcast(f, x2)
    ReverseDiff.value!(xt, x)

    empty!(tp)
end

function test_map(f, fopt, a, b, tp)
    at, bt = track(copy(a), tp), track(copy(b), tp)
    c = map(f, a, b)

    ########################################

    # record
    ct = map(fopt, at, b)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(a), length(a)))
    ReverseDiff.seeded_reverse_pass!(out, ct, at, tp)
    test_approx(out, ForwardDiff.jacobian(x -> map(f, x, b), a))

    # forward
    a2 = a .- offset
    ReverseDiff.value!(at, a2)
    ReverseDiff.forward_pass!(tp)
    @test ct == map(f, a2, b)
    ReverseDiff.value!(at, a)

    empty!(tp)

    ########################################

    # record
    ct = map(fopt, a, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(a), length(a)))
    ReverseDiff.seeded_reverse_pass!(out, ct, bt, tp)
    test_approx(out, ForwardDiff.jacobian(x -> map(f, a, x), b))

    # forward
    b2 = b .- offset
    ReverseDiff.value!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test ct == map(f, a, b2)
    ReverseDiff.value!(bt, b)

    empty!(tp)

    ########################################

    # record
    ct = map(fopt, at, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out_a = similar(c, (length(a), length(a)))
    out_b = similar(c, (length(a), length(a)))
    ReverseDiff.seeded_reverse_pass!(out_a, ct, at, tp)
    ReverseDiff.seeded_reverse_pass!(out_b, ct, bt, tp)
    test_approx(out_a, ForwardDiff.jacobian(x -> map(f, x, b), a))
    test_approx(out_b, ForwardDiff.jacobian(x -> map(f, a, x), b))

    # forward
    a2, b2 = a .- offset, b .- offset
    ReverseDiff.value!(at, a2)
    ReverseDiff.value!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test ct == map(f, a2, b2)
    ReverseDiff.value!(at, a)
    ReverseDiff.value!(bt, b)

    empty!(tp)
end

function test_broadcast(f, fopt, a::AbstractArray, b::AbstractArray, tp, builtin::Bool = false)
    at, bt = track(copy(a), tp), track(copy(b), tp)

    if builtin
        g = fopt
    else
        g = (x, y) -> broadcast(fopt, x, y)
    end

    c = g(a, b)

    ########################################

    # record
    ct = g(at, b)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(c), length(a)))
    ReverseDiff.seeded_reverse_pass!(out, ct, at, tp)
    test_approx(out, ForwardDiff.jacobian(x -> g(x, b), a))

    # forward
    a2 = a .- offset
    ReverseDiff.value!(at, a2)
    ReverseDiff.forward_pass!(tp)
    @test ct == g(a2, b)
    ReverseDiff.value!(at, a)

    empty!(tp)

    ########################################

    # record
    ct = g(a, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out = similar(c, (length(c), length(b)))
    ReverseDiff.seeded_reverse_pass!(out, ct, bt, tp)
    test_approx(out, ForwardDiff.jacobian(x -> g(a, x), b))

    # forward
    b2 = b .- offset
    ReverseDiff.value!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test ct == g(a, b2)
    ReverseDiff.value!(bt, b)

    empty!(tp)

    ########################################

    # record
    ct = g(at, bt)
    @test ct == c
    @test length(tp) == 1

    # reverse
    out_a = similar(c, (length(c), length(a)))
    out_b = similar(c, (length(c), length(b)))
    ReverseDiff.seeded_reverse_pass!(out_a, ct, at, tp)
    ReverseDiff.seeded_reverse_pass!(out_b, ct, bt, tp)
    test_approx(out_a, ForwardDiff.jacobian(x -> g(x, b), a))
    test_approx(out_b, ForwardDiff.jacobian(x -> g(a, x), b))

    # forward
    a2, b2 = a .- offset, b .- offset
    ReverseDiff.value!(at, a2)
    ReverseDiff.value!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test ct == g(a2, b2)
    ReverseDiff.value!(at, a)
    ReverseDiff.value!(bt, b)

    empty!(tp)
end

function test_broadcast(f, fopt, n::Number, x::AbstractArray, tp, builtin::Bool = false)
    nt, xt = track(copy(n), tp), track(copy(x), tp)

    if builtin
        g = fopt
    else
        g = (x, y) -> broadcast(fopt, x, y)
    end

    y = g(n, x)

    ########################################

    # record
    yt = g(nt, x)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out = similar(y)
    ReverseDiff.seeded_reverse_pass!(out, yt, nt, tp)
    test_approx(out, ForwardDiff.derivative(z -> g(z, x), n))

    # forward
    n2 = n + offset
    ReverseDiff.value!(nt, n2)
    ReverseDiff.forward_pass!(tp)
    @test yt == g(n2, x)
    ReverseDiff.value!(nt, n)

    empty!(tp)

    ########################################

    # record
    yt = g(n, xt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out = similar(y, (length(y), length(x)))
    ReverseDiff.seeded_reverse_pass!(out, yt, xt, tp)
    test_approx(out, ForwardDiff.jacobian(z -> g(n, z), x))

    # forward
    x2 = x .- offset
    ReverseDiff.value!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test yt == g(n, x2)
    ReverseDiff.value!(xt, x)

    empty!(tp)

    ########################################

    # record
    yt = g(nt, xt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out_n = similar(y)
    out_x = similar(y, (length(y), length(x)))
    ReverseDiff.seeded_reverse_pass!(out_n, yt, nt, tp)
    ReverseDiff.seeded_reverse_pass!(out_x, yt, xt, tp)
    test_approx(out_n, ForwardDiff.derivative(z -> g(z, x), n))
    test_approx(out_x, ForwardDiff.jacobian(z -> g(n, z), x))

    # forward
    n2, x2 = n + offset , x .- offset
    ReverseDiff.value!(nt, n2)
    ReverseDiff.value!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test yt == g(n2, x2)
    ReverseDiff.value!(nt, n)
    ReverseDiff.value!(xt, x)

    empty!(tp)
end

function test_broadcast(f, fopt, x::AbstractArray, n::Number, tp, builtin::Bool = false)
    xt, nt = track(copy(x), tp), track(copy(n), tp)

    if builtin
        g = fopt
    else
        g = (x, y) -> broadcast(fopt, x, y)
    end

    y = g(x, n)

    ########################################

    # record
    yt = g(x, nt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out = similar(y)
    ReverseDiff.seeded_reverse_pass!(out, yt, nt, tp)
    test_approx(out, ForwardDiff.derivative(z -> g(x, z), n))

    # forward
    n2 = n + offset
    ReverseDiff.value!(nt, n2)
    ReverseDiff.forward_pass!(tp)
    @test yt == g(x, n2)
    ReverseDiff.value!(nt, n)

    empty!(tp)

    ########################################

    # record
    yt = g(xt, n)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out = similar(y, (length(y), length(x)))
    ReverseDiff.seeded_reverse_pass!(out, yt, xt, tp)
    test_approx(out, ForwardDiff.jacobian(z -> g(z, n), x))

    # forward
    x2 = x .- offset
    ReverseDiff.value!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test yt == g(x2, n)
    ReverseDiff.value!(xt, x)

    empty!(tp)

    ########################################

    # record
    yt = g(xt, nt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    out_n = similar(y)
    out_x = similar(y, (length(y), length(x)))
    ReverseDiff.seeded_reverse_pass!(out_n, yt, nt, tp)
    ReverseDiff.seeded_reverse_pass!(out_x, yt, xt, tp)
    test_approx(out_n, ForwardDiff.derivative(z -> g(x, z), n))
    test_approx(out_x, ForwardDiff.jacobian(z -> g(z, n), x))

    # forward
    x2, n2 = x .- offset, n + offset
    ReverseDiff.value!(xt, x2)
    ReverseDiff.value!(nt, n2)
    ReverseDiff.forward_pass!(tp)
    @test yt == g(x2, n2)
    ReverseDiff.value!(xt, x)
    ReverseDiff.value!(nt, n)

    empty!(tp)
end

for f in DiffBase.NUMBER_TO_NUMBER_FUNCS
    test_println("DiffBase.NUMBER_TO_NUMBER_FUNCS", f)
    test_elementwise(f, ReverseDiff.@forward(f), x, tp)
    test_elementwise(f, ReverseDiff.@forward(f), a, tp)
end

DOMAIN_ERR_FUNCS = (:asec, :acsc, :asecd, :acscd, :acoth, :acosh)

for fsym in ReverseDiff.FORWARD_UNARY_SCALAR_FUNCS
    f = eval(fsym)
    is_domain_err_func = in(fsym, DOMAIN_ERR_FUNCS)
    test_println("FORWARD_UNARY_SCALAR_FUNCS", f)
    test_elementwise(f, f, is_domain_err_func ? x .+ 1 : x, tp)
    test_elementwise(f, f, is_domain_err_func ? a .+ 1 : a, tp)
end

for fsym in ReverseDiff.FORWARD_BINARY_SCALAR_FUNCS
    f = eval(fsym)
    test_println("FORWARD_BINARY_SCALAR_FUNCS", f)
    test_map(f, f, x, y, tp)
    test_map(f, f, a, b, tp)
    test_broadcast(f, f, x, y, tp)
    test_broadcast(f, f, a, b, tp)
    test_broadcast(f, f, x, a, tp)
    test_broadcast(f, f, a, x, tp)
    test_broadcast(f, f, n, x, tp)
    test_broadcast(f, f, x, n, tp)
    test_broadcast(f, f, n, a, tp)
    test_broadcast(f, f, a, n, tp)
end

for f in DiffBase.BINARY_BROADCAST_OPS
    test_println("built-in broadcast operators", f)
    test_broadcast(f, f, x, y, tp, true)
    test_broadcast(f, f, a, b, tp, true)
    test_broadcast(f, f, x, a, tp, true)
    test_broadcast(f, f, a, x, tp, true)
    test_broadcast(f, f, n, x, tp, true)
    test_broadcast(f, f, x, n, tp, true)
    test_broadcast(f, f, n, a, tp, true)
    test_broadcast(f, f, a, n, tp, true)
end

############################################################################################

println("done (took $(toq()) seconds)")

end # module
