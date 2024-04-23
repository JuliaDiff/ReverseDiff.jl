module ElementwiseTests

using ReverseDiff

using DiffRules
using DiffTests
using ForwardDiff
using LogExpFunctions
using NaNMath
using SpecialFunctions

using Test

include(joinpath(dirname(@__FILE__), "../utils.jl"))

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
    test_approx(out, ForwardDiff.jacobian(z -> map(f, z), x); nans=true)

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
    test_approx(out, ForwardDiff.jacobian(z -> broadcast(f, z), x); nans=true)

    # forward
    x2 = x .- offset
    ReverseDiff.value!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test yt == broadcast(f, x2)
    ReverseDiff.value!(xt, x)

    return empty!(tp)
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
    out = similar(c, (length(c), length(a)))
    ReverseDiff.seeded_reverse_pass!(out, ct, at, tp)
    test_approx(out, ForwardDiff.jacobian(x -> map(f, x, b), a); nans=true)

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
    out = similar(c, (length(c), length(b)))
    ReverseDiff.seeded_reverse_pass!(out, ct, bt, tp)
    test_approx(out, ForwardDiff.jacobian(x -> map(f, a, x), b); nans=true)

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
    out_a = similar(c, (length(c), length(a)))
    out_b = similar(c, (length(c), length(b)))
    ReverseDiff.seeded_reverse_pass!(out_a, ct, at, tp)
    ReverseDiff.seeded_reverse_pass!(out_b, ct, bt, tp)
    jac = let a = a, b = b, f = f
        ForwardDiff.jacobian(vcat(vec(a), vec(b))) do x
            map(f, reshape(x[1:length(a)], size(a)), reshape(x[(length(a) + 1):end], size(b)))
        end
    end
    test_approx(out_a, jac[:, 1:length(a)]; nans=true)
    test_approx(out_b, jac[:, (length(a) + 1):end]; nans=true)
    # forward
    a2, b2 = a .- offset, b .- offset
    ReverseDiff.value!(at, a2)
    ReverseDiff.value!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test ct == map(f, a2, b2)
    ReverseDiff.value!(at, a)
    ReverseDiff.value!(bt, b)

    return empty!(tp)
end

function test_broadcast(
    f, fopt, a::AbstractArray, b::AbstractArray, tp, builtin::Bool=false
)
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
    test_approx(out, ForwardDiff.jacobian(x -> g(x, b), a); nans=true)

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
    test_approx(out, ForwardDiff.jacobian(x -> g(a, x), b); nans=true)

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
    jac = let a = a, b = b, g = g
        ForwardDiff.jacobian(vcat(vec(a), vec(b))) do x
            g(reshape(x[1:length(a)], size(a)), reshape(x[(length(a) + 1):end], size(b)))
        end
    end
    test_approx(out_a, jac[:, 1:length(a)]; nans=true)
    test_approx(out_b, jac[:, (length(a) + 1):end]; nans=true)

    # forward
    a2, b2 = a .- offset, b .- offset
    ReverseDiff.value!(at, a2)
    ReverseDiff.value!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test ct == g(a2, b2)
    ReverseDiff.value!(at, a)
    ReverseDiff.value!(bt, b)

    return empty!(tp)
end

function test_broadcast(f, fopt, n::Number, x::AbstractArray, tp, builtin::Bool=false)
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
    test_approx(out, ForwardDiff.derivative(z -> g(z, x), n); nans=true)

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
    test_approx(out, ForwardDiff.jacobian(z -> g(n, z), x); nans=true)

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
    jac = let x = x, g = g
        ForwardDiff.jacobian(z -> g(z[1], reshape(z[2:end], size(x))), vcat(n, vec(x)))
    end
    test_approx(out_n, reshape(jac[:, 1], size(y)); nans=true)
    test_approx(out_x, jac[:, 2:end]; nans=true)

    # forward
    n2, x2 = n + offset, x .- offset
    ReverseDiff.value!(nt, n2)
    ReverseDiff.value!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test yt == g(n2, x2)
    ReverseDiff.value!(nt, n)
    ReverseDiff.value!(xt, x)

    return empty!(tp)
end

function test_broadcast(f, fopt, x::AbstractArray, n::Number, tp, builtin::Bool=false)
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
    test_approx(out, ForwardDiff.derivative(z -> g(x, z), n); nans=true)

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
    test_approx(out, ForwardDiff.jacobian(z -> g(z, n), x); nans=true)

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
    jac = let x = x, g = g
        ForwardDiff.jacobian(z -> g(reshape(z[1:(end - 1)], size(x)), z[end]), vcat(vec(x), n))
    end
    test_approx(out_x, jac[:, 1:(end - 1)]; nans=true)
    test_approx(out_n, reshape(jac[:, end], size(y)); nans=true)

    # forward
    x2, n2 = x .- offset, n + offset
    ReverseDiff.value!(xt, x2)
    ReverseDiff.value!(nt, n2)
    ReverseDiff.forward_pass!(tp)
    @test yt == g(x2, n2)
    ReverseDiff.value!(xt, x)
    ReverseDiff.value!(nt, n)

    return empty!(tp)
end

for f in DiffTests.NUMBER_TO_NUMBER_FUNCS
    test_println("DiffTests.NUMBER_TO_NUMBER_FUNCS", f)
    test_elementwise(f, ReverseDiff.@forward(f), x, tp)
    test_elementwise(f, ReverseDiff.@forward(f), a, tp)
end

for (M, fsym, arity) in DiffRules.diffrules(; filter_modules=nothing)
    # ensure that all rules can be tested
    if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), fsym))
        error("$M.$fsym is not available")
    end
    (M, fsym) in ReverseDiff.SKIPPED_DIFFRULES && continue
    if arity == 1
        f = eval(:($M.$fsym))
        test_println("forward-mode unary scalar functions", f)
        test_elementwise(f, f, modify_input(fsym, x), tp)
        test_elementwise(f, f, modify_input(fsym, a), tp)
    elseif arity == 2
        in(fsym, SKIPPED_BINARY_SCALAR_TESTS) && continue
        f = eval(:($M.$fsym))
        test_println("forward-mode binary scalar functions", f)
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
end

for f in DiffTests.BINARY_BROADCAST_OPS
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

end # module
