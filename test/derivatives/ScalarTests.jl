module ScalarTests

using ReverseDiff, ForwardDiff, Test, DiffRules, SpecialFunctions, NaNMath

include(joinpath(dirname(@__FILE__), "../utils.jl"))

x, a, b = rand(3)
tp = InstructionTape()
int_range = 1:10

function test_forward(f, x, tp::InstructionTape, is_domain_err_func::Bool)
    xt = ReverseDiff.TrackedReal(x, zero(x), tp)
    y = f(x)

    # record
    yt = f(xt)
    @test yt == y
    @test length(tp) == 1

    # reverse
    ReverseDiff.seed!(yt)
    ReverseDiff.reverse_pass!(tp)
    @test deriv(xt) == ForwardDiff.derivative(f, x)

    # forward
    x2 = is_domain_err_func ? rand() + 1 : rand()
    ReverseDiff.value!(xt, x2)
    ReverseDiff.forward_pass!(tp)
    @test value(yt) == f(x2)
    ReverseDiff.value!(xt, x)

    empty!(tp)
end

function test_forward(f, a, b, tp)
    at = ReverseDiff.TrackedReal(a, 0.0, tp)
    bt = ReverseDiff.TrackedReal(b, 0.0, tp)
    c = f(a, b)

    ########################################

    # record
    ct = f(at, b)
    @test ct == c
    @test length(tp) == 1

    # reverse
    ReverseDiff.seed!(ct)
    ReverseDiff.reverse_pass!(tp)
    test_approx(deriv(at), ForwardDiff.derivative(x -> f(x, b), a))
    ReverseDiff.unseed!(at)

    # forward
    a2 = isa(a, Int) ? rand(int_range) : rand()
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
    ReverseDiff.seed!(ct)
    ReverseDiff.reverse_pass!(tp)
    test_approx(deriv(bt), ForwardDiff.derivative(x -> f(a, x), b))
    ReverseDiff.unseed!(bt)

    # forward
    b2 = isa(b, Int) ? rand(int_range) : rand()
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
    ReverseDiff.seed!(ct)
    ReverseDiff.reverse_pass!(tp)
    test_approx(deriv(at), ForwardDiff.derivative(x -> f(x, b), a))
    test_approx(deriv(bt), ForwardDiff.derivative(x -> f(a, x), b))

    # forward
    a2 = isa(a, Int) ? rand(int_range) : rand()
    b2 = isa(b, Int) ? rand(int_range) : rand()
    ReverseDiff.value!(at, a2)
    ReverseDiff.value!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == f(a2, b2)
    ReverseDiff.value!(bt, a)
    ReverseDiff.value!(bt, b)

    empty!(tp)
end

function test_skip(f, x, tp)
    xt = ReverseDiff.TrackedReal(x, zero(x), tp)
    y = f(x)
    yt = f(xt)
    @test yt == y
    @test isempty(tp)
end

function test_skip(f, a, b, tp)
    at = ReverseDiff.TrackedReal(a, zero(a), tp)
    bt = ReverseDiff.TrackedReal(b, zero(b), tp)
    c = f(a, b)

    ct = f(at, b)
    @test ct == c
    @test isempty(tp)

    ct = f(a, bt)
    @test ct == c
    @test isempty(tp)

    ct = f(at, bt)
    @test ct == c
    @test isempty(tp)
end

DOMAIN_ERR_FUNCS = (:asec, :acsc, :asecd, :acscd, :acoth, :acosh)

for (M, f, arity) in DiffRules.diffrules()
    f === :rem2pi && continue
    if arity == 1
        test_println("forward-mode unary scalar functions", string(M, ".", f))
        is_domain_err_func = in(f, DOMAIN_ERR_FUNCS)
        n = is_domain_err_func ? x + 1 : x
        test_forward(eval(:($M.$f)), n, tp, is_domain_err_func)
    elseif arity == 2
        in(f, SKIPPED_BINARY_SCALAR_TESTS) && continue
        test_println("forward-mode binary scalar functions", f)
        test_forward(eval(:($M.$f)), a, b, tp)
    end
end

INT_ONLY_FUNCS = (:iseven, :isodd)

for f in ReverseDiff.SKIPPED_UNARY_SCALAR_FUNCS
    test_println("SKIPPED_UNARY_SCALAR_FUNCS", f)
    n = in(f, DOMAIN_ERR_FUNCS) ? x + 1 : x
    n = in(f, INT_ONLY_FUNCS) ? ceil(Int, n) : n
    test_skip(eval(f), n, tp)
end

for f in ReverseDiff.SKIPPED_BINARY_SCALAR_FUNCS
    test_println("SKIPPED_BINARY_SCALAR_FUNCS", f)
    test_skip(eval(f), a, b, tp)
end

end # module
