module LinAlgTests

using ReverseDiff, ForwardDiff, Test, LinearAlgebra, Statistics

include(joinpath(dirname(@__FILE__), "../utils.jl"))

x, a, b = rand(3, 3), rand(3, 3), rand(3, 3)
tp = InstructionTape()

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
    test_approx(deriv(xt), ForwardDiff.gradient(f, x))

    # forward
    x2 = rand(eltype(x), size(x))
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
    test_approx(out, ForwardDiff.jacobian(f, x))

    # forward
    x2 = rand(eltype(x), size(x))
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
    test_approx(out, ForwardDiff.jacobian(x -> f(x, b), a))

    # forward
    a2 = rand(eltype(a), size(a))
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
    test_approx(out, ForwardDiff.jacobian(x -> f(a, x), b))

    # forward
    b2 = rand(eltype(b), size(b))
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
    test_approx(out_a, ForwardDiff.jacobian(x -> f(x, b), a))
    test_approx(out_b, ForwardDiff.jacobian(x -> f(a, x), b))

    # forward
    a2, b2 = rand(eltype(a), size(a)), rand(eltype(b), size(b))
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
    test_approx(out, ForwardDiff.jacobian(x -> f(x, b), a))

    # forward
    a2 = rand(eltype(a), size(a))
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
    test_approx(out, ForwardDiff.jacobian(x -> f(a, x), b))

    # forward
    b2 = rand(eltype(b), size(b))
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
    test_approx(out_a, ForwardDiff.jacobian(x -> f(x, b), a))
    test_approx(out_b, ForwardDiff.jacobian(x -> f(a, x), b))

    # forward
    a2, b2 = rand(eltype(a), size(a)), rand(eltype(b), size(b))
    ReverseDiff.value!(at, a2)
    ReverseDiff.value!(bt, b2)
    ReverseDiff.forward_pass!(tp)
    @test value(ct) == f(a2, b2)
    ReverseDiff.value!(at, a)
    ReverseDiff.value!(bt, b)

    empty!(tp)
end

for f in (sum, det, y -> dot(vec(y), vec(y)), mean)
    test_println("Array -> Number functions", f)
    test_arr2num(f, x, tp)
end

for f in (-, inv)
    test_println("Array -> Array functions", f)
    test_arr2arr(f, x, tp)
end

for f in (+, -)
    test_println("(Array, Array) -> Array functions", f)
    test_arr2arr(f, a, b, tp)
end

test_println("*(A, B) functions", "*(a, b)")

test_arr2arr(*, a, b, tp)
test_arr2arr_inplace(mul!, *, x, a, b, tp)

for f in (transpose, adjoint)
    test_println("*(A, B) functions", string("*(", f, "(a), b)"))
    test_arr2arr(*, f(a), b, tp)
    test_arr2arr_inplace(mul!, *, x, f(a), b, tp)
    test_println("*(A, B) functions", string("*(a, ", f, "(b))"))
    test_arr2arr(*, a, f(b), tp)
    test_arr2arr_inplace(mul!, *, x, a, f(b), tp)
    test_println("*(A, B) functions", string("*(", f, "(a), ", f, "(b))"))
    test_arr2arr(*, f(a), f(b), tp)
    test_arr2arr_inplace(mul!, *, x, f(a), f(b), tp)
end

test_println("*(A, B) functions", "*(adjoint(a), transpose(b))")
test_arr2arr(*, adjoint(a), transpose(b), tp)
test_arr2arr_inplace(mul!, *, x, adjoint(a), transpose(b), tp)

test_println("*(A, B) functions", "*(transpose(a), adjoint(b))")
test_arr2arr(*, transpose(a), adjoint(b), tp)
test_arr2arr_inplace(mul!, *, x, transpose(a), adjoint(b), tp)

end # module
