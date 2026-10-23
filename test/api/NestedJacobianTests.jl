using DiffTests, ForwardDiff, ReverseDiff, Test

include(joinpath(dirname(@__FILE__), "../utils.jl"))

for f in (DiffTests.ARRAY_TO_ARRAY_FUNCS..., DiffTests.MATRIX_TO_MATRIX_FUNCS...)
    test_println("ARRAY_TO_ARRAY_FUNCS + MATRIX_TO_MATRIX_FUNCS", f)

    x = rand(5, 5)
    test = ForwardDiff.jacobian(y -> ForwardDiff.jacobian(f, y), x)

    # without JacobianTape

    J = ReverseDiff.jacobian(y -> ReverseDiff.jacobian(f, y), x)
    test_approx(J, test)

    # with JacobianTape

    tp = ReverseDiff.JacobianTape(y -> ReverseDiff.jacobian(f, y), rand(eltype(x), size(x)))
    J = ReverseDiff.jacobian!(tp, x)
    test_approx(J, test)
end

for f in DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS
    test_println("BINARY_MATRIX_TO_MATRIX_FUNCS", f)

    a, b = rand(5, 5), rand(5, 5)

    test_val = f(a, b)
    test_a = ForwardDiff.jacobian(y -> ForwardDiff.jacobian(x -> f(x, b), y), a)
    test_b = ForwardDiff.jacobian(y -> ForwardDiff.jacobian(x -> f(a, x), y), b)

    # without JacobianTape

    Ja = ReverseDiff.jacobian(y -> ReverseDiff.jacobian(x -> f(x, b), y), a)
    Jb = ReverseDiff.jacobian(y -> ReverseDiff.jacobian(x -> f(a, x), y), b)
    test_approx(Ja, test_a)
    test_approx(Jb, test_b)

    # with JacobianTape

    ra = ReverseDiff.JacobianTape(y -> ReverseDiff.jacobian(x -> f(x, b), y), rand(eltype(a), size(a)))
    rb = ReverseDiff.JacobianTape(y -> ReverseDiff.jacobian(x -> f(a, x), y), rand(eltype(b), size(b)))
    Ja = ReverseDiff.jacobian!(ra, a)
    Jb = ReverseDiff.jacobian!(rb, b)
    test_approx(Ja, test_a)
    test_approx(Jb, test_b)

    # The below will fail until support for the Jacobian of
    # functions with multiple output arrays is implemented

    # Ja, Jb = ReverseDiff.jacobian((x, y) -> ReverseDiff.jacobian(f, (x, y)), (a, b))
    # test_approx(Ja test_a)
    # test_approx(Jb test_b)
end
