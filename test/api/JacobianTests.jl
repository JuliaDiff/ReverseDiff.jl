module JacobianTests

using DiffTests, ForwardDiff, ReverseDiff, Base.Test

include(joinpath(dirname(@__FILE__), "../utils.jl"))

println("testing jacobian/jacobian!...")
tic()

############################################################################################

function test_unary_jacobian(f, x)
    test_val = f(x)
    test = ForwardDiff.jacobian!(DiffResults.JacobianResult(test_val, x), f, x, ForwardDiff.JacobianConfig(f, x))

    # without JacobianConfig

    test_approx(ReverseDiff.jacobian(f, x), DiffResults.jacobian(test))

    out = similar(DiffResults.jacobian(test))
    ReverseDiff.jacobian!(out, f, x)
    test_approx(out, DiffResults.jacobian(test))

    result = DiffResults.JacobianResult(test_val, x)
    ReverseDiff.jacobian!(result, f, x)
    test_approx(DiffResults.value(result), DiffResults.value(test))
    test_approx(DiffResults.jacobian(result), DiffResults.jacobian(test))

    # with JacobianConfig

    cfg = ReverseDiff.JacobianConfig(x)

    test_approx(ReverseDiff.jacobian(f, x, cfg), DiffResults.jacobian(test))

    out = similar(DiffResults.jacobian(test))
    ReverseDiff.jacobian!(out, f, x, cfg)
    test_approx(out, DiffResults.jacobian(test))

    result = DiffResults.JacobianResult(test_val, x)
    ReverseDiff.jacobian!(result, f, x, cfg)
    test_approx(DiffResults.value(result), DiffResults.value(test))
    test_approx(DiffResults.jacobian(result), DiffResults.jacobian(test))

    # with JacobianTape

    tp = ReverseDiff.JacobianTape(f, rand(size(x)))

    test_approx(ReverseDiff.jacobian!(tp, x), DiffResults.jacobian(test))

    out = similar(DiffResults.jacobian(test))
    ReverseDiff.jacobian!(out, tp, x)
    test_approx(out, DiffResults.jacobian(test))

    result = DiffResults.JacobianResult(test_val, x)
    ReverseDiff.jacobian!(result, tp, x)
    test_approx(DiffResults.value(result), DiffResults.value(test))
    test_approx(DiffResults.jacobian(result), DiffResults.jacobian(test))

    # with compiled JacobianTape

    if length(tp.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        ctp = ReverseDiff.compile(tp)

        test_approx(ReverseDiff.jacobian!(ctp, x), DiffResults.jacobian(test))

        out = similar(DiffResults.jacobian(test))
        ReverseDiff.jacobian!(out, ctp, x)
        test_approx(out, DiffResults.jacobian(test))

        result = DiffResults.JacobianResult(test_val, x)
        ReverseDiff.jacobian!(result, ctp, x)
        test_approx(DiffResults.value(result), DiffResults.value(test))
        test_approx(DiffResults.jacobian(result), DiffResults.jacobian(test))
    end
end

function test_unary_jacobian(f!, y, x)
    y_original = copy(y)
    y_copy = copy(y)
    test = ForwardDiff.jacobian!(DiffResults.JacobianResult(y_copy, x), f!, y_copy, x)

    # without JacobianConfig

    out = ReverseDiff.jacobian(f!, y, x)
    test_approx(y, DiffResults.value(test))
    test_approx(out, DiffResults.jacobian(test))
    copy!(y, y_original)

    out = similar(DiffResults.jacobian(test))
    ReverseDiff.jacobian!(out, f!, y, x)
    test_approx(y,   DiffResults.value(test))
    test_approx(out, DiffResults.jacobian(test))
    copy!(y, y_original)

    result = DiffResults.JacobianResult(y, x)
    ReverseDiff.jacobian!(result, f!, y, x)
    @test DiffResults.value(result) == y
    test_approx(y, DiffResults.value(test))
    test_approx(DiffResults.jacobian(result), DiffResults.jacobian(test))
    copy!(y, y_original)

    # with JacobianConfig

    cfg = ReverseDiff.JacobianConfig(y, x)

    out = ReverseDiff.jacobian(f!, y, x, cfg)
    test_approx(y,   DiffResults.value(test))
    test_approx(out, DiffResults.jacobian(test))
    copy!(y, y_original)

    out = similar(DiffResults.jacobian(test))
    ReverseDiff.jacobian!(out, f!, y, x, cfg)
    test_approx(y,   DiffResults.value(test))
    test_approx(out, DiffResults.jacobian(test))
    copy!(y, y_original)

    result = DiffResults.JacobianResult(y, x)
    ReverseDiff.jacobian!(result, f!, y, x, cfg)
    @test DiffResults.value(result) == y
    test_approx(y, DiffResults.value(test))
    test_approx(DiffResults.jacobian(result), DiffResults.jacobian(test))
    copy!(y, y_original)

    # with JacobianTape

    tp = ReverseDiff.JacobianTape(f!, y, rand(size(x)))

    out = ReverseDiff.jacobian!(tp, x)
    test_approx(out, DiffResults.jacobian(test))

    out = similar(DiffResults.jacobian(test))
    ReverseDiff.jacobian!(out, tp, x)
    test_approx(out, DiffResults.jacobian(test))

    result = DiffResults.JacobianResult(y, x)
    ReverseDiff.jacobian!(result, tp, x)
    test_approx(DiffResults.value(result), DiffResults.value(test))
    test_approx(DiffResults.jacobian(result), DiffResults.jacobian(test))

    # with compiled JacobianTape

    if length(tp.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        ctp = ReverseDiff.compile(tp)

        out = ReverseDiff.jacobian!(ctp, x)

        test_approx(out, DiffResults.jacobian(test))

        out = similar(DiffResults.jacobian(test))
        ReverseDiff.jacobian!(out, ctp, x)
        test_approx(out, DiffResults.jacobian(test))

        result = DiffResults.JacobianResult(y, x)
        ReverseDiff.jacobian!(result, ctp, x)
        test_approx(DiffResults.value(result), DiffResults.value(test))
        test_approx(DiffResults.jacobian(result), DiffResults.jacobian(test))
    end
end

function test_binary_jacobian(f, a, b)
    test_val = f(a, b)
    test_a = ForwardDiff.jacobian(x -> f(x, b), a)
    test_b = ForwardDiff.jacobian(x -> f(a, x), b)

    # without JacobianConfig

    Ja, Jb = ReverseDiff.jacobian(f, (a, b))
    test_approx(Ja, test_a)
    test_approx(Jb, test_b)

    Ja = similar(a, length(test_val), length(a))
    Jb = similar(b, length(test_val), length(b))
    ReverseDiff.jacobian!((Ja, Jb), f, (a, b))
    test_approx(Ja, test_a)
    test_approx(Jb, test_b)

    Ja = DiffResults.JacobianResult(test_val, a)
    Jb = DiffResults.JacobianResult(test_val, b)
    ReverseDiff.jacobian!((Ja, Jb), f, (a, b))
    test_approx(DiffResults.value(Ja), test_val)
    test_approx(DiffResults.value(Jb), test_val)
    test_approx(DiffResults.gradient(Ja), test_a)
    test_approx(DiffResults.gradient(Jb), test_b)

    # with JacobianConfig

    cfg = ReverseDiff.JacobianConfig((a, b))

    Ja, Jb = ReverseDiff.jacobian(f, (a, b), cfg)
    test_approx(Ja, test_a)
    test_approx(Jb, test_b)

    Ja = similar(a, length(test_val), length(a))
    Jb = similar(b, length(test_val), length(b))
    ReverseDiff.jacobian!((Ja, Jb), f, (a, b), cfg)
    test_approx(Ja, test_a)
    test_approx(Jb, test_b)

    Ja = DiffResults.JacobianResult(test_val, a)
    Jb = DiffResults.JacobianResult(test_val, b)
    ReverseDiff.jacobian!((Ja, Jb), f, (a, b), cfg)
    test_approx(DiffResults.value(Ja), test_val)
    test_approx(DiffResults.value(Jb), test_val)
    test_approx(DiffResults.jacobian(Ja), test_a)
    test_approx(DiffResults.jacobian(Jb), test_b)

    # with JacobianTape

    tp = ReverseDiff.JacobianTape(f, (rand(size(a)), rand(size(b))))

    Ja, Jb = ReverseDiff.jacobian!(tp, (a, b))
    test_approx(Ja, test_a)
    test_approx(Jb, test_b)

    Ja = similar(a, length(test_val), length(a))
    Jb = similar(b, length(test_val), length(b))
    ReverseDiff.jacobian!((Ja, Jb), tp, (a, b))
    test_approx(Ja, test_a)
    test_approx(Jb, test_b)

    Ja = DiffResults.JacobianResult(test_val, a)
    Jb = DiffResults.JacobianResult(test_val, b)
    ReverseDiff.jacobian!((Ja, Jb), tp, (a, b))
    test_approx(DiffResults.value(Ja), test_val)
    test_approx(DiffResults.value(Jb), test_val)
    test_approx(DiffResults.gradient(Ja), test_a)
    test_approx(DiffResults.gradient(Jb), test_b)

    # with compiled JacobianTape

    if length(tp.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        ctp = ReverseDiff.compile(tp)

        Ja, Jb = ReverseDiff.jacobian!(ctp, (a, b))
        test_approx(Ja, test_a)
        test_approx(Jb, test_b)

        Ja = similar(a, length(test_val), length(a))
        Jb = similar(b, length(test_val), length(b))
        ReverseDiff.jacobian!((Ja, Jb), ctp, (a, b))
        test_approx(Ja, test_a)
        test_approx(Jb, test_b)

        Ja = DiffResults.JacobianResult(test_val, a)
        Jb = DiffResults.JacobianResult(test_val, b)
        ReverseDiff.jacobian!((Ja, Jb), ctp, (a, b))
        test_approx(DiffResults.value(Ja), test_val)
        test_approx(DiffResults.value(Jb), test_val)
        test_approx(DiffResults.gradient(Ja), test_a)
        test_approx(DiffResults.gradient(Jb), test_b)
    end
end

for f in (DiffTests.ARRAY_TO_ARRAY_FUNCS..., DiffTests.MATRIX_TO_MATRIX_FUNCS...)
    test_println("ARRAY_TO_ARRAY_FUNCS + MATRIX_TO_MATRIX_FUNCS", f)
    test_unary_jacobian(f, rand(5, 5))
end

for f! in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS
    test_println("INPLACE_ARRAY_TO_ARRAY_FUNCS", f!)
    test_unary_jacobian(f!, rand(25), rand(25))
end

for f in DiffTests.BINARY_MATRIX_TO_MATRIX_FUNCS
    test_println("BINARY_MATRIX_TO_MATRIX_FUNCS", f)
    test_binary_jacobian(f, rand(5, 5), rand(5, 5))
end


############################################################################################

println("done (took $(toq()) seconds)")

println("testing nested jacobians...")
tic()

############################################################################################

for f in (DiffTests.ARRAY_TO_ARRAY_FUNCS..., DiffTests.MATRIX_TO_MATRIX_FUNCS...)
    test_println("ARRAY_TO_ARRAY_FUNCS + MATRIX_TO_MATRIX_FUNCS", f)

    x = rand(5, 5)
    test = ForwardDiff.jacobian(y -> ForwardDiff.jacobian(f, y), x)

    # without JacobianTape

    J = ReverseDiff.jacobian(y -> ReverseDiff.jacobian(f, y), x)
    test_approx(J, test)

    # with JacobianTape

    tp = ReverseDiff.JacobianTape(y -> ReverseDiff.jacobian(f, y), rand(size(x)))
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

    ra = ReverseDiff.JacobianTape(y -> ReverseDiff.jacobian(x -> f(x, b), y), rand(size(a)))
    rb = ReverseDiff.JacobianTape(y -> ReverseDiff.jacobian(x -> f(a, x), y), rand(size(b)))
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

############################################################################################

println("done (took $(toq()) seconds)")


end # module
