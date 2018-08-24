module GradientTests

using DiffTests, ForwardDiff, ReverseDiff, Test

include(joinpath(dirname(@__FILE__), "../utils.jl"))

function test_unary_gradient(f, x)
    test = ForwardDiff.gradient!(DiffResults.GradientResult(x), f, x)

    # without GradientConfig

    test_approx(ReverseDiff.gradient(f, x), DiffResults.gradient(test))

    out = similar(x)
    ReverseDiff.gradient!(out, f, x)
    test_approx(out, DiffResults.gradient(test))

    result = DiffResults.GradientResult(x)
    ReverseDiff.gradient!(result, f, x)
    test_approx(DiffResults.value(result), DiffResults.value(test))
    test_approx(DiffResults.gradient(result), DiffResults.gradient(test))

    # with GradientConfig

    cfg = ReverseDiff.GradientConfig(x)

    test_approx(ReverseDiff.gradient(f, x, cfg), DiffResults.gradient(test))

    out = similar(x)
    ReverseDiff.gradient!(out, f, x, cfg)
    test_approx(out, DiffResults.gradient(test))

    result = DiffResults.GradientResult(x)
    ReverseDiff.gradient!(result, f, x, cfg)
    test_approx(DiffResults.value(result), DiffResults.value(test))
    test_approx(DiffResults.gradient(result), DiffResults.gradient(test))

    # with GradientTape

    seedx = rand(eltype(x), size(x))
    tp = ReverseDiff.GradientTape(f, seedx)

    test_approx(ReverseDiff.gradient!(tp, x), DiffResults.gradient(test))

    out = similar(x)
    ReverseDiff.gradient!(out, tp, x)
    test_approx(out, DiffResults.gradient(test))

    result = DiffResults.GradientResult(x)
    ReverseDiff.gradient!(result, tp, x)
    test_approx(DiffResults.value(result), DiffResults.value(test))
    test_approx(DiffResults.gradient(result), DiffResults.gradient(test))

    # with compiled GradientTape

    if length(tp.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        ctp = ReverseDiff.compile(tp)

        test_approx(ReverseDiff.gradient!(ctp, x), DiffResults.gradient(test))

        out = similar(x)
        ReverseDiff.gradient!(out, ctp, x)
        test_approx(out, DiffResults.gradient(test))

        result = DiffResults.GradientResult(x)
        ReverseDiff.gradient!(result, ctp, x)
        test_approx(DiffResults.value(result), DiffResults.value(test))
        test_approx(DiffResults.gradient(result), DiffResults.gradient(test))
    end
end

function test_ternary_gradient(f, a, b, c)
    test_val = f(a, b, c)
    test_a = ForwardDiff.gradient(x -> f(x, b, c), a)
    test_b = ForwardDiff.gradient(x -> f(a, x, c), b)
    test_c = ForwardDiff.gradient(x -> f(a, b, x), c)

    # without GradientConfig

    ∇a, ∇b, ∇c = ReverseDiff.gradient(f, (a, b, c))
    test_approx(∇a, test_a)
    test_approx(∇b, test_b)
    test_approx(∇c, test_c)

    ∇a, ∇b, ∇c = map(similar, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), f, (a, b, c))
    test_approx(∇a, test_a)
    test_approx(∇b, test_b)
    test_approx(∇c, test_c)

    ∇a, ∇b, ∇c = map(DiffResults.GradientResult, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), f, (a, b, c))
    test_approx(DiffResults.value(∇a), test_val)
    test_approx(DiffResults.value(∇b), test_val)
    test_approx(DiffResults.value(∇c), test_val)
    test_approx(DiffResults.gradient(∇a), test_a)
    test_approx(DiffResults.gradient(∇b), test_b)
    test_approx(DiffResults.gradient(∇c), test_c)

    # with GradientConfig

    cfg = ReverseDiff.GradientConfig((a, b, c))

    ∇a, ∇b, ∇c = ReverseDiff.gradient(f, (a, b, c), cfg)
    test_approx(∇a, test_a)
    test_approx(∇b, test_b)
    test_approx(∇c, test_c)

    ∇a, ∇b, ∇c = map(similar, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), f, (a, b, c), cfg)
    test_approx(∇a, test_a)
    test_approx(∇b, test_b)
    test_approx(∇c, test_c)

    ∇a, ∇b, ∇c = map(DiffResults.GradientResult, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), f, (a, b, c), cfg)
    test_approx(DiffResults.value(∇a), test_val)
    test_approx(DiffResults.value(∇b), test_val)
    test_approx(DiffResults.value(∇c), test_val)
    test_approx(DiffResults.gradient(∇a), test_a)
    test_approx(DiffResults.gradient(∇b), test_b)
    test_approx(DiffResults.gradient(∇c), test_c)

    # with GradientTape

    tp = ReverseDiff.GradientTape(f, (rand(eltype(a), size(a)), rand(eltype(b), size(b)), rand(eltype(c), size(c))))

    ∇a, ∇b, ∇c = ReverseDiff.gradient!(tp, (a, b, c))
    test_approx(∇a, test_a)
    test_approx(∇b, test_b)
    test_approx(∇c, test_c)

    ∇a, ∇b, ∇c = map(similar, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), tp, (a, b, c))
    test_approx(∇a, test_a)
    test_approx(∇b, test_b)
    test_approx(∇c, test_c)

    ∇a, ∇b, ∇c = map(DiffResults.GradientResult, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), tp, (a, b, c))
    test_approx(DiffResults.value(∇a), test_val)
    test_approx(DiffResults.value(∇b), test_val)
    test_approx(DiffResults.value(∇c), test_val)
    test_approx(DiffResults.gradient(∇a), test_a)
    test_approx(DiffResults.gradient(∇b), test_b)
    test_approx(DiffResults.gradient(∇c), test_c)

    # with compiled GradientTape

    if length(tp.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        ctp = ReverseDiff.compile(tp)

        ∇a, ∇b, ∇c = ReverseDiff.gradient!(ctp, (a, b, c))
        test_approx(∇a, test_a)
        test_approx(∇b, test_b)
        test_approx(∇c, test_c)

        ∇a, ∇b, ∇c = map(similar, (a, b, c))
        ReverseDiff.gradient!((∇a, ∇b, ∇c), ctp, (a, b, c))
        test_approx(∇a, test_a)
        test_approx(∇b, test_b)
        test_approx(∇c, test_c)

        ∇a, ∇b, ∇c = map(DiffResults.GradientResult, (a, b, c))
        ReverseDiff.gradient!((∇a, ∇b, ∇c), ctp, (a, b, c))
        test_approx(DiffResults.value(∇a), test_val)
        test_approx(DiffResults.value(∇b), test_val)
        test_approx(DiffResults.value(∇c), test_val)
        test_approx(DiffResults.gradient(∇a), test_a)
        test_approx(DiffResults.gradient(∇b), test_b)
        test_approx(DiffResults.gradient(∇c), test_c)
    end
end

for f in DiffTests.MATRIX_TO_NUMBER_FUNCS
    test_println("MATRIX_TO_NUMBER_FUNCS", f)
    test_unary_gradient(f, rand(5, 5))
end

for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
    test_println("VECTOR_TO_NUMBER_FUNCS", f)
    test_unary_gradient(f, rand(5))
end

for f in DiffTests.TERNARY_MATRIX_TO_NUMBER_FUNCS
    test_println("TERNARY_MATRIX_TO_NUMBER_FUNCS", f)
    test_ternary_gradient(f, rand(5, 5), rand(5, 5), rand(5, 5))
end

end # module
