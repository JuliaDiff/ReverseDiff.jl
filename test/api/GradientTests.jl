module GradientTests

using DiffBase, ForwardDiff, ReverseDiff, Base.Test

include(joinpath(dirname(@__FILE__), "../utils.jl"))

println("testing gradient/gradient!...")
tic()

############################################################################################

function test_unary_gradient(f, x)
    test = ForwardDiff.gradient!(DiffBase.GradientResult(x), f, x)

    # without GradientConfig

    test_approx(ReverseDiff.gradient(f, x), DiffBase.gradient(test))

    out = similar(x)
    ReverseDiff.gradient!(out, f, x)
    test_approx(out, DiffBase.gradient(test))

    result = DiffBase.GradientResult(x)
    ReverseDiff.gradient!(result, f, x)
    test_approx(DiffBase.value(result), DiffBase.value(test))
    test_approx(DiffBase.gradient(result), DiffBase.gradient(test))

    # with GradientConfig

    cfg = ReverseDiff.GradientConfig(x)

    test_approx(ReverseDiff.gradient(f, x, cfg), DiffBase.gradient(test))

    out = similar(x)
    ReverseDiff.gradient!(out, f, x, cfg)
    test_approx(out, DiffBase.gradient(test))

    result = DiffBase.GradientResult(x)
    ReverseDiff.gradient!(result, f, x, cfg)
    test_approx(DiffBase.value(result), DiffBase.value(test))
    test_approx(DiffBase.gradient(result), DiffBase.gradient(test))

    # with GradientTape

    seedx = rand(size(x))
    tp = ReverseDiff.GradientTape(f, seedx)

    test_approx(ReverseDiff.gradient!(tp, x), DiffBase.gradient(test))

    out = similar(x)
    ReverseDiff.gradient!(out, tp, x)
    test_approx(out, DiffBase.gradient(test))

    result = DiffBase.GradientResult(x)
    ReverseDiff.gradient!(result, tp, x)
    test_approx(DiffBase.value(result), DiffBase.value(test))
    test_approx(DiffBase.gradient(result), DiffBase.gradient(test))

    # with compiled GradientTape

    if length(tp.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        ∇f! = ReverseDiff.compile_gradient(f, seedx)
        ctp = ReverseDiff.compile(tp)

        # circumvent world-age problems (`ctp` and `∇f!` were generated via `eval`)
        @eval begin
            test, x = $test, $x
            ∇f!, ctp = $∇f!, $ctp

            test_approx(ReverseDiff.gradient!(ctp, x), DiffBase.gradient(test))

            out = similar(x)
            ReverseDiff.gradient!(out, ctp, x)
            test_approx(out, DiffBase.gradient(test))

            out = similar(x)
            ∇f!(out, x)
            test_approx(out, DiffBase.gradient(test))

            result = DiffBase.GradientResult(x)
            ReverseDiff.gradient!(result, ctp, x)
            test_approx(DiffBase.value(result), DiffBase.value(test))
            test_approx(DiffBase.gradient(result), DiffBase.gradient(test))

            result = DiffBase.GradientResult(x)
            ∇f!(result, x)
            test_approx(DiffBase.value(result), DiffBase.value(test))
            test_approx(DiffBase.gradient(result), DiffBase.gradient(test))
        end
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

    ∇a, ∇b, ∇c = map(DiffBase.GradientResult, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), f, (a, b, c))
    test_approx(DiffBase.value(∇a), test_val)
    test_approx(DiffBase.value(∇b), test_val)
    test_approx(DiffBase.value(∇c), test_val)
    test_approx(DiffBase.gradient(∇a), test_a)
    test_approx(DiffBase.gradient(∇b), test_b)
    test_approx(DiffBase.gradient(∇c), test_c)

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

    ∇a, ∇b, ∇c = map(DiffBase.GradientResult, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), f, (a, b, c), cfg)
    test_approx(DiffBase.value(∇a), test_val)
    test_approx(DiffBase.value(∇b), test_val)
    test_approx(DiffBase.value(∇c), test_val)
    test_approx(DiffBase.gradient(∇a), test_a)
    test_approx(DiffBase.gradient(∇b), test_b)
    test_approx(DiffBase.gradient(∇c), test_c)

    # with GradientTape

    tp = ReverseDiff.GradientTape(f, (rand(size(a)), rand(size(b)), rand(size(c))))

    ∇a, ∇b, ∇c = ReverseDiff.gradient!(tp, (a, b, c))
    test_approx(∇a, test_a)
    test_approx(∇b, test_b)
    test_approx(∇c, test_c)

    ∇a, ∇b, ∇c = map(similar, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), tp, (a, b, c))
    test_approx(∇a, test_a)
    test_approx(∇b, test_b)
    test_approx(∇c, test_c)

    ∇a, ∇b, ∇c = map(DiffBase.GradientResult, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), tp, (a, b, c))
    test_approx(DiffBase.value(∇a), test_val)
    test_approx(DiffBase.value(∇b), test_val)
    test_approx(DiffBase.value(∇c), test_val)
    test_approx(DiffBase.gradient(∇a), test_a)
    test_approx(DiffBase.gradient(∇b), test_b)
    test_approx(DiffBase.gradient(∇c), test_c)

    # with compiled GradientTape

    if length(tp.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        ∇f! = ReverseDiff.compile_gradient(f, (rand(size(a)), rand(size(b)), rand(size(c))))
        ctp = ReverseDiff.compile(tp)

        # circumvent world-age problems (`ctp` and `∇f!` have a future world age)
        @eval begin
            test_val, test_a, test_b, test_c = $test_val, $test_a, $test_b, $test_c
            a, b, c = $a, $b, $c
            ∇f!, ctp = $∇f!, $ctp

            ∇a, ∇b, ∇c = ReverseDiff.gradient!(ctp, (a, b, c))
            test_approx(∇a, test_a)
            test_approx(∇b, test_b)
            test_approx(∇c, test_c)

            ∇a, ∇b, ∇c = map(similar, (a, b, c))
            ReverseDiff.gradient!((∇a, ∇b, ∇c), ctp, (a, b, c))
            test_approx(∇a, test_a)
            test_approx(∇b, test_b)
            test_approx(∇c, test_c)

            ∇a, ∇b, ∇c = map(similar, (a, b, c))
            ∇f!((∇a, ∇b, ∇c), (a, b, c))
            test_approx(∇a, test_a)
            test_approx(∇b, test_b)
            test_approx(∇c, test_c)

            ∇a, ∇b, ∇c = map(DiffBase.GradientResult, (a, b, c))
            ReverseDiff.gradient!((∇a, ∇b, ∇c), ctp, (a, b, c))
            test_approx(DiffBase.value(∇a), test_val)
            test_approx(DiffBase.value(∇b), test_val)
            test_approx(DiffBase.value(∇c), test_val)
            test_approx(DiffBase.gradient(∇a), test_a)
            test_approx(DiffBase.gradient(∇b), test_b)
            test_approx(DiffBase.gradient(∇c), test_c)

            ∇a, ∇b, ∇c = map(DiffBase.GradientResult, (a, b, c))
            ∇f!((∇a, ∇b, ∇c), (a, b, c))
            test_approx(DiffBase.value(∇a), test_val)
            test_approx(DiffBase.value(∇b), test_val)
            test_approx(DiffBase.value(∇c), test_val)
            test_approx(DiffBase.gradient(∇a), test_a)
            test_approx(DiffBase.gradient(∇b), test_b)
            test_approx(DiffBase.gradient(∇c), test_c)
        end
    end
end

for f in DiffBase.MATRIX_TO_NUMBER_FUNCS
    test_println("MATRIX_TO_NUMBER_FUNCS", f)
    test_unary_gradient(f, rand(5, 5))
end

for f in DiffBase.VECTOR_TO_NUMBER_FUNCS
    test_println("VECTOR_TO_NUMBER_FUNCS", f)
    test_unary_gradient(f, rand(5))
end

for f in DiffBase.TERNARY_MATRIX_TO_NUMBER_FUNCS
    test_println("TERNARY_MATRIX_TO_NUMBER_FUNCS", f)
    test_ternary_gradient(f, rand(5, 5), rand(5, 5), rand(5, 5))
end

############################################################################################

println("done (took $(toq()) seconds)")

end # module
