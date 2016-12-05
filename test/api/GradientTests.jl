module GradientTests

using DiffBase, ForwardDiff, ReverseDiff, Base.Test

include(joinpath(dirname(@__FILE__), "../utils.jl"))

println("testing gradient/gradient!...")
tic()

############################################################################################

function test_unary_gradient(f, x)
    test = ForwardDiff.gradient!(DiffBase.GradientResult(x), f, x)

    # without GradientConfig

    @test_approx_eq_eps ReverseDiff.gradient(f, x) DiffBase.gradient(test) EPS

    out = similar(x)
    ReverseDiff.gradient!(out, f, x)
    @test_approx_eq_eps out DiffBase.gradient(test) EPS

    result = DiffBase.GradientResult(x)
    ReverseDiff.gradient!(result, f, x)
    @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.gradient(result) DiffBase.gradient(test) EPS

    # with GradientConfig

    cfg = ReverseDiff.GradientConfig(x)

    @test_approx_eq_eps ReverseDiff.gradient(f, x, cfg) DiffBase.gradient(test) EPS

    out = similar(x)
    ReverseDiff.gradient!(out, f, x, cfg)
    @test_approx_eq_eps out DiffBase.gradient(test) EPS

    result = DiffBase.GradientResult(x)
    ReverseDiff.gradient!(result, f, x, cfg)
    @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.gradient(result) DiffBase.gradient(test) EPS

    # with GradientTape

    seedx = rand(size(x))
    tp = ReverseDiff.GradientTape(f, seedx)

    @test_approx_eq_eps ReverseDiff.gradient!(tp, x) DiffBase.gradient(test) EPS

    out = similar(x)
    ReverseDiff.gradient!(out, tp, x)
    @test_approx_eq_eps out DiffBase.gradient(test) EPS

    result = DiffBase.GradientResult(x)
    ReverseDiff.gradient!(result, tp, x)
    @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.gradient(result) DiffBase.gradient(test) EPS

    # with compiled GradientTape

    if length(tp.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        ∇f! = ReverseDiff.compile_gradient(f, seedx)
        ctp = ReverseDiff.compile(tp)

        @test_approx_eq_eps ReverseDiff.gradient!(ctp, x) DiffBase.gradient(test) EPS

        out = similar(x)
        ReverseDiff.gradient!(out, ctp, x)
        @test_approx_eq_eps out DiffBase.gradient(test) EPS

        out = similar(x)
        ∇f!(out, x)
        @test_approx_eq_eps out DiffBase.gradient(test) EPS

        result = DiffBase.GradientResult(x)
        ReverseDiff.gradient!(result, ctp, x)
        @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
        @test_approx_eq_eps DiffBase.gradient(result) DiffBase.gradient(test) EPS

        result = DiffBase.GradientResult(x)
        ∇f!(result, x)
        @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
        @test_approx_eq_eps DiffBase.gradient(result) DiffBase.gradient(test) EPS
    end
end

for f in DiffBase.MATRIX_TO_NUMBER_FUNCS
    testprintln("MATRIX_TO_NUMBER_FUNCS", f)
    test_unary_gradient(f, rand(5, 5))
end

for f in DiffBase.VECTOR_TO_NUMBER_FUNCS
    testprintln("VECTOR_TO_NUMBER_FUNCS", f)
    test_unary_gradient(f, rand(5))
end

for f in DiffBase.TERNARY_MATRIX_TO_NUMBER_FUNCS
    testprintln("TERNARY_MATRIX_TO_NUMBER_FUNCS", f)

    a, b, c = rand(5, 5), rand(5, 5), rand(5, 5)

    test_val = f(a, b, c)
    test_a = ForwardDiff.gradient(x -> f(x, b, c), a)
    test_b = ForwardDiff.gradient(x -> f(a, x, c), b)
    test_c = ForwardDiff.gradient(x -> f(a, b, x), c)

    # without GradientConfig

    ∇a, ∇b, ∇c = ReverseDiff.gradient(f, (a, b, c))
    @test_approx_eq_eps ∇a test_a EPS
    @test_approx_eq_eps ∇b test_b EPS
    @test_approx_eq_eps ∇c test_c EPS

    ∇a, ∇b, ∇c = map(similar, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), f, (a, b, c))
    @test_approx_eq_eps ∇a test_a EPS
    @test_approx_eq_eps ∇b test_b EPS
    @test_approx_eq_eps ∇c test_c EPS

    ∇a, ∇b, ∇c = map(DiffBase.GradientResult, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), f, (a, b, c))
    @test_approx_eq_eps DiffBase.value(∇a) test_val EPS
    @test_approx_eq_eps DiffBase.value(∇b) test_val EPS
    @test_approx_eq_eps DiffBase.value(∇c) test_val EPS
    @test_approx_eq_eps DiffBase.gradient(∇a) test_a EPS
    @test_approx_eq_eps DiffBase.gradient(∇b) test_b EPS
    @test_approx_eq_eps DiffBase.gradient(∇c) test_c EPS

    # with GradientConfig

    cfg = ReverseDiff.GradientConfig((a, b, c))

    ∇a, ∇b, ∇c = ReverseDiff.gradient(f, (a, b, c), cfg)
    @test_approx_eq_eps ∇a test_a EPS
    @test_approx_eq_eps ∇b test_b EPS
    @test_approx_eq_eps ∇c test_c EPS

    ∇a, ∇b, ∇c = map(similar, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), f, (a, b, c), cfg)
    @test_approx_eq_eps ∇a test_a EPS
    @test_approx_eq_eps ∇b test_b EPS
    @test_approx_eq_eps ∇c test_c EPS

    ∇a, ∇b, ∇c = map(DiffBase.GradientResult, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), f, (a, b, c), cfg)
    @test_approx_eq_eps DiffBase.value(∇a) test_val EPS
    @test_approx_eq_eps DiffBase.value(∇b) test_val EPS
    @test_approx_eq_eps DiffBase.value(∇c) test_val EPS
    @test_approx_eq_eps DiffBase.gradient(∇a) test_a EPS
    @test_approx_eq_eps DiffBase.gradient(∇b) test_b EPS
    @test_approx_eq_eps DiffBase.gradient(∇c) test_c EPS

    # with GradientTape

    tp = ReverseDiff.GradientTape(f, (rand(size(a)), rand(size(b)), rand(size(c))))

    ∇a, ∇b, ∇c = ReverseDiff.gradient!(tp, (a, b, c))
    @test_approx_eq_eps ∇a test_a EPS
    @test_approx_eq_eps ∇b test_b EPS
    @test_approx_eq_eps ∇c test_c EPS

    ∇a, ∇b, ∇c = map(similar, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), tp, (a, b, c))
    @test_approx_eq_eps ∇a test_a EPS
    @test_approx_eq_eps ∇b test_b EPS
    @test_approx_eq_eps ∇c test_c EPS

    ∇a, ∇b, ∇c = map(DiffBase.GradientResult, (a, b, c))
    ReverseDiff.gradient!((∇a, ∇b, ∇c), tp, (a, b, c))
    @test_approx_eq_eps DiffBase.value(∇a) test_val EPS
    @test_approx_eq_eps DiffBase.value(∇b) test_val EPS
    @test_approx_eq_eps DiffBase.value(∇c) test_val EPS
    @test_approx_eq_eps DiffBase.gradient(∇a) test_a EPS
    @test_approx_eq_eps DiffBase.gradient(∇b) test_b EPS
    @test_approx_eq_eps DiffBase.gradient(∇c) test_c EPS

    # with compiled GradientTape

    if length(tp.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        ∇f! = ReverseDiff.compile_gradient(f, (rand(size(a)), rand(size(b)), rand(size(c))))
        ctp = ReverseDiff.compile(tp)

        ∇a, ∇b, ∇c = ReverseDiff.gradient!(ctp, (a, b, c))
        @test_approx_eq_eps ∇a test_a EPS
        @test_approx_eq_eps ∇b test_b EPS
        @test_approx_eq_eps ∇c test_c EPS

        ∇a, ∇b, ∇c = map(similar, (a, b, c))
        ReverseDiff.gradient!((∇a, ∇b, ∇c), ctp, (a, b, c))
        @test_approx_eq_eps ∇a test_a EPS
        @test_approx_eq_eps ∇b test_b EPS
        @test_approx_eq_eps ∇c test_c EPS

        ∇a, ∇b, ∇c = map(similar, (a, b, c))
        ∇f!((∇a, ∇b, ∇c), (a, b, c))
        @test_approx_eq_eps ∇a test_a EPS
        @test_approx_eq_eps ∇b test_b EPS
        @test_approx_eq_eps ∇c test_c EPS

        ∇a, ∇b, ∇c = map(DiffBase.GradientResult, (a, b, c))
        ReverseDiff.gradient!((∇a, ∇b, ∇c), ctp, (a, b, c))
        @test_approx_eq_eps DiffBase.value(∇a) test_val EPS
        @test_approx_eq_eps DiffBase.value(∇b) test_val EPS
        @test_approx_eq_eps DiffBase.value(∇c) test_val EPS
        @test_approx_eq_eps DiffBase.gradient(∇a) test_a EPS
        @test_approx_eq_eps DiffBase.gradient(∇b) test_b EPS
        @test_approx_eq_eps DiffBase.gradient(∇c) test_c EPS

        ∇a, ∇b, ∇c = map(DiffBase.GradientResult, (a, b, c))
        ∇f!((∇a, ∇b, ∇c), (a, b, c))
        @test_approx_eq_eps DiffBase.value(∇a) test_val EPS
        @test_approx_eq_eps DiffBase.value(∇b) test_val EPS
        @test_approx_eq_eps DiffBase.value(∇c) test_val EPS
        @test_approx_eq_eps DiffBase.gradient(∇a) test_a EPS
        @test_approx_eq_eps DiffBase.gradient(∇b) test_b EPS
        @test_approx_eq_eps DiffBase.gradient(∇c) test_c EPS
    end
end

############################################################################################

println("done (took $(toq()) seconds)")

end # module
