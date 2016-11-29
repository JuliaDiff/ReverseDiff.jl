module JacobianTests

using DiffBase, ForwardDiff, ReverseDiff, Base.Test

include(joinpath(dirname(@__FILE__), "../utils.jl"))

println("testing jacobian/jacobian!...")
tic()

############################################################################################

function test_unary_jacobian(f, x)
    test = ForwardDiff.jacobian!(DiffBase.JacobianResult(x), f, x, ForwardDiff.JacobianConfig(x))

    # without JacobianConfig

    @test_approx_eq_eps ReverseDiff.jacobian(f, x) DiffBase.jacobian(test) EPS

    out = similar(DiffBase.jacobian(test))
    ReverseDiff.jacobian!(out, f, x)
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS

    result = DiffBase.JacobianResult(x)
    ReverseDiff.jacobian!(result, f, x)
    @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS

    # with JacobianConfig

    cfg = ReverseDiff.JacobianConfig(x)

    @test_approx_eq_eps ReverseDiff.jacobian(f, x, cfg) DiffBase.jacobian(test) EPS

    out = similar(DiffBase.jacobian(test))
    ReverseDiff.jacobian!(out, f, x, cfg)
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS

    result = DiffBase.JacobianResult(x)
    ReverseDiff.jacobian!(result, f, x, cfg)
    @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS

    # with JacobianTape

    tp = ReverseDiff.JacobianTape(f, rand(size(x)))

    @test_approx_eq_eps ReverseDiff.jacobian!(tp, x) DiffBase.jacobian(test) EPS

    out = similar(DiffBase.jacobian(test))
    ReverseDiff.jacobian!(out, tp, x)
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS

    result = DiffBase.JacobianResult(x)
    ReverseDiff.jacobian!(result, tp, x)
    @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS

    # with compiled JacobianTape

    if length(tp.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        ctp = ReverseDiff.compile(tp)

        @test_approx_eq_eps ReverseDiff.jacobian!(ctp, x) DiffBase.jacobian(test) EPS

        out = similar(DiffBase.jacobian(test))
        ReverseDiff.jacobian!(out, ctp, x)
        @test_approx_eq_eps out DiffBase.jacobian(test) EPS

        result = DiffBase.JacobianResult(x)
        ReverseDiff.jacobian!(result, ctp, x)
        @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
        @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS
    end
end

function test_unary_jacobian(f!, y, x)
    y_original = copy(y)
    y_copy = copy(y)
    test = ForwardDiff.jacobian!(DiffBase.JacobianResult(y_copy, x), f!, y_copy, x)

    # without JacobianConfig

    out = ReverseDiff.jacobian(f!, y, x)
    @test_approx_eq_eps y DiffBase.value(test) EPS
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS
    copy!(y, y_original)

    out = similar(DiffBase.jacobian(test))
    ReverseDiff.jacobian!(out, f!, y, x)
    @test_approx_eq_eps y   DiffBase.value(test) EPS
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS
    copy!(y, y_original)

    result = DiffBase.JacobianResult(y, x)
    ReverseDiff.jacobian!(result, f!, y, x)
    @test DiffBase.value(result) == y
    @test_approx_eq_eps y DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS
    copy!(y, y_original)

    # with JacobianConfig

    cfg = ReverseDiff.JacobianConfig(y, x)

    out = ReverseDiff.jacobian(f!, y, x, cfg)
    @test_approx_eq_eps y   DiffBase.value(test) EPS
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS
    copy!(y, y_original)

    out = similar(DiffBase.jacobian(test))
    ReverseDiff.jacobian!(out, f!, y, x, cfg)
    @test_approx_eq_eps y   DiffBase.value(test) EPS
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS
    copy!(y, y_original)

    result = DiffBase.JacobianResult(y, x)
    ReverseDiff.jacobian!(result, f!, y, x, cfg)
    @test DiffBase.value(result) == y
    @test_approx_eq_eps y DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS
    copy!(y, y_original)

    # with JacobianTape

    tp = ReverseDiff.JacobianTape(f!, y, rand(size(x)))

    out = ReverseDiff.jacobian!(tp, x)
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS

    out = similar(DiffBase.jacobian(test))
    ReverseDiff.jacobian!(out, tp, x)
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS

    result = DiffBase.JacobianResult(y, x)
    ReverseDiff.jacobian!(result, tp, x)
    @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS

    # with compiled JacobianTape

    if length(tp.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        ctp = ReverseDiff.compile(tp)

        out = ReverseDiff.jacobian!(ctp, x)
        @test_approx_eq_eps out DiffBase.jacobian(test) EPS

        out = similar(DiffBase.jacobian(test))
        ReverseDiff.jacobian!(out, ctp, x)
        @test_approx_eq_eps out DiffBase.jacobian(test) EPS

        result = DiffBase.JacobianResult(y, x)
        ReverseDiff.jacobian!(result, ctp, x)
        @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
        @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS
    end
end

function test_binary_jacobian(f, a, b)
    test_val = f(a, b)
    test_a = ForwardDiff.jacobian(x -> f(x, b), a)
    test_b = ForwardDiff.jacobian(x -> f(a, x), b)

    # without JacobianConfig

    Ja, Jb = ReverseDiff.jacobian(f, (a, b))
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    Ja = similar(a, length(a), length(b))
    Jb = copy(Ja)
    ReverseDiff.jacobian!((Ja, Jb), f, (a, b))
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    Ja = DiffBase.JacobianResult(a, b)
    Jb = copy(Ja)
    ReverseDiff.jacobian!((Ja, Jb), f, (a, b))
    @test_approx_eq_eps DiffBase.value(Ja) test_val EPS
    @test_approx_eq_eps DiffBase.value(Jb) test_val EPS
    @test_approx_eq_eps DiffBase.gradient(Ja) test_a EPS
    @test_approx_eq_eps DiffBase.gradient(Jb) test_b EPS

    # with JacobianConfig

    cfg = ReverseDiff.JacobianConfig((a, b))

    Ja, Jb = ReverseDiff.jacobian(f, (a, b), cfg)
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    Ja = similar(a, length(a), length(b))
    Jb = copy(Ja)
    ReverseDiff.jacobian!((Ja, Jb), f, (a, b), cfg)
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    Ja = DiffBase.JacobianResult(a, b)
    Jb = copy(Ja)
    ReverseDiff.jacobian!((Ja, Jb), f, (a, b), cfg)
    @test_approx_eq_eps DiffBase.value(Ja) test_val EPS
    @test_approx_eq_eps DiffBase.value(Jb) test_val EPS
    @test_approx_eq_eps DiffBase.jacobian(Ja) test_a EPS
    @test_approx_eq_eps DiffBase.jacobian(Jb) test_b EPS

    # with JacobianTape

    tp = ReverseDiff.JacobianTape(f, (rand(size(a)), rand(size(b))))

    Ja, Jb = ReverseDiff.jacobian!(tp, (a, b))
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    Ja = similar(a, length(a), length(b))
    Jb = copy(Ja)
    ReverseDiff.jacobian!((Ja, Jb), tp, (a, b))
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    Ja = DiffBase.JacobianResult(a, b)
    Jb = copy(Ja)
    ReverseDiff.jacobian!((Ja, Jb), tp, (a, b))
    @test_approx_eq_eps DiffBase.value(Ja) test_val EPS
    @test_approx_eq_eps DiffBase.value(Jb) test_val EPS
    @test_approx_eq_eps DiffBase.gradient(Ja) test_a EPS
    @test_approx_eq_eps DiffBase.gradient(Jb) test_b EPS

    # with compiled JacobianTape

    if length(tp.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        ctp = ReverseDiff.compile(tp)

        Ja, Jb = ReverseDiff.jacobian!(ctp, (a, b))
        @test_approx_eq_eps Ja test_a EPS
        @test_approx_eq_eps Jb test_b EPS

        Ja = similar(a, length(a), length(b))
        Jb = copy(Ja)
        ReverseDiff.jacobian!((Ja, Jb), ctp, (a, b))
        @test_approx_eq_eps Ja test_a EPS
        @test_approx_eq_eps Jb test_b EPS

        Ja = DiffBase.JacobianResult(a, b)
        Jb = copy(Ja)
        ReverseDiff.jacobian!((Ja, Jb), ctp, (a, b))
        @test_approx_eq_eps DiffBase.value(Ja) test_val EPS
        @test_approx_eq_eps DiffBase.value(Jb) test_val EPS
        @test_approx_eq_eps DiffBase.gradient(Ja) test_a EPS
        @test_approx_eq_eps DiffBase.gradient(Jb) test_b EPS
    end
end

for f in (DiffBase.ARRAY_TO_ARRAY_FUNCS..., DiffBase.MATRIX_TO_MATRIX_FUNCS...)
    testprintln("ARRAY_TO_ARRAY_FUNCS + MATRIX_TO_MATRIX_FUNCS", f)
    test_unary_jacobian(f, rand(5, 5))
end

for f! in DiffBase.INPLACE_ARRAY_TO_ARRAY_FUNCS
    testprintln("INPLACE_ARRAY_TO_ARRAY_FUNCS", f!)
    test_unary_jacobian(f!, rand(25), rand(25))
end

for f in DiffBase.BINARY_MATRIX_TO_MATRIX_FUNCS
    testprintln("BINARY_MATRIX_TO_MATRIX_FUNCS", f)
    test_binary_jacobian(f, rand(5, 5), rand(5, 5))
end


############################################################################################

println("done (took $(toq()) seconds)")

println("testing nested jacobians...")
tic()

############################################################################################

for f in (DiffBase.ARRAY_TO_ARRAY_FUNCS..., DiffBase.MATRIX_TO_MATRIX_FUNCS...)
    testprintln("ARRAY_TO_ARRAY_FUNCS + MATRIX_TO_MATRIX_FUNCS", f)

    x = rand(5, 5)
    test = ForwardDiff.jacobian(y -> ForwardDiff.jacobian(f, y), x)

    # without JacobianTape

    J = ReverseDiff.jacobian(y -> ReverseDiff.jacobian(f, y), x)
    @test_approx_eq_eps J test EPS

    # with JacobianTape

    tp = ReverseDiff.JacobianTape(y -> ReverseDiff.jacobian(f, y), rand(size(x)))
    J = ReverseDiff.jacobian!(tp, x)
    @test_approx_eq_eps J test EPS

    # with compiled JacobianTape

    if length(tp.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        J = ReverseDiff.jacobian!(ReverseDiff.compile(tp), x)
        @test_approx_eq_eps J test EPS
    end
end

for f in DiffBase.BINARY_MATRIX_TO_MATRIX_FUNCS
    testprintln("BINARY_MATRIX_TO_MATRIX_FUNCS", f)

    a, b = rand(5, 5), rand(5, 5)

    test_val = f(a, b)
    test_a = ForwardDiff.jacobian(y -> ForwardDiff.jacobian(x -> f(x, b), y), a)
    test_b = ForwardDiff.jacobian(y -> ForwardDiff.jacobian(x -> f(a, x), y), b)

    # without JacobianTape

    Ja = ReverseDiff.jacobian(y -> ReverseDiff.jacobian(x -> f(x, b), y), a)
    Jb = ReverseDiff.jacobian(y -> ReverseDiff.jacobian(x -> f(a, x), y), b)
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    # with JacobianTape

    ra = ReverseDiff.JacobianTape(y -> ReverseDiff.jacobian(x -> f(x, b), y), rand(size(a)))
    rb = ReverseDiff.JacobianTape(y -> ReverseDiff.jacobian(x -> f(a, x), y), rand(size(b)))
    Ja = ReverseDiff.jacobian!(ra, a)
    Jb = ReverseDiff.jacobian!(rb, b)
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    # with compiled JacobianTape

    if length(ra.tape) <= COMPILED_TAPE_LIMIT # otherwise compile time can be crazy
        cra = ReverseDiff.compile(ra)
        Ja = ReverseDiff.jacobian!(cra, a)
        @test_approx_eq_eps Ja test_a EPS
    end

    if length(rb.tape) <= COMPILED_TAPE_LIMIT
        crb = ReverseDiff.compile(rb)
        Jb = ReverseDiff.jacobian!(crb, b)
        @test_approx_eq_eps Jb test_b EPS
    end

    # The below will fail until support for the Jacobian of
    # functions with multiple output arrays is implemented

    # Ja, Jb = ReverseDiff.jacobian((x, y) -> ReverseDiff.jacobian(f, (x, y)), (a, b))
    # @test_approx_eq_eps Ja test_a EPS
    # @test_approx_eq_eps Jb test_b EPS
end

############################################################################################

println("done (took $(toq()) seconds)")


end # module
