module JacobianTests

using DiffBase, ForwardDiff, ReverseDiff, Base.Test

include("../utils.jl")

println("testing jacobian/jacobian!...")
tic()

############################################################################################

function test_unary_jacobian(f, x)
    test = ForwardDiff.jacobian!(DiffBase.JacobianResult(x), f, x, ForwardDiff.Options(x))

    # without Options

    @test_approx_eq_eps ReverseDiff.jacobian(f, x) DiffBase.jacobian(test) EPS

    out = similar(DiffBase.jacobian(test))
    ReverseDiff.jacobian!(out, f, x)
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS

    result = DiffBase.JacobianResult(x)
    ReverseDiff.jacobian!(result, f, x)
    @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS

    # with Options

    opts = ReverseDiff.Options(x)

    @test_approx_eq_eps ReverseDiff.jacobian(f, x, opts) DiffBase.jacobian(test) EPS

    out = similar(DiffBase.jacobian(test))
    ReverseDiff.jacobian!(out, f, x, opts)
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS

    result = DiffBase.JacobianResult(x)
    ReverseDiff.jacobian!(result, f, x, opts)
    @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS

    # with Record

    r = ReverseDiff.Record(f, rand(size(x)))

    @test_approx_eq_eps ReverseDiff.jacobian!(r, x) DiffBase.jacobian(test) EPS

    out = similar(DiffBase.jacobian(test))
    ReverseDiff.jacobian!(out, r, x)
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS

    result = DiffBase.JacobianResult(x)
    ReverseDiff.jacobian!(result, r, x)
    @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS

end

function test_unary_jacobian(f!, y, x)
    y_original = copy(y)
    y_copy = copy(y)
    test = ForwardDiff.jacobian!(DiffBase.JacobianResult(y_copy, x), f!, y_copy, x)

    # without Options

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

    # with Options

    opts = ReverseDiff.Options(y, x)

    out = ReverseDiff.jacobian(f!, y, x, opts)
    @test_approx_eq_eps y   DiffBase.value(test) EPS
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS
    copy!(y, y_original)

    out = similar(DiffBase.jacobian(test))
    ReverseDiff.jacobian!(out, f!, y, x, opts)
    @test_approx_eq_eps y   DiffBase.value(test) EPS
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS
    copy!(y, y_original)

    result = DiffBase.JacobianResult(y, x)
    ReverseDiff.jacobian!(result, f!, y, x, opts)
    @test DiffBase.value(result) == y
    @test_approx_eq_eps y DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS
    copy!(y, y_original)

    # with Record

    r = ReverseDiff.Record(f!, y, rand(size(x)))

    out = ReverseDiff.jacobian!(r, x)
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS

    out = similar(DiffBase.jacobian(test))
    ReverseDiff.jacobian!(out, r, x)
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS

    result = DiffBase.JacobianResult(y, x)
    ReverseDiff.jacobian!(result, r, x)
    @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS
end

function test_binary_jacobian(f, a, b)
    test_val = f(a, b)
    test_a = ForwardDiff.jacobian(x -> f(x, b), a)
    test_b = ForwardDiff.jacobian(x -> f(a, x), b)

    # without Options

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

    # with Options

    opts = ReverseDiff.Options((a, b))

    Ja, Jb = ReverseDiff.jacobian(f, (a, b), opts)
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    Ja = similar(a, length(a), length(b))
    Jb = copy(Ja)
    ReverseDiff.jacobian!((Ja, Jb), f, (a, b), opts)
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    Ja = DiffBase.JacobianResult(a, b)
    Jb = copy(Ja)
    ReverseDiff.jacobian!((Ja, Jb), f, (a, b), opts)
    @test_approx_eq_eps DiffBase.value(Ja) test_val EPS
    @test_approx_eq_eps DiffBase.value(Jb) test_val EPS
    @test_approx_eq_eps DiffBase.jacobian(Ja) test_a EPS
    @test_approx_eq_eps DiffBase.jacobian(Jb) test_b EPS

    # with Record

    r = ReverseDiff.Record(f, (rand(size(a)), rand(size(b))))

    Ja, Jb = ReverseDiff.jacobian!(r, (a, b))
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    Ja = similar(a, length(a), length(b))
    Jb = copy(Ja)
    ReverseDiff.jacobian!((Ja, Jb), r, (a, b))
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    Ja = DiffBase.JacobianResult(a, b)
    Jb = copy(Ja)
    ReverseDiff.jacobian!((Ja, Jb), r, (a, b))
    @test_approx_eq_eps DiffBase.value(Ja) test_val EPS
    @test_approx_eq_eps DiffBase.value(Jb) test_val EPS
    @test_approx_eq_eps DiffBase.gradient(Ja) test_a EPS
    @test_approx_eq_eps DiffBase.gradient(Jb) test_b EPS
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

    # without Record

    J = ReverseDiff.jacobian(y -> ReverseDiff.jacobian(f, y), x)
    @test_approx_eq_eps J test EPS

    # with Record

    r = ReverseDiff.Record(y -> ReverseDiff.jacobian(f, y), rand(size(x)))
    J = ReverseDiff.jacobian!(r, x)
    @test_approx_eq_eps J test EPS
end

for f in DiffBase.BINARY_MATRIX_TO_MATRIX_FUNCS
    testprintln("BINARY_MATRIX_TO_MATRIX_FUNCS", f)

    a, b = rand(5, 5), rand(5, 5)

    test_val = f(a, b)
    test_a = ForwardDiff.jacobian(y -> ForwardDiff.jacobian(x -> f(x, b), y), a)
    test_b = ForwardDiff.jacobian(y -> ForwardDiff.jacobian(x -> f(a, x), y), b)

    # without Record

    Ja = ReverseDiff.jacobian(y -> ReverseDiff.jacobian(x -> f(x, b), y), a)
    Jb = ReverseDiff.jacobian(y -> ReverseDiff.jacobian(x -> f(a, x), y), b)
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    # with Record

    ra = ReverseDiff.Record(y -> ReverseDiff.jacobian(x -> f(x, b), y), rand(size(a)))
    rb = ReverseDiff.Record(y -> ReverseDiff.jacobian(x -> f(a, x), y), rand(size(b)))
    Ja = ReverseDiff.jacobian!(ra, a)
    Jb = ReverseDiff.jacobian!(rb, b)
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    # The below will fail until support for the Jacobian of
    # functions with multiple output arrays is implemented

    # Ja, Jb = ReverseDiff.jacobian((x, y) -> ReverseDiff.jacobian(f, (x, y)), (a, b))
    # @test_approx_eq_eps Ja test_a EPS
    # @test_approx_eq_eps Jb test_b EPS
end

############################################################################################

println("done (took $(toq()) seconds)")


end # module
