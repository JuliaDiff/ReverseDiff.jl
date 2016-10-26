module JacobianTests

using DiffBase, ForwardDiff, ReverseDiffPrototype, Base.Test

include("../utils.jl")

println("testing jacobian/jacobian!...")
tic()

############################################################################################

function test_unary_jacobian(f, x)
    test = ForwardDiff.jacobian!(DiffBase.JacobianResult(x), f, x, ForwardDiff.Options(x))

    # without Options

    @test_approx_eq_eps RDP.jacobian(f, x) DiffBase.jacobian(test) EPS

    out = similar(DiffBase.jacobian(test))
    RDP.jacobian!(out, f, x)
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS

    result = DiffBase.JacobianResult(x)
    RDP.jacobian!(result, f, x)
    @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS

    # with Options

    opts = RDP.Options(x)

    @test_approx_eq_eps RDP.jacobian(f, x, opts) DiffBase.jacobian(test) EPS

    out = similar(DiffBase.jacobian(test))
    RDP.jacobian!(out, f, x, opts)
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS

    result = DiffBase.JacobianResult(x)
    RDP.jacobian!(result, f, x, opts)
    @test_approx_eq_eps DiffBase.value(result) DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS
end

function test_unary_jacobian(f!, y, x)
    y_original = copy(y)
    y_copy = copy(y)
    test = ForwardDiff.jacobian!(DiffBase.JacobianResult(y_copy, x), f!, y_copy, x)

    # without Options

    out = RDP.jacobian(f!, y, x)
    @test_approx_eq_eps y DiffBase.value(test) EPS
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS
    copy!(y, y_original)

    out = similar(DiffBase.jacobian(test))
    RDP.jacobian!(out, f!, y, x)
    @test_approx_eq_eps y   DiffBase.value(test) EPS
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS
    copy!(y, y_original)

    result = DiffBase.JacobianResult(y, x)
    RDP.jacobian!(result, f!, y, x)
    @test DiffBase.value(result) == y
    @test_approx_eq_eps y DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS
    copy!(y, y_original)

    # with Options

    opts = RDP.Options(y, x)

    out = RDP.jacobian(f!, y, x, opts)
    @test_approx_eq_eps y   DiffBase.value(test) EPS
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS
    copy!(y, y_original)

    out = similar(DiffBase.jacobian(test))
    RDP.jacobian!(out, f!, y, x, opts)
    @test_approx_eq_eps y   DiffBase.value(test) EPS
    @test_approx_eq_eps out DiffBase.jacobian(test) EPS
    copy!(y, y_original)

    result = DiffBase.JacobianResult(y, x)
    RDP.jacobian!(result, f!, y, x, opts)
    @test DiffBase.value(result) == y
    @test_approx_eq_eps y DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS
    copy!(y, y_original)
end


for f in (DiffBase.ARRAY_TO_ARRAY_FUNCS..., DiffBase.MATRIX_TO_MATRIX_FUNCS...)
    testprintln("ARRAY_TO_ARRAY_FUNCS + MATRIX_TO_MATRIX_FUNCS", f)
    test_unary_jacobian(f, rand(5, 5))
end

for f! in DiffBase.INPLACE_ARRAY_TO_ARRAY_FUNCS
    testprintln("INPLACE_ARRAY_TO_ARRAY_FUNCS", f!)
    test_unary_jacobian(f!, rand(25), rand(25))
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
    J = RDP.jacobian(y -> RDP.jacobian(f, y), x)

    @test_approx_eq_eps J test EPS
end

for f in DiffBase.BINARY_MATRIX_TO_MATRIX_FUNCS
    testprintln("BINARY_MATRIX_TO_MATRIX_FUNCS", f)

    a, b = rand(5, 5), rand(5, 5)

    test_val = f(a, b)
    test_a = ForwardDiff.jacobian(y -> ForwardDiff.jacobian(x -> f(x, b), y), a)
    test_b = ForwardDiff.jacobian(y -> ForwardDiff.jacobian(x -> f(a, x), y), b)

    Ja = RDP.jacobian(y -> RDP.jacobian(x -> f(x, b), y), a)
    Jb = RDP.jacobian(y -> RDP.jacobian(x -> f(a, x), y), b)
    @test_approx_eq_eps Ja test_a EPS
    @test_approx_eq_eps Jb test_b EPS

    # The below will fail until support for the Jacobian of
    # functions with multiple output arrays is implemented

    # Ja, Jb = RDP.jacobian((x, y) -> RDP.jacobian(f, (x, y)), (a, b))
    # @test_approx_eq_eps Ja test_a EPS
    # @test_approx_eq_eps Jb test_b EPS
end

############################################################################################

println("done (took $(toq()) seconds)")


end # module
