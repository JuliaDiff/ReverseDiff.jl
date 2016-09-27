module JacobianTests

using DiffBase, ForwardDiff, ReverseDiffPrototype, Base.Test

const RDP = ReverseDiffPrototype

include("utils.jl")

println("testing jacobian/jacobian!...")
tic()

############################################################################################

for f in (DiffBase.ARRAY_TO_ARRAY_FUNCS..., DiffBase.MATRIX_TO_MATRIX_FUNCS...)
    testprintln("ARRAY_TO_ARRAY_FUNCS + MATRIX_TO_MATRIX_FUNCS", f)

    x = rand(5, 5)
    test = ForwardDiff.jacobian!(DiffBase.JacobianResult(x), f, x, ForwardDiff.Options(x))
    out = similar(DiffBase.jacobian(test))

    @test_approx_eq_eps RDP.jacobian(f, x)       DiffBase.jacobian(test) EPS
    @test_approx_eq_eps RDP.jacobian!(out, f, x) DiffBase.jacobian(test) EPS

    result = RDP.jacobian!(DiffBase.JacobianResult(x), f, x)
    @test_approx_eq_eps DiffBase.value(result)    DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS

    opts = RDP.Options(x)
    @test_approx_eq_eps RDP.jacobian(f, x, opts)       DiffBase.jacobian(test) EPS
    @test_approx_eq_eps RDP.jacobian!(out, f, x, opts) DiffBase.jacobian(test) EPS

    result = RDP.jacobian!(DiffBase.JacobianResult(x), f, x, opts)
    @test_approx_eq_eps DiffBase.value(result)    DiffBase.value(test) EPS
    @test_approx_eq_eps DiffBase.jacobian(result) DiffBase.jacobian(test) EPS
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

    # will fail until support for multiple output arrays is implemented
    # Ja, Jb = RDP.jacobian((x, y) -> RDP.jacobian(f, (x, y)), (a, b))
    # @test_approx_eq_eps Ja test_a EPS
    # @test_approx_eq_eps Jb test_b EPS
end

############################################################################################

println("done (took $(toq()) seconds)")


end # module
