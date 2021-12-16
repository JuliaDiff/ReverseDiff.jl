using ReverseDiff: InstructionTape, GradientConfig, JacobianConfig, HessianConfig,
                   value, deriv, tape, valtype,
                   derivtype, track, track!

using Random

const COMPILED_TAPE_LIMIT = 5000

# These functions correctly emit NaNs for certain arguments, but ReverseDiff's test
# machinery is currently too dumb to handle them properly.
const SKIPPED_BINARY_SCALAR_TESTS = Symbol[:hankelh1, :hankelh1x, :hankelh2, :hankelh2x,
                                           :pow, :besselj, :besseli, :bessely, :besselk,
                                           :polygamma, :ldexp]

# make RNG deterministic, and thus make result inaccuracies
# deterministic so we don't have to retune EPS for arbitrary inputs
Random.seed!(1)

test_println(kind, f, pad = "  ") = println(pad, "testing $(kind): `$(f)`...")

@inline test_approx(A, B, _atol = 1e-5) = @test isapprox(A, B, atol = _atol)

tracked_is(a, b) = value(a) === value(b) && deriv(a) === deriv(b) && tape(a) === tape(b)
tracked_is(a::AbstractArray, b::AbstractArray) = all(map(tracked_is, a, b))
tracked_is(a::Tuple, b::Tuple) = all(map(tracked_is, a, b))

# ensure that input is in domain of function
# here `x` is a scalar or array generated with `rand(dims...)`, i.e., values of `x`
# are between 0 and 1
function modify_input(f, x)
    return if in(f, (:asec, :acsc, :asecd, :acscd, :acosh, :acoth))
        x .+ one(eltype(x))
    elseif f === :log1mexp || f === :log2mexp
        x .- one(eltype(x))
    else
        x
    end
end
