using ReverseDiff: Tape, GradientConfig, JacobianConfig, HessianConfig,
                   value, deriv, tape, valtype,
                   derivtype, track, track!

const EPS = 1e-5

const COMPILED_TAPE_LIMIT = 5000
# make RNG deterministic, and thus make result inaccuracies
# deterministic so we don't have to retune EPS for arbitrary inputs
srand(1)

testprintln(kind, f, pad = "  ") = println(pad, "testing $(kind): `$(f)`...")

tracked_is(a, b) = value(a) === value(b) && deriv(a) === deriv(b) && tape(a) === tape(b)
tracked_is(a::AbstractArray, b::AbstractArray) = all(map(tracked_is, a, b))
tracked_is(a::Tuple, b::Tuple) = all(map(tracked_is, a, b))
