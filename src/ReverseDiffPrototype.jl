module ReverseDiffPrototype

using ForwardDiff
using Base.RefValue

import ForwardDiff: Dual, Partials, GradientResult,
                    JacobianResult, HessianResult,
                    value, partials

############################
# initial type definitions #
############################

include("Tape.jl")
include("Tracer.jl")
include("arrays.jl")

################
# track/track! #
################

track(x, tp::Tape = Tape()) = track(eltype(x), x, tp)
track{S}(::Type{S}, x, tp::Tape = Tape()) = track(S, x, Nullable(tp))
track{S}(::Type{S}, x::Real, tp::Nullable{Tape}) = Tracer{S}(x, tp)

function track{S}(::Type{S}, x, tp::Nullable{Tape})
    return track!(similar(x, Tracer{S,eltype(x)}), x, tp)
end

function track!(out, x, tp::Nullable{Tape})
    S = adjtype(eltype(out))
    for i in eachindex(out)
        out[i] = Tracer{S}(x[i], tp)
    end
    return out
end

###############################
# ...and the rest of the code #
###############################

# Not all operations will be valid over all of these types, but that's okay; such cases
# will simply error when they hit the original operation in the overloaded definition.
const ARRAY_TYPES = (:AbstractArray, :AbstractVector, :AbstractMatrix, :Array, :Vector, :Matrix)
const REAL_TYPES = (:Bool, :Integer, :Rational, :Real, :Dual)

const FORWARD_UNARY_SCALAR_FUNCS = (ForwardDiff.AUTO_DEFINED_UNARY_FUNCS..., :-, :abs, :conj)
const FORWARD_BINARY_SCALAR_FUNCS = (:*, :/, :+, :-, :^, :atan2)
const SKIPPED_SCALAR_COMPARATORS = (:isequal, :isless, :isinf, :isnan, :isfinite, :iseven,
                                    :isodd, :isreal, :isinteger, :<, :>, :(==), :(!=),
                                    :(<=), :(>=))

include("optimizations/macros.jl")
include("optimizations/scalars.jl")
include("optimizations/arrays.jl")
include("optimizations/elementwise.jl")
include("backprop.jl")
include("api.jl")

end # module
