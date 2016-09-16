module ReverseDiffPrototype

using ForwardDiff
using Base.RefValue

import ForwardDiff: Dual, Partials, GradientResult,
                    JacobianResult, HessianResult,
                    value, partials

############################
# initial type definitions #
############################

include("TraceNode.jl")
include("TraceReal.jl")
include("TraceArray.jl")

##############
# wrap/wrap! #
##############

wrap(x, tr::Trace = Trace()) = wrap(eltype(x), x, tr)
wrap{S}(::Type{S}, x, tr::Trace = Trace()) = wrap(S, x, Nullable(tr))
wrap{S}(::Type{S}, x::Real, tr::Nullable{Trace}) = TraceReal{S}(x, tr)

function wrap{S}(::Type{S}, x, tr::Nullable{Trace})
    return wrap!(similar(x, TraceReal{S,eltype(x)}), x, tr)
end

function wrap!(out, x, tr::Nullable{Trace})
    S = adjtype(eltype(out))
    for i in eachindex(out)
        out[i] = TraceReal{S}(x[i], tr)
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
