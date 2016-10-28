module ReverseDiff

using Base: RefValue

using DiffBase
using DiffBase: DiffResult

using ForwardDiff
using ForwardDiff: Dual, Partials, partials

# Not all operations will be valid over all of these types, but that's okay; such cases
# will simply error when they hit the original operation in the overloaded definition.
const ARRAY_TYPES = (:AbstractArray, :AbstractVector, :AbstractMatrix, :Array, :Vector, :Matrix)
const REAL_TYPES = (:Bool, :Integer, :Rational, :AbstractFloat, :Real, :Dual)

const FORWARD_UNARY_SCALAR_FUNCS = (ForwardDiff.AUTO_DEFINED_UNARY_FUNCS..., :-, :abs, :conj)
const FORWARD_BINARY_SCALAR_FUNCS = (:*, :/, :+, :-, :^, :atan2)
const SKIPPED_UNARY_SCALAR_FUNCS = (:isinf, :isnan, :isfinite, :iseven, :isodd, :isreal,
                                    :isinteger)
const SKIPPED_BINARY_SCALAR_FUNCS = (:isequal, :isless, :<, :>, :(==), :(!=), :(<=), :(>=))

include("Tape.jl")
include("Tracked.jl")
include("utils.jl")
include("derivatives/macros.jl")
include("derivatives/scalars.jl")
include("derivatives/linalg.jl")
include("derivatives/elementwise.jl")
include("api/options.jl")
include("api/record.jl")
include("api/gradients.jl")
include("api/jacobians.jl")
include("api/hessians.jl")

export DiffBase

end # module
