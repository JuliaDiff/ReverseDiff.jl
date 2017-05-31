__precompile__()

module ReverseDiff

using Base: RefValue
using FunctionWrappers: FunctionWrapper

using Compat

using DiffBase
using DiffBase: DiffResult

using ForwardDiff
using ForwardDiff: Dual, Partials

if VERSION >= v"0.6.0-dev.1024"
    const compat_product = Base.Iterators.product
else
    const compat_product = Base.product
end

# Not all operations will be valid over all of these types, but that's okay; such cases
# will simply error when they hit the original operation in the overloaded definition.
const ARRAY_TYPES = (:AbstractArray, :AbstractVector, :AbstractMatrix, :Array, :Vector, :Matrix)
const REAL_TYPES = (:Bool, :Integer, :Rational, :BigFloat, :BigInt, :AbstractFloat, :Real, :Dual)

const FORWARD_UNARY_SCALAR_FUNCS = (ForwardDiff.AUTO_DEFINED_UNARY_FUNCS..., :-, :abs, :conj)
const FORWARD_BINARY_SCALAR_FUNCS = (:*, :/, :+, :-, :^, :atan2)
const SKIPPED_UNARY_SCALAR_FUNCS = (:isinf, :isnan, :isfinite, :iseven, :isodd, :isreal,
                                    :isinteger)
const SKIPPED_BINARY_SCALAR_FUNCS = (:isequal, :isless, :<, :>, :(==), :(!=), :(<=), :(>=))

include("tape.jl")
include("tracked.jl")
include("macros.jl")
include("derivatives/propagation.jl")
include("derivatives/scalars.jl")
include("derivatives/elementwise.jl")
include("derivatives/linalg/arithmetic.jl")
include("derivatives/linalg/reductions.jl")
include("derivatives/linalg/special.jl")
include("api/utils.jl")
include("api/Config.jl")
include("api/tape.jl")
include("api/gradients.jl")
include("api/jacobians.jl")
include("api/hessians.jl")

export DiffBase

end # module
