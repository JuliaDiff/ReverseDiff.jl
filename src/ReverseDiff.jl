module ReverseDiff

using Base: RefValue
using ChainRulesCore: ChainRulesCore, AbstractThunk, unthunk
using DiffResults: DiffResults, DiffResult
using DiffRules: DiffRules
using ForwardDiff: ForwardDiff, Dual
using FunctionWrappers: FunctionWrapper
using LinearAlgebra: LinearAlgebra, Adjoint, Transpose, det, dot, lmul!, mul!
using LogExpFunctions: LogExpFunctions
using MacroTools: MacroTools, @capture
using NaNMath: NaNMath
using Random: Random
using SpecialFunctions: SpecialFunctions
using StaticArrays: StaticArrays, SVector

# Not all operations will be valid over all of these types, but that's okay; such cases
# will simply error when they hit the original operation in the overloaded definition.
const ARRAY_TYPES = (:AbstractArray, :AbstractVector, :AbstractMatrix, :Array, :Vector, :Matrix)
const REAL_TYPES = (:Bool, :Integer, :AbstractIrrational, :Rational, :BigFloat, :BigInt, :AbstractFloat, :Real, :Dual)

const SKIPPED_UNARY_SCALAR_FUNCS  = Symbol[:isinf, :isnan, :isfinite, :iseven, :isodd, :isreal, :isinteger, :exponent]
const SKIPPED_BINARY_SCALAR_FUNCS = Symbol[:isequal, :isless, :<, :>, :(==), :(!=), :(<=), :(>=)]

# Some functions with derivatives in DiffRules are not supported
# For instance, ReverseDiff does not support functions with complex results and derivatives
const SKIPPED_DIFFRULES = Tuple{Symbol,Symbol}[
    (:SpecialFunctions, :hankelh1),
    (:SpecialFunctions, :hankelh1x),
    (:SpecialFunctions, :hankelh2),
    (:SpecialFunctions, :hankelh2x),
    (:SpecialFunctions, :besselh),
    (:SpecialFunctions, :besselhx),
]

include("tape.jl")
include("tracked.jl")
include("macros.jl")
include("derivatives/arrays.jl")
include("derivatives/propagation.jl")
include("derivatives/broadcast.jl")
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

export DiffResults

end # module
