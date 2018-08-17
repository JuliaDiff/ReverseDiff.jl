##################
# AbstractConfig #
##################

abstract type AbstractConfig end

Base.show(io::IO, cfg::AbstractConfig) = print(io, typeof(cfg).name)

##################
# GradientConfig #
##################

struct GradientConfig{I} <: AbstractConfig
    input::I
    tape::InstructionTape
    # disable default outer constructor
    GradientConfig{I}(input, tape) where {I} = new{I}(input, tape)
end

# "private" convienence constructor
_GradientConfig(input::I, tape::InstructionTape) where {I} = GradientConfig{I}(input, tape)

"""
    ReverseDiff.GradientConfig(input, tp::InstructionTape = InstructionTape())

Return a `GradientConfig` instance containing the preallocated tape and work buffers used
by the `ReverseDiff.gradient`/`ReverseDiff.gradient!` methods.

Note that `input` is only used for type and shape information; it is not stored or modified
in any way. It is assumed that the element type of `input` is same as the element type of
the target function's output.

See `ReverseDiff.gradient` for a description of acceptable types for `input`.
"""
GradientConfig(input::AbstractArray{T}, tp::InstructionTape = InstructionTape()) where {T} = GradientConfig(input, T, tp)

GradientConfig(input::Tuple, tp::InstructionTape = InstructionTape()) = GradientConfig(input, eltype(first(input)), tp)

"""
    ReverseDiff.GradientConfig(input, ::Type{D}, tp::InstructionTape = InstructionTape())

Like `GradientConfig(input, tp)`, except the provided type `D` is assumed to be the element
type of the target function's output.
"""
function GradientConfig(input::Tuple, ::Type{D}, tp::InstructionTape = InstructionTape()) where D
    return _GradientConfig(map(x -> track(similar(x), D, tp), input), tp)
end

function GradientConfig(input::AbstractArray, ::Type{D}, tp::InstructionTape = InstructionTape()) where D
    return _GradientConfig(track(similar(input), D, tp), tp)
end

##################
# JacobianConfig #
##################

struct JacobianConfig{I,O} <: AbstractConfig
    input::I
    output::O
    tape::InstructionTape
    # disable default outer constructor
    JacobianConfig{I,O}(input, output, tape) where {I,O} = new{I,O}(input, output, tape)
end

# "private" convienence constructor
_JacobianConfig(input::I, output::O, tape::InstructionTape) where {I,O} = JacobianConfig{I,O}(input, output, tape)

"""
    ReverseDiff.JacobianConfig(input, tp::InstructionTape = InstructionTape())

Return a `JacobianConfig` instance containing the preallocated tape and work buffers used
by the `ReverseDiff.jacobian`/`ReverseDiff.jacobian!` methods.

Note that `input` is only used for type and shape information; it is not stored or modified
in any way. It is assumed that the element type of `input` is same as the element type of
the target function's output.

See `ReverseDiff.jacobian` for a description of acceptable types for `input`.

    ReverseDiff.JacobianConfig(input, ::Type{D}, tp::InstructionTape = InstructionTape())

Like `JacobianConfig(input, tp)`, except the provided type `D` is assumed to be the element
type of the target function's output.
"""
function JacobianConfig(args...)
    gcfg = GradientConfig(args...)
    return _JacobianConfig(gcfg.input, nothing, gcfg.tape)
end

"""
    ReverseDiff.JacobianConfig(output::AbstractArray, input, tp::InstructionTape = InstructionTape())

Return a `JacobianConfig` instance containing the preallocated tape and work buffers used
by the `ReverseDiff.jacobian`/`ReverseDiff.jacobian!` methods. This method assumes the
target function has the form `f!(output, input)`

Note that `input` and `output` are only used for type and shape information; they are not
stored or modified in any way.

See `ReverseDiff.jacobian` for a description of acceptable types for `input`.
"""
function JacobianConfig(output::AbstractArray{D}, input::Tuple, tp::InstructionTape = InstructionTape()) where D
    cfg_input = map(x -> track(similar(x), D, tp), input)
    cfg_output = track!(similar(output, TrackedReal{D,D,Nothing}), output, tp)
    return _JacobianConfig(cfg_input, cfg_output, tp)
end

# we dispatch on V<:Real here because InstructionTape is actually also an AbstractArray
function JacobianConfig(output::AbstractArray{D}, input::AbstractArray{V}, tp::InstructionTape = InstructionTape()) where {D,V<:Real}
    cfg_input = track(similar(input), D, tp)
    cfg_output = track!(similar(output, TrackedReal{D,D,Nothing}), output, tp)
    return _JacobianConfig(cfg_input, cfg_output, tp)
end

"""
    ReverseDiff.JacobianConfig(result::DiffResults.DiffResult, input, tp::InstructionTape = InstructionTape())

A convenience method for `JacobianConfig(DiffResults.value(result), input, tp)`.
"""
JacobianConfig(result::DiffResult, input, tp::InstructionTape) = JacobianConfig(DiffResults.value(result), input, tp)

#################
# HessianConfig #
#################

struct HessianConfig{G<:GradientConfig,J<:JacobianConfig} <: AbstractConfig
    gradient_config::G
    jacobian_config::J
end

"""
    ReverseDiff.HessianConfig(input::AbstractArray, gtp::InstructionTape = InstructionTape(), jtp::InstructionTape = InstructionTape())

Return a `HessianConfig` instance containing the preallocated tape and work buffers used
by the `ReverseDiff.hessian`/`ReverseDiff.hessian!` methods. `gtp` is the tape used for
the inner gradient calculation, while `jtp` is used for outer Jacobian calculation.

Note that `input` is only used for type and shape information; it is not stored or modified
in any way. It is assumed that the element type of `input` is same as the element type of
the target function's output.
"""
function HessianConfig(input::AbstractArray, gtp::InstructionTape = InstructionTape(), jtp::InstructionTape = InstructionTape())
    return HessianConfig(input, eltype(input), gtp, jtp)
end

"""
    ReverseDiff.HessianConfig(input::AbstractArray, ::Type{D}, gtp::InstructionTape = InstructionTape(), jtp::InstructionTape = InstructionTape())

Like `HessianConfig(input, tp)`, except the provided type `D` is assumed to be the element
type of the target function's output.
"""
function HessianConfig(input::AbstractArray, ::Type{D}, gtp::InstructionTape = InstructionTape(), jtp::InstructionTape = InstructionTape()) where D
    jcfg = JacobianConfig(input, D, jtp)
    gcfg = GradientConfig(jcfg.input, gtp)
    return HessianConfig(gcfg, jcfg)
end

"""
    ReverseDiff.HessianConfig(result::DiffResults.DiffResult, input::AbstractArray, gtp::InstructionTape = InstructionTape(), jtp::InstructionTape = InstructionTape())

Like `HessianConfig(input, tp)`, but utilize `result` along with `input` to construct work
buffers.

Note that `result` and `input` are only used for type and shape information; they are not
stored or modified in any way.
"""
function HessianConfig(result::DiffResult, input::AbstractArray, gtp::InstructionTape = InstructionTape(), jtp::InstructionTape = InstructionTape())
    jcfg = JacobianConfig(DiffResults.gradient(result), input, jtp)
    gcfg = GradientConfig(jcfg.input, gtp)
    return HessianConfig(gcfg, jcfg)
end
