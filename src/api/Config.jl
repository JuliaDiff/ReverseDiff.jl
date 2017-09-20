##################
# AbstractConfig #
##################

@compat abstract type AbstractConfig end

Base.show(io::IO, cfg::AbstractConfig) = print(io, typeof(cfg).name)

##################
# GradientConfig #
##################

@compat immutable GradientConfig{I} <: AbstractConfig
    input::I
    tape::InstructionTape
    # disable default outer constructor
    (::Type{GradientConfig{I}}){I}(input, tape) = new{I}(input, tape)
end

# "private" convienence constructor
_GradientConfig{I}(input::I, tape::InstructionTape) = GradientConfig{I}(input, tape)

"""
    ReverseDiff.GradientConfig(input, tp::InstructionTape = InstructionTape())

Return a `GradientConfig` instance containing the preallocated tape and work buffers used
by the `ReverseDiff.gradient`/`ReverseDiff.gradient!` methods.

Note that `input` is only used for type and shape information; it is not stored or modified
in any way. It is assumed that the element type of `input` is same as the element type of
the target function's output.

See `ReverseDiff.gradient` for a description of acceptable types for `input`.
"""
GradientConfig{T}(input::AbstractArray{T}, tp::InstructionTape = InstructionTape()) = GradientConfig(input, T, tp)

GradientConfig(input::Tuple, tp::InstructionTape = InstructionTape()) = GradientConfig(input, eltype(first(input)), tp)

"""
    ReverseDiff.GradientConfig(input, ::Type{D}, tp::InstructionTape = InstructionTape())

Like `GradientConfig(input, tp)`, except the provided type `D` is assumed to be the element
type of the target function's output.
"""
function GradientConfig{D}(input::Tuple, ::Type{D}, tp::InstructionTape = InstructionTape())
    return _GradientConfig(map(x -> track(similar(x), D, tp), input), tp)
end

function GradientConfig{D}(input::AbstractArray, ::Type{D}, tp::InstructionTape = InstructionTape())
    return _GradientConfig(track(similar(input), D, tp), tp)
end

##################
# JacobianConfig #
##################

@compat immutable JacobianConfig{I,O} <: AbstractConfig
    input::I
    output::O
    tape::InstructionTape
    # disable default outer constructor
    (::Type{JacobianConfig{I,O}}){I,O}(input, output, tape) = new{I,O}(input, output, tape)
end

# "private" convienence constructor
_JacobianConfig{I,O}(input::I, output::O, tape::InstructionTape) = JacobianConfig{I,O}(input, output, tape)

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
function JacobianConfig{D}(output::AbstractArray{D}, input::Tuple, tp::InstructionTape = InstructionTape())
    cfg_input = map(x -> track(similar(x), D, tp), input)
    cfg_output = track!(similar(output, TrackedReal{D,D,Void}), output, tp)
    return _JacobianConfig(cfg_input, cfg_output, tp)
end

# we dispatch on V<:Real here because InstructionTape is actually also an AbstractArray
function JacobianConfig{D,V<:Real}(output::AbstractArray{D}, input::AbstractArray{V}, tp::InstructionTape = InstructionTape())
    cfg_input = track(similar(input), D, tp)
    cfg_output = track!(similar(output, TrackedReal{D,D,Void}), output, tp)
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

immutable HessianConfig{G<:GradientConfig,J<:JacobianConfig} <: AbstractConfig
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
function HessianConfig{D}(input::AbstractArray, ::Type{D}, gtp::InstructionTape = InstructionTape(), jtp::InstructionTape = InstructionTape())
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
