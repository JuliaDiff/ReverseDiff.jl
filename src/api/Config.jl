##################
# AbstractConfig #
##################

abstract AbstractConfig

Base.show(io::IO, cfg::AbstractConfig) = print(io, typeof(cfg).name)

##################
# GradientConfig #
##################

immutable GradientConfig{I} <: AbstractConfig
    input::I
    tape::Tape
    # disable default outer constructor
    GradientConfig(input, tape) = new(input, tape)
end

# "private" convienence constructor
_GradientConfig{I}(input::I, tape::Tape) = GradientConfig{I}(input, tape)

GradientConfig{T}(input::AbstractArray{T}, tp::Tape = Tape()) = GradientConfig(input, T, tp)

GradientConfig(input::Tuple, tp::Tape = Tape()) = GradientConfig(input, eltype(first(input)), tp)

function GradientConfig{D}(input::Tuple, ::Type{D}, tp::Tape = Tape())
    return _GradientConfig(map(x -> track(similar(x), D, tp), input), tp)
end

function GradientConfig{D}(input::AbstractArray, ::Type{D}, tp::Tape = Tape())
    return _GradientConfig(track(similar(input), D, tp), tp)
end

##################
# JacobianConfig #
##################

immutable JacobianConfig{I,O} <: AbstractConfig
    input::I
    output::O
    tape::Tape
    # disable default outer constructor
    JacobianConfig(input, output, tape) = new(input, output, tape)
end

# "private" convienence constructor
_JacobianConfig{I,O}(input::I, output::O, tape::Tape) = JacobianConfig{I,O}(input, output, tape)

JacobianConfig(result::DiffResult, args...) = JacobianConfig(DiffBase.value(result), args...)

function JacobianConfig(args...)
    gcfg = GradientConfig(args...)
    return _JacobianConfig(gcfg.input, nothing, gcfg.tape)
end

function JacobianConfig{D}(output::AbstractArray{D}, input::Tuple, tp::Tape = Tape())
    cfg_input = map(x -> track(similar(x), D, tp), input)
    cfg_output = track!(similar(output, TrackedReal{D,D,Void}), output, tp)
    return _JacobianConfig(cfg_input, cfg_output, tp)
end

function JacobianConfig{D,V<:Real}(output::AbstractArray{D}, input::AbstractArray{V}, tp::Tape = Tape())
    cfg_input = track(similar(input), D, tp)
    cfg_output = track!(similar(output, TrackedReal{D,D,Void}), output, tp)
    return _JacobianConfig(cfg_input, cfg_output, tp)
end

#################
# HessianConfig #
#################

immutable HessianConfig{G<:GradientConfig,J<:JacobianConfig} <: AbstractConfig
    gradient_config::G
    jacobian_config::J
end

function HessianConfig(input::AbstractArray, gtp::Tape = Tape(), jtp::Tape = Tape())
    return HessianConfig(input, eltype(input), gtp, jtp)
end

function HessianConfig{D}(input::AbstractArray, ::Type{D}, gtp::Tape = Tape(), jtp::Tape = Tape())
    jcfg = JacobianConfig(input, D, jtp)
    gcfg = GradientConfig(jcfg.input, gtp)
    return HessianConfig(gcfg, jcfg)
end

function HessianConfig(result::DiffResult, input::AbstractArray, gtp::Tape = Tape(), jtp::Tape = Tape())
    jcfg = JacobianConfig(DiffBase.gradient(result), input, jtp)
    gcfg = GradientConfig(jcfg.input, gtp)
    return HessianConfig(gcfg, jcfg)
end
