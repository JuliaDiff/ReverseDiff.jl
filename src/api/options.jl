abstract AbstractOptions

###########
# Options #
###########

immutable Options{S,T} <: AbstractOptions
    state::S
    tape::T
    # disable default outer constructor
    Options(state::S, tape::T) = new(state, tape)
end

# hidden convienence constructor

_Options{S,T}(state::S, tape::T) = Options{S,T}(state, tape)

# public outer constructors

Options{T}(x::AbstractArray{T}, tp::Tape = Tape()) = Options(x, T, tp)
Options(xs::Tuple, tp::Tape = Tape()) = Options(xs, eltype(first(xs)), tp)

function Options{T,A}(x::AbstractArray{T}, ::Type{A}, tp::Tape = Tape())
    state = similar(x, Tracked{T,A})
    return _Options(state, tp)
end

function Options{A}(xs::Tuple, ::Type{A}, tp::Tape = Tape())
    state = map(x -> similar(x, Tracked{eltype(x),A}), xs)
    return _Options(state, tp)
end

function Options{S,T}(y::AbstractArray{S}, x::AbstractArray{T}, tp::Tape = Tape())
    state = (similar(y, Tracked{S,S}), similar(x, Tracked{T,S}))
    return _Options(state, tp)
end

function Options{S}(y::AbstractArray{S}, xs::Tuple, tp::Tape = Tape())
    state = (similar(y, Tracked{S,S}), map(x -> similar(x, Tracked{eltype(x),S}), xs))
    return _Options(state, tp)
end

Options(out::DiffResult, args...) = Options(DiffBase.value(out), args...)

##################
# HessianOptions #
##################

immutable HessianOptions{G<:Options,J<:Options} <: AbstractOptions
    gradient_options::G
    jacobian_options::J
end

HessianOptions(x, gtp::Tape = Tape(), jtp::Tape = Tape()) = HessianOptions(x, eltype(x), gtp, jtp)

function HessianOptions{A}(x, ::Type{A}, gtp::Tape = Tape(), jtp::Tape = Tape())
    jopts = Options(x, A, jtp)
    gopts = Options(jopts.state, gtp)
    return HessianOptions(gopts, jopts)
end

function HessianOptions(y, x, gtp::Tape = Tape(), jtp::Tape = Tape())
    jopts = Options(y, x, jtp)
    gopts = Options(jopts.state[2], gtp)
    return HessianOptions(gopts, jopts)
end

HessianOptions(out::DiffResult, args...) = HessianOptions(DiffBase.gradient(out), args...)

gradient_options(opts::HessianOptions) = opts.gradient_options
jacobian_options(opts::HessianOptions) = opts.jacobian_options
