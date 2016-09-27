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
function Options{T}(x::AbstractArray{T}, tp::Tape = Tape())
    state = similar(x, Tracked{T,T})
    return _Options(state, tp)
end

function Options{T,A}(x::AbstractArray{T}, ::Type{A}, tp::Tape = Tape())
    state = similar(x, Tracked{T,A})
    return _Options(state, tp)
end

function Options(xs::Tuple, tp::Tape = Tape())
    state = map(x -> similar(x, Tracked{eltype(x),eltype(x)}), xs)
    return _Options(state, tp)
end

function Options(xs::Tuple, types::Tuple{Vararg{DataType}}, tp::Tape = Tape())
    state = map((x, A) -> similar(x, Tracked{eltype(x),A}), xs, types)
    return _Options(state, tp)
end

###########
# Options #
###########

immutable HessianOptions{G<:Options,J<:Options} <: AbstractOptions
    gradient_options::G
    jacobian_options::J
end

function HessianOptions(x, gtp::Tape = Tape(), jtp::Tape = Tape())
    jopts = Options(x, jtp)
    gopts = Options(jopts.state, gtp)
    return HessianOptions(gopts, jopts)
end

function HessianOptions(out, x, gtp::Tape = Tape(), jtp::Tape = Tape())
    jopts = Options(x, eltype(out), jtp)
    gopts = Options(jopts.state, gtp)
    return HessianOptions(gopts, jopts)
end
