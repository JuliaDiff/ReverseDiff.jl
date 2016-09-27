##########
# Tracer #
##########

type Tracer{S<:Real,T<:Real} <: Real
    adjoint::S
    value::T
    tape::Nullable{Tape}
end

(::Type{Tracer{S}}){S,T}(x::T, tp::Nullable{Tape} = Nullable{Tape}()) = Tracer{S,T}(zero(S), x, tp)

@inline adjoint(t::Tracer) = t.adjoint
@inline ForwardDiff.value{S,T}(t::Tracer{S,T}) = t.value

adjtype(t::Tracer) = adjtype(typeof(t))
adjtype{S,T}(::Type{Tracer{S,T}}) = S

valtype(t::Tracer) = valtype(typeof(t))
valtype{S,T}(::Type{Tracer{S,T}}) = T

###################
# tape selection #
###################

@inline tape(t::Tracer) = t.tape
@inline tape(a::Tracer, b::Tracer) = ifelse(isnull(tape(a)), tape(b), tape(a))

########################
# Conversion/Promotion #
########################

Base.convert{R<:Real}(::Type{R}, t::Tracer) = R(value(t))
Base.convert{S,T}(::Type{Tracer{S,T}}, x::Real) = Tracer{S}(T(value(x)))
Base.convert{S,T}(::Type{Tracer{S,T}}, t::Tracer) = Tracer{S,T}(S(adjoint(t)), T(value(t)), t.tape)
Base.convert{T<:Tracer}(::Type{T}, t::T) = t

Base.promote_rule{R<:Real,S,T}(::Type{R}, ::Type{Tracer{S,T}}) = Tracer{S,promote_type(R,T)}
Base.promote_rule{S,A,B}(::Type{Tracer{S,A}}, ::Type{Tracer{S,B}}) = Tracer{S,promote_type(A,B)}

Base.promote_array_type{T<:Tracer, A<:AbstractFloat}(_, ::Type{T}, ::Type{A}) = promote_type(T, A)
Base.promote_array_type{T<:Tracer, A<:AbstractFloat, P}(_, ::Type{T}, ::Type{A}, ::Type{P}) = P
Base.promote_array_type{A<:AbstractFloat, T<:Tracer}(_, ::Type{A}, ::Type{T}) = promote_type(T, A)
Base.promote_array_type{A<:AbstractFloat, T<:Tracer, P}(_, ::Type{A}, ::Type{T}, ::Type{P}) = P

####################
# `Real` Interface #
####################

Base.copy(t::Tracer) = t

Base.float{S,T}(t::Tracer{S,T}) = Tracer{S}(float(value(t)))
Base.float{S,T<:AbstractFloat}(t::Tracer{S,T}) = t

Base.one{S,T}(::Type{Tracer{S,T}}) = Tracer{S}(one(T))
Base.zero{S,T}(::Type{Tracer{S,T}}) = Tracer{S}(zero(T))

Base.rand{S,T}(::Type{Tracer{S,T}}) = Tracer{S}(rand(T))
Base.rand{S,T}(rng::AbstractRNG, ::Type{Tracer{S,T}}) = Tracer{S}(rand(rng, T))

Base.eps(t::Tracer) = eps(value(t))
Base.eps{T<:Tracer}(::Type{T}) = eps(valtype(T))

Base.floor(t::Tracer) = floor(value(t))
Base.floor{T<:Real}(::Type{T}, t::Tracer) = floor(T, value(t))

Base.ceil(t::Tracer) = ceil(value(t))
Base.ceil{T<:Real}(::Type{T}, t::Tracer) = ceil(T, value(t))

Base.trunc(t::Tracer) = trunc(value(t))
Base.trunc{T<:Real}(::Type{T}, t::Tracer) = trunc(T, value(t))

Base.round(t::Tracer) = round(value(t))
Base.round{T<:Real}(::Type{T}, t::Tracer) = round(T, value(t))

###################
# Pretty Printing #
###################

Base.show(io::IO, t::Tracer) = print(io, "Tracer($(adjoint(t)),$(value(t))$(isnull(tape(t)) ? ",␀" : ""))")
