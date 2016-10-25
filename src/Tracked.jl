###########
# Tracked #
###########

type Tracked{V<:Real,A<:Real} <: Real
    value::V
    adjoint::A
    tape::Nullable{Tape}
end

Tracked{V,A}(x::V, ::Type{A} = V, tp::Nullable{Tape} = Nullable{Tape}()) = Tracked{V,A}(x, zero(A), tp)
Tracked{V}(x::V, tp::Nullable{Tape}) = Tracked(x, V, tp)

@inline adjoint(t::Tracked) = t.adjoint

@inline value(x::Real) = x
@inline value(d::Dual) = ForwardDiff.value(d)
@inline value{V,A}(t::Tracked{V,A}) = t.value

adjtype(t::Tracked) = adjtype(typeof(t))
adjtype{V,A}(::Type{Tracked{V,A}}) = A

valtype(t::Tracked) = valtype(typeof(t))
valtype{V,A}(::Type{Tracked{V,A}}) = V

###################
# tape selection #
###################

@inline hastape(x::Real) = false
@inline hastape(t::Tracked) = !(isnull(tape(t)))

@inline tape(x::Real) = Nullable{Tape}()
@inline tape(t::Tracked) = t.tape
@inline tape(a::Tracked, b::Tracked) = ifelse(isnull(tape(a)), tape(b), tape(a))

########################
# Conversion/Promotion #
########################

Base.convert{R<:Real}(::Type{R}, t::Tracked) = R(value(t))
Base.convert{V,A}(::Type{Tracked{V,A}}, x::Real) = Tracked(V(value(x)), A)
Base.convert{V,A}(::Type{Tracked{V,A}}, t::Tracked) = Tracked(V(value(t)), A(adjoint(t)), t.tape)
Base.convert{T<:Tracked}(::Type{T}, t::T) = t

Base.promote_rule{R<:Real,V,A}(::Type{R}, ::Type{Tracked{V,A}}) = Tracked{promote_type(R,V),A}
Base.promote_rule{V1,V2,A1,A2}(::Type{Tracked{V1,A1}}, ::Type{Tracked{V2,A2}}) = Tracked{promote_type(V1,V2),promote_type(A1,A2)}

Base.promote_array_type{T<:Tracked, F<:AbstractFloat}(_, ::Type{T}, ::Type{F}) = promote_type(T, F)
Base.promote_array_type{T<:Tracked, F<:AbstractFloat, P}(_, ::Type{T}, ::Type{F}, ::Type{P}) = P
Base.promote_array_type{F<:AbstractFloat, T<:Tracked}(_, ::Type{F}, ::Type{T}) = promote_type(T, F)
Base.promote_array_type{F<:AbstractFloat, T<:Tracked, P}(_, ::Type{F}, ::Type{T}, ::Type{P}) = P

####################
# `Real` Interface #
####################

Base.hash(t::Tracked) = hash(value(t))
Base.hash(t::Tracked, hsh::UInt64) = hash(value(t), hsh)

Base.deepcopy{T<:Tracked}(t::T) = t
Base.copy{T<:Tracked}(t::T) = t

Base.float{V,A}(t::Tracked{V,A}) = Tracked(float(value(t)), A)
Base.float{V,A<:AbstractFloat}(t::Tracked{V,A}) = t

Base.one{V,A}(::Type{Tracked{V,A}}) = Tracked(one(V), A)
Base.zero{V,A}(::Type{Tracked{V,A}}) = Tracked(zero(V), A)

Base.rand{V,A}(::Type{Tracked{V,A}}) = Tracked(rand(V), A)
Base.rand{V,A}(rng::AbstractRNG, ::Type{Tracked{V,A}}) = Tracked(rand(rng, V), A)

Base.eps(t::Tracked) = eps(value(t))
Base.eps{T<:Tracked}(::Type{T}) = eps(valtype(T))

Base.floor(t::Tracked) = floor(value(t))
Base.floor{R<:Real}(::Type{R}, t::Tracked) = floor(R, value(t))

Base.ceil(t::Tracked) = ceil(value(t))
Base.ceil{R<:Real}(::Type{R}, t::Tracked) = ceil(R, value(t))

Base.trunc(t::Tracked) = trunc(value(t))
Base.trunc{R<:Real}(::Type{R}, t::Tracked) = trunc(R, value(t))

Base.round(t::Tracked) = round(value(t))
Base.round{R<:Real}(::Type{R}, t::Tracked) = round(R, value(t))

###################
# Pretty Printing #
###################

Base.show(io::IO, t::Tracked) = print(io, "Tracked($(value(t)), $(adjoint(t))$(isnull(tape(t)) ? ",␀" : ""))")
