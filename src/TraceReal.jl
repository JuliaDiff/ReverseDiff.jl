#############
# TraceReal #
#############

immutable TraceReal{tag,S<:Real,T<:Real} <: Real
    adjoint::RefValue{S}
    value::T
end

@inline adjoint(t::TraceReal) = t.adjoint[]
@inline value{tag,S,T}(t::TraceReal{tag,S,T}) = t.value
@inline value{tag,S,N,T}(t::TraceReal{tag,S,Dual{N,T}}) = value(t.value)

tagtype(t::TraceReal) = tagtype(typeof(t))
adjtype(t::TraceReal) = adjtype(typeof(t))
valtype(t::TraceReal) = valtype(typeof(t))

tagtype{tag,S,T}(::Type{TraceReal{tag,S,T}}) = tag
adjtype{tag,S,T}(::Type{TraceReal{tag,S,T}}) = S
valtype{tag,S,T}(::Type{TraceReal{tag,S,T}}) = T
valtype{tag,S,N,T}(::Type{TraceReal{tag,S,Dual{N,T}}}) = T

########################
# Conversion/Promotion #
########################

Base.convert{tag,S,T}(::Type{TraceReal{tag,S}}, x::T) = TraceReal{tag,S,T}(x)
Base.convert{tag,S,T}(::Type{TraceReal{tag,S,T}}, x::Real) = TraceReal{tag,S,T}(Ref(zero(S)), convert(T, x))
Base.convert{tag,S,T}(::Type{TraceReal{tag,S,T}}, t::TraceReal{tag,S}) = TraceReal{tag,S,T}(t.adjoint, value(t))
Base.convert{T<:TraceReal}(::Type{T}, t::T) = t

Base.promote_rule{R<:Real,tag,S,T}(::Type{R}, ::Type{TraceReal{tag,S,T}}) = TraceReal{tag,S,promote_type(R,T)}
Base.promote_rule{tag,S,A,B}(::Type{TraceReal{tag,S,A}}, ::Type{TraceReal{tag,S,B}}) = TraceReal{tag,S,promote_type(A,B)}

Base.promote_array_type{T<:TraceReal, A<:AbstractFloat}(_, ::Type{T}, ::Type{A}) = promote_type(T, A)
Base.promote_array_type{T<:TraceReal, A<:AbstractFloat, P}(_, ::Type{T}, ::Type{A}, ::Type{P}) = P
Base.promote_array_type{A<:AbstractFloat, T<:TraceReal}(_, ::Type{A}, ::Type{T}) = promote_type(T, A)
Base.promote_array_type{A<:AbstractFloat, T<:TraceReal, P}(_, ::Type{A}, ::Type{T}, ::Type{P}) = P

Base.float{tag,S,T}(::Type{TraceReal{tag,S,T}}) = TraceReal{tag,S,T}(float(T))
Base.one{tag,S,T}(::Type{TraceReal{tag,S,T}}) = TraceReal{tag,S,T}(one(T))
Base.zero{tag,S,T}(::Type{TraceReal{tag,S,T}}) = TraceReal{tag,S,T}(zero(T))
Base.rand{tag,S,T}(::Type{TraceReal{tag,S,T}}) = TraceReal{tag,S,T}(rand(T))
Base.rand{tag,S,T}(rng::AbstractRNG, ::Type{TraceReal{tag,S,T}}) = TraceReal{tag,S,T}(rand(rng, T))

####################
# Math Overloading #
####################

# unary functions #
#-----------------#

for f in (ForwardDiff.AUTO_DEFINED_UNARY_FUNCS..., :-, :abs, :conj)
    @eval begin
        @inline function Base.$(f){tag,S}(t::TraceReal{tag,S})
            out = TraceReal{tag,S}($(f)(Dual(value(t), one(valtype(t)))))
            record!(tag, t, out)
            return out
        end
    end
end

# binary functions #
#------------------#

const REAL_DEF_TYPES = (:Bool, :Integer, :Rational, :Real, :Dual)

for f in (:*, :/, :+, :-, :^, :atan2)
    @eval begin
        @inline function Base.$(f){tag,S}(a::TraceReal{tag,S}, b::TraceReal{tag,S})
            A, B = valtype(a), valtype(b)
            dual_a = Dual(value(a), one(A), zero(A))
            dual_b = Dual(value(b), zero(B), one(B))
            out = TraceReal{tag,S}($(f)(dual_a, dual_b))
            record!(tag, (a, b), out)
            return out
        end
    end
    for R in REAL_DEF_TYPES
        xexpr = R == :Dual ? :(value(x)) : :x
        @eval begin
            @inline function Base.$(f){tag,S}(x::$R, t::TraceReal{tag,S})
                out = TraceReal{tag,S}($(f)($(xexpr), Dual(value(t), one(valtype(t)))))
                record!(tag, t, out)
                return out
            end

            @inline function Base.$(f){tag,S}(t::TraceReal{tag,S}, x::$R)
                out = TraceReal{tag,S}($(f)(Dual(value(t), one(valtype(t))), $(xexpr)))
                record!(tag, t, out)
                return out
            end
        end
    end
end

for f in (:<, :>, :(==), :(<=), :(>=))
    @eval begin
        @inline Base.$(f)(a::TraceReal, b::TraceReal) = $(f)(value(a), value(b))
    end
    for R in REAL_DEF_TYPES
        xexpr = R == :Dual ? :(value(x)) : :x
        @eval begin
            @inline Base.$(f)(x::$R, t::TraceReal) = $(f)($(xexpr), value(t))
            @inline Base.$(f)(t::TraceReal, x::$R) = $(f)(value(t), $(xexpr))
        end
    end
end

###################
# Pretty Printing #
###################

Base.show{tag}(io::IO, t::TraceReal{tag}) = print(io, "TraceReal{$tag}($(adjoint(t)), $(t.value))")
