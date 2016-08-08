#############
# TraceReal #
#############

immutable TraceReal{tag,S<:Real,N,T<:Real} <: Real
    adjoint::RefValue{S}
    dual::Dual{N,T}
end

value(t::TraceReal) = value(t.dual)
partials(t::TraceReal, args...) = partials(t.dual, args...)

tagtype(t::TraceReal) = tagtype(typeof(t))
adjtype(t::TraceReal) = adjtype(typeof(t))
numtype(t::TraceReal) = numtype(typeof(t))

tagtype{tag,S,N,T}(::Type{TraceReal{tag,S,N,T}}) = tag
adjtype{tag,S,N,T}(::Type{TraceReal{tag,S,N,T}}) = S
numtype{tag,S,N,T}(::Type{TraceReal{tag,S,N,T}}) = T

########################
# Conversion/Promotion #
########################

Base.convert{tag,S,N,T}(::Type{TraceReal{tag,S}}, x::Dual{N,T}) = TraceReal{tag,S,N,T}(RefValue{S}(zero(S)), x)
Base.convert{tag,S,N,T}(::Type{TraceReal{tag,S,N,T}}, x::Real) = TraceReal{tag,S,N,T}(RefValue{S}(zero(S)), x)
Base.convert{tag,S,N,T}(::Type{TraceReal{tag,S,N,T}}, adjoint::RefValue{S}, x::Real) = TraceReal{tag,S}(adjoint, Dual{N,T}(x, one(Partials{N,T})))
Base.convert{tag,S,N,T}(::Type{TraceReal{tag,S,N,T}}, t::TraceReal{tag}) = TraceReal{tag,S,N,T}(t.adjoint, value(t))
Base.convert{T<:TraceReal}(::Type{T}, t::T) = t

Base.promote_rule{R<:Real,tag,S,N,T}(::Type{R}, ::Type{TraceReal{tag,S,N,T}}) = TraceReal{tag,S,N,promote_type(R,T)}
Base.promote_rule{tag,S,N,A,B}(::Type{TraceReal{tag,S,N,A}}, ::Type{TraceReal{tag,S,N,B}}) = TraceReal{tag,S,N,promote_type(A,B)}

Base.promote_array_type{T<:TraceReal, A<:AbstractFloat}(_, ::Type{T}, ::Type{A}) = promote_type(T, A)
Base.promote_array_type{T<:TraceReal, A<:AbstractFloat, P}(_, ::Type{T}, ::Type{A}, ::Type{P}) = P
Base.promote_array_type{A<:AbstractFloat, T<:TraceReal}(_, ::Type{A}, ::Type{T}) = promote_type(T, A)
Base.promote_array_type{A<:AbstractFloat, T<:TraceReal, P}(_, ::Type{A}, ::Type{T}, ::Type{P}) = P

Base.float{tag,S,N,T}(t::TraceReal{tag,S,N,T}) = TraceReal{tag,S,N,promote_type(T,Float16)}(t)
Base.one{tag,S,N,T}(t::TraceReal{tag,S,N,T}) = TraceReal{tag,S,N,T}(t.adjoint, one(T))
Base.zero{tag,S,N,T}(t::TraceReal{tag,S,N,T}) = TraceReal{tag,S,N,T}(t.adjoint, zero(T))

####################
# Math Overloading #
####################

# unary functions #
#-----------------#

for f in (ForwardDiff.AUTO_DEFINED_UNARY_FUNCS..., :-, :abs, :conj)
    @eval begin
        @inline function Base.$(f){tag,S}(t::TraceReal{tag,S})
            dual = Dual(value(t), one(numtype(t)))
            result = TraceReal{tag,S}($(f)(dual))
            record!(tag, t, result)
            return result
        end
    end
end

# binary functions #
#------------------#

const REAL_DEF_TYPES = (:Bool, :Integer, :Rational, :Real)

for f in (:*, :/, :+, :-, :^, :atan2)
    @eval begin
        @inline function Base.$(f){tag,S}(a::TraceReal{tag,S}, b::TraceReal{tag,S})
            A, B = numtype(a), numtype(b)
            dual_a = Dual(value(a), one(A), zero(A))
            dual_b = Dual(value(b), zero(B), one(B))
            result = TraceReal{tag,S}($(f)(dual_a, dual_b))
            record!(tag, (a, b), result)
            return result
        end
    end
    for R in REAL_DEF_TYPES
        @eval begin
            @inline function Base.$(f){tag,S}(x::$R, t::TraceReal{tag,S})
                dual = Dual(value(t), one(numtype(t)))
                result = TraceReal{tag,S}($(f)(x, dual))
                record!(tag, t, result)
                return result
            end

            @inline function Base.$(f){tag,S}(t::TraceReal{tag,S}, x::$R)
                dual = Dual(value(t), one(numtype(t)))
                result = TraceReal{tag,S}($(f)(dual, x))
                record!(tag, t, result)
                return result
            end
        end
    end
end

for f in (:<, :>, :(==), :(<=), :(>=))
    @eval begin
        @inline Base.$(f)(a::TraceReal, b::TraceReal) = $(f)(a.dual, b.dual)
    end
    for R in REAL_DEF_TYPES
        @eval begin
            @inline Base.$(f)(x::$R, t::TraceReal) = $(f)(x, t.dual)
            @inline Base.$(f)(t::TraceReal, x::$R) = $(f)(t.dual, x)
        end
    end
end
