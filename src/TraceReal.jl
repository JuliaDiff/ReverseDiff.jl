#############
# TraceReal #
#############

immutable TraceReal{F,S<:Real,N,T<:Real} <: Real
    adjoint::RefValue{S}
    dual::Dual{N,T}
end

value(t::TraceReal) = value(t.dual)
partials(t::TraceReal, args...) = partials(t.dual, args...)
numtype{F,S,N,T}(::TraceReal{F,S,N,T}) = T
adjtype{F,S,N,T}(::TraceReal{F,S,N,T}) = S

########################
# Conversion/Promotion #
########################

Base.convert{F,S,N,T}(::Type{TraceReal{F,S}}, x::Dual{N,T}) = TraceReal{F,S,N,T}(RefValue{S}(zero(S)), x)
Base.convert{F,S,N,T}(::Type{TraceReal{F,S,N,T}}, x::Real) = TraceReal{F,S,N,T}(RefValue{S}(zero(S)), x)
Base.convert{F,S,N,T}(::Type{TraceReal{F,S,N,T}}, adjoint::RefValue{S}, x::Real) = TraceReal{F,S}(adjoint, Dual{N,T}(x, one(Partials{N,T})))
Base.convert{F,S,N,T}(::Type{TraceReal{F,S,N,T}}, t::TraceReal{F}) = TraceReal{F,S,N,T}(t.adjoint, value(t))
Base.convert{T<:TraceReal}(::Type{T}, t::T) = t

Base.promote_rule{R<:Real,F,S,N,T}(::Type{R}, ::Type{TraceReal{F,S,N,T}}) = TraceReal{F,S,N,promote_type(R,T)}
Base.promote_rule{F,S,N,A,B}(::Type{TraceReal{F,S,N,A}}, ::Type{TraceReal{F,S,N,B}}) = TraceReal{F,S,N,promote_type(A,B)}

Base.promote_array_type{T<:TraceReal, A<:AbstractFloat}(_, ::Type{T}, ::Type{A}) = promote_type(T, A)
Base.promote_array_type{T<:TraceReal, A<:AbstractFloat, P}(_, ::Type{T}, ::Type{A}, ::Type{P}) = P
Base.promote_array_type{A<:AbstractFloat, T<:TraceReal}(_, ::Type{A}, ::Type{T}) = promote_type(T, A)
Base.promote_array_type{A<:AbstractFloat, T<:TraceReal, P}(_, ::Type{A}, ::Type{T}, ::Type{P}) = P

Base.float{F,S,N,T}(t::TraceReal{F,S,N,T}) = TraceReal{F,S,N,promote_type(T,Float16)}(t)
Base.one{F,S,N,T}(t::TraceReal{F,S,N,T}) = TraceReal{F,S,N,T}(t.adjoint, one(T))
Base.zero{F,S,N,T}(t::TraceReal{F,S,N,T}) = TraceReal{F,S,N,T}(t.adjoint, zero(T))

####################
# Math Overloading #
####################

# unary functions #
#-----------------#

for f in (ForwardDiff.AUTO_DEFINED_UNARY_FUNCS..., :-, :abs, :conj)
    @eval begin
        @inline function Base.$(f){F,S}(t::TraceReal{F,S})
            dual = Dual(value(t), one(numtype(t)))
            result = TraceReal{F,S}($(f)(dual))
            record!(F, t, result)
            return result
        end
    end
end

# binary functions #
#------------------#

const REAL_DEF_TYPES = (:Bool, :Integer, :Rational, :Real)

for f in (:*, :/, :+, :-, :^, :atan2)
    @eval begin
        @inline function Base.$(f){F,S}(a::TraceReal{F,S}, b::TraceReal{F,S})
            A, B = numtype(a), numtype(b)
            dual_a = Dual(value(a), one(A), zero(A))
            dual_b = Dual(value(b), zero(B), one(B))
            result = TraceReal{F,S}($(f)(dual_a, dual_b))
            record!(F, (a, b), result)
            return result
        end
    end
    for R in REAL_DEF_TYPES
        @eval begin
            @inline function Base.$(f){F,S}(x::$R, t::TraceReal{F,S})
                dual = Dual(value(t), one(numtype(t)))
                result = TraceReal{F,S}($(f)(x, dual))
                record!(F, t, result)
                return result
            end

            @inline function Base.$(f){F,S}(t::TraceReal{F,S}, x::$R)
                dual = Dual(value(t), one(numtype(t)))
                result = TraceReal{F,S}($(f)(dual, x))
                record!(F, t, result)
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
