#############
# TraceReal #
#############

immutable TraceReal{S<:Real,T<:Real} <: Real
    adjoint::RefValue{S}
    value::T
    trace::Nullable{Trace}
end

(::Type{TraceReal{S}}){S,T}(x::T, tr::Nullable{Trace} = Nullable{Trace}()) = TraceReal{S,T}(RefValue(zero(S)), x, tr)

@inline adjoint(t::TraceReal) = t.adjoint[]
@inline ForwardDiff.value{S,T}(t::TraceReal{S,T}) = t.value

adjtype(t::TraceReal) = adjtype(typeof(t))
ForwardDiff.valtype(t::TraceReal) = valtype(typeof(t))

adjtype{S,T}(::Type{TraceReal{S,T}}) = S
ForwardDiff.valtype{S,T}(::Type{TraceReal{S,T}}) = T

###################
# trace selection #
###################

@inline trace(t::TraceReal) = t.trace
@inline trace(a::TraceReal, b::TraceReal) = ifelse(isnull(trace(a)), trace(b), trace(a))

########################
# Conversion/Promotion #
########################

Base.convert{R<:Real}(::Type{R}, t::TraceReal) = R(value(t))
Base.convert{S,T}(::Type{TraceReal{S,T}}, x::Real) = TraceReal{S}(T(x))
Base.convert{S,T}(::Type{TraceReal{S,T}}, t::TraceReal{S}) = TraceReal{S,T}(t.adjoint, T(value(t)), t.trace)
Base.convert{T<:TraceReal}(::Type{T}, t::T) = t

Base.promote_rule{R<:Real,S,T}(::Type{R}, ::Type{TraceReal{S,T}}) = TraceReal{S,promote_type(R,T)}
Base.promote_rule{S,A,B}(::Type{TraceReal{S,A}}, ::Type{TraceReal{S,B}}) = TraceReal{S,promote_type(A,B)}

Base.promote_array_type{T<:TraceReal, A<:AbstractFloat}(_, ::Type{T}, ::Type{A}) = promote_type(T, A)
Base.promote_array_type{T<:TraceReal, A<:AbstractFloat, P}(_, ::Type{T}, ::Type{A}, ::Type{P}) = P
Base.promote_array_type{A<:AbstractFloat, T<:TraceReal}(_, ::Type{A}, ::Type{T}) = promote_type(T, A)
Base.promote_array_type{A<:AbstractFloat, T<:TraceReal, P}(_, ::Type{A}, ::Type{T}, ::Type{P}) = P

Base.float{S,T}(::Type{TraceReal{S,T}}) = TraceReal{S}(float(T))
Base.one{S,T}(::Type{TraceReal{S,T}}) = TraceReal{S}(one(T))
Base.zero{S,T}(::Type{TraceReal{S,T}}) = TraceReal{S}(zero(T))
Base.rand{S,T}(::Type{TraceReal{S,T}}) = TraceReal{S}(rand(T))
Base.rand{S,T}(rng::AbstractRNG, ::Type{TraceReal{S,T}}) = TraceReal{S}(rand(rng, T))

####################
# Math Overloading #
####################

# unary functions #
#-----------------#

for f in (ForwardDiff.AUTO_DEFINED_UNARY_FUNCS..., :-, :abs, :conj)
    @eval begin
        @inline function Base.$(f){S}(t::TraceReal{S})
            dual = $(f)(Dual(value(t), one(valtype(t))))
            tr = trace(t)
            out = TraceReal{S}(value(dual), tr)
            record!(tr, nothing, t, out, partials(dual))
            return out
        end
    end
end

# binary functions #
#------------------#

const REAL_DEF_TYPES = (:Bool, :Integer, :Rational, :Real, :Dual)

for f in (:*, :/, :+, :-, :^, :atan2)
    @eval begin
        @inline function Base.$(f){S}(a::TraceReal{S}, b::TraceReal{S})
            A, B = valtype(a), valtype(b)
            dual_a = Dual(value(a), one(A), zero(A))
            dual_b = Dual(value(b), zero(B), one(B))
            dual_c = $(f)(dual_a, dual_b)
            tr = trace(a, b)
            out = TraceReal{S}(value(dual_c), tr)
            record!(tr, nothing, (a, b), out, partials(dual_c))
            return out
        end
    end
    for R in REAL_DEF_TYPES
        @eval begin
            @inline function Base.$(f){S}(x::$R, t::TraceReal{S})
                dual = $(f)(x, Dual(value(t), one(valtype(t))))
                tr = trace(t)
                out = TraceReal{S}(value(dual), tr)
                record!(tr, nothing, t, out, partials(dual))
                return out
            end

            @inline function Base.$(f){S}(t::TraceReal{S}, x::$R)
                dual = $(f)(Dual(value(t), one(valtype(t))), x)
                tr = trace(t)
                out = TraceReal{S}(value(dual), tr)
                record!(tr, nothing, t, out, partials(dual))
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
        @eval begin
            @inline Base.$(f)(x::$R, t::TraceReal) = $(f)(x, value(t))
            @inline Base.$(f)(t::TraceReal, x::$R) = $(f)(value(t), x)
        end
    end
end

###################
# Pretty Printing #
###################

Base.show(io::IO, t::TraceReal) = print(io, "TraceReal($(adjoint(t)),$(value(t))$(isnull(trace(t)) ? ",␀" : ""))")
