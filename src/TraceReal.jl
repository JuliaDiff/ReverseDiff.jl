#############
# TraceReal #
#############

immutable TraceReal{T<:Real,S<:Real,P} <: Real
    dual::Dual{P,T}
    adjoint::RefValue{S}
    parents::Tuple{Vararg{TraceReal}}
    parent_adjoints::NTuple{P,RefValue{S}}
    TraceReal(dual) = new(dual, RefValue(zero(S)), tuple(), tuple())
    TraceReal(dual, parents...) = new(dual, RefValue(zero(S)), parents, get_parent_adjoints(parents...))
end

Base.show(io::IO, t::TraceReal) = print(io, "TraceReal($(t.dual), $(t.adjoint), $(length(t.parents)) parents)")

function TraceReal{P,T}(dual::Dual{P,T}, parents...)
    S = parent_adjtype(parents...)
    return TraceReal{T,S,P}(dual, parents...)
end

parent_adjtype{T,S}(::TraceReal{T,S}, args...) = S
get_parent_adjoints(a) = tuple(a.adjoint)
get_parent_adjoints(a, b) = tuple(a.adjoint, b.adjoint)
get_parent_adjoints(a, b, c) = tuple(a.adjoint, b.adjoint, c.adjoint)

value(t::TraceReal) = value(t.dual)
partials(t::TraceReal, args...) = partials(t.dual, args...)

########################
# Conversion/Promotion #
########################

function Base.convert{T1,T2,S,P1,P2}(::Type{TraceReal{T1,S,P1}}, t::TraceReal{T2,S,P2})
    if P1 > P2
        # The strategy here is to fill the excess parent slots with dummy values that
        # serve as terminal nodes in the graph during backpropagation, while maintaining
        # the trace for the original parents.
        leftover = P1 - P2
        dual::Dual{P1,T1} = Dual(T1(value(t)), partials(t)..., zeros(T1, leftover)...)
        parents = (t.parents..., fill(TraceReal{T1,S,0}(zero(T1)), leftover)...)
        return TraceReal{T1,S,P1}(dual, parents...)
    else
        # is there a way to reasonably do this without erroring?
        error("cannot convert to TraceReal with fewer parents than original value ($P1 parents < $P2 parents)")
    end
end

Base.convert{T,S,P}(::Type{TraceReal{T,S,P}}, x::Real) = TraceReal{T,S,P}(Dual{P,T}(x))
Base.convert{T1,T2,S,P}(::Type{TraceReal{T1,S,P}}, t::TraceReal{T2,S,P}) = TraceReal{T1,S,P}(Dual{T1,P}(t.dual), t.parents)
Base.convert{T,S,P}(::Type{TraceReal{T,S,P}}, t::TraceReal{T,S,P}) = t

Base.promote_rule{T1<:Real,T2,S,P}(::Type{T1}, ::Type{TraceReal{T2,S,P}}) = TraceReal{promote_type(T1,T2),S,P}

@generated function Base.promote_rule{T1,T2,S,P1,P2}(::Type{TraceReal{T1,S,P1}}, ::Type{TraceReal{T2,S,P2}})
    T = promote_type(T1,T2)
    P = ifelse(P1 > P2, P1, P2)
    return :(TraceReal{$T,S,$P})
end

Base.promote_array_type{T<:TraceReal, A<:AbstractFloat}(F, ::Type{T}, ::Type{A}) = promote_type(T, A)
Base.promote_array_type{T<:TraceReal, A<:AbstractFloat, P}(F, ::Type{T}, ::Type{A}, ::Type{P}) = P
Base.promote_array_type{A<:AbstractFloat, T<:TraceReal}(F, ::Type{A}, ::Type{T}) = promote_type(T, A)
Base.promote_array_type{A<:AbstractFloat, T<:TraceReal, P}(F, ::Type{A}, ::Type{T}, ::Type{P}) = P

Base.float{T,S,P}(t::TraceReal{T,S,P}) = TraceReal{promote_type(T, Float16),S,P}(t)

Base.one{T}(t::TraceReal{T}) = TraceReal(Dual(one(T), one(T)), t)
Base.zero{T}(t::TraceReal{T}) = TraceReal(Dual(zero(T), one(T)), t)

####################
# Math Overloading #
####################

# unary functions #
#-----------------#

for f in (ForwardDiff.AUTO_DEFINED_UNARY_FUNCS..., :-, :abs, :conj)
    @eval begin
        @inline Base.$(f){T}(t::TraceReal{T}) = TraceReal($(f)(Dual(value(t), one(T))), t)
    end
end

# binary functions #
#------------------#

const REAL_DEF_TYPES = (:Bool, :Integer, :Rational, :Real)

for f in (:*, :/, :+, :-, :^, :atan2)
    @eval begin
        @inline function Base.$(f){S,B}(a::TraceReal{S}, b::TraceReal{B})
            dual_a = Dual(value(a), one(S), zero(S))
            dual_b = Dual(value(b), zero(B), one(B))
            return TraceReal($(f)(dual_a, dual_b), a, b)
        end
    end
    for R in REAL_DEF_TYPES
        @eval begin
            @inline Base.$(f){T,S}(x::$R, t::TraceReal{T,S}) = TraceReal($(f)(x, Dual(value(t), one(T))), t)
            @inline Base.$(f){T,S}(t::TraceReal{T,S}, x::$R) = TraceReal($(f)(Dual(value(t), one(T)), x), t)
        end
    end
end

for f in (:<, :>, :(==), :(<=), :(>=))
    @eval begin
        @inline Base.$(f)(a::TraceReal, b::TraceReal) = $(f)(a.dual, b.dual)
    end
    for R in REAL_DEF_TYPES
        @eval begin
            @inline Base.$(f){T,S}(x::$R, t::TraceReal{T,S}) = $(f)(x, t.dual)
            @inline Base.$(f){T,S}(t::TraceReal{T,S}, x::$R) = $(f)(t.dual, x)
        end
    end
end
