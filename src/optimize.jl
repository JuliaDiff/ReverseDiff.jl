#############
# utilities #
#############

const REAL_DEF_TYPES = (:Bool, :Integer, :Rational, :Real, :Dual)
const FORWARD_UNARY_SCALAR_FUNCS = (ForwardDiff.AUTO_DEFINED_UNARY_FUNCS..., :-, :abs, :conj)
const FORWARD_BINARY_SCALAR_FUNCS = (:*, :/, :+, :-, :^, :atan2)
const SKIP_BINARY_SCALAR_FUNCS = (:<, :>, :(==), :(<=), :(>=))

#=
works for the following formats:
- `@forward(f)(args...)`
- `@forward f(args...) = ...`
- `@forward f = (args...) -> ...`
=#
function annotate_func_expr(typesym, expr)
    if expr.head == :(=)
        lhs = expr.args[1]
        if isa(lhs, Expr) && lhs.head == :call # named function definition site
            name = lhs.args[1]
            hidden_name = gensym(name)
            lhs.args[1] = hidden_name
            return quote
                $expr
                @inline function $(name)(args...)
                    return ReverseDiffPrototype.$(typesym)($(hidden_name))(args...)
                end
            end
        elseif isa(lhs, Symbol) # variable assignment site
            expr.args[2] = :(ReverseDiffPrototype.$(typesym)($(expr.args[2])))
            return expr
        else
            error("failed to apply $typesym to expression $expr")
        end
    else # call site
        return :(ReverseDiffPrototype.$(typesym)($expr))
    end
end

#############################
# Forward-Mode Optimization #
#############################

immutable ForwardOptimize{F}
    f::F
end


macro forward(ex)
    return esc(annotate_func_expr(:ForwardOptimize, ex))
end

# fallback #
#----------#

@inline (self::ForwardOptimize{F}){F}(args...) = self.f(args...)

# unary #
#-------#

for f in FORWARD_UNARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(t::TraceReal) = ForwardOptimize($f)(t)
end

@inline function (self::ForwardOptimize{F}){F,S}(t::TraceReal{S})
    dual = self.f(Dual(value(t), one(valtype(t))))
    tr = trace(t)
    out = TraceReal{S}(value(dual), tr)
    record!(tr, nothing, t, out, partials(dual))
    return out
end

# binary #
#--------#

for f in FORWARD_BINARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(a::TraceReal, b::TraceReal) = ForwardOptimize($f)(a, b)
    for R in REAL_DEF_TYPES
        @eval begin
            @inline Base.$(f)(a::TraceReal, b::$R) = ForwardOptimize($f)(a, b)
            @inline Base.$(f)(a::$R, b::TraceReal) = ForwardOptimize($f)(a, b)
        end
    end
end

@inline function (self::ForwardOptimize{F}){F,S}(a::TraceReal{S}, b::TraceReal{S})
    A, B = valtype(a), valtype(b)
    dual_a = Dual(value(a), one(A), zero(A))
    dual_b = Dual(value(b), zero(B), one(B))
    dual_c = self.f(dual_a, dual_b)
    tr = trace(a, b)
    out = TraceReal{S}(value(dual_c), tr)
    record!(tr, nothing, (a, b), out, partials(dual_c))
    return out
end

@inline function (self::ForwardOptimize{F}){F,S}(x::Real, t::TraceReal{S})
    dual = self.f(x, Dual(value(t), one(valtype(t))))
    tr = trace(t)
    out = TraceReal{S}(value(dual), tr)
    record!(tr, nothing, t, out, partials(dual))
    return out
end

@inline function (self::ForwardOptimize{F}){F,S}(t::TraceReal{S}, x::Real)
    dual = self.f(Dual(value(t), one(valtype(t))), x)
    tr = trace(t)
    out = TraceReal{S}(value(dual), tr)
    record!(tr, nothing, t, out, partials(dual))
    return out
end

#####################################################
# Higher-Order Optimizations (map, broadcast, etc.) #
#####################################################

function dual_unwrap{R<:TraceReal,N}(arr::AbstractArray{R}, ::Type{Val{N}}, i)
    T = valtype(R)
    out_partials = Partials{N,T}((z = zeros(T,N); z[i] = one(T); (z...)))
    out = similar(arr, Dual{N,T})
    for j in eachindex(out)
        out[j] = Dual{N,T}(value(arr[j]), out_partials)
    end
    return out
end

function dual_rewrap{S,N,T}(::Type{S}, duals::AbstractArray{Dual{N,T}}, tr::Nullable{Trace})
    ts = similar(duals, TraceReal{S,T})
    ps = similar(duals, Partials{N,T})
    for i in eachindex(duals)
        dual = duals[i]
        ts[i] = TraceReal{S}(value(dual), tr)
        ps[i] = partials(dual)
    end
    return ts, ps
end

for g in (:map, :broadcast)
    @eval begin
        # 1 arg
        function Base.$(g){F,S,T,N}(fopt::ForwardOptimize{F},
                                    x::AbstractArray{TraceReal{S,T},N})
            duals = $(g)(fopt.f, dual_unwrap(x, Val{1}, 1))
            tr = trace(x)
            out, partials = dual_rewrap(S, duals, tr)
            record!(tr, nothing, x, out, partials)
            return out
        end

        # 2 args
        function Base.$(g){F,S,T1,T2,N}(fopt::ForwardOptimize{F},
                                        x1::AbstractArray{TraceReal{S,T1},N},
                                        x2::AbstractArray{TraceReal{S,T2},N})
            dual1 = dual_unwrap(x1, Val{2}, 1)
            dual2 = dual_unwrap(x2, Val{2}, 2)
            duals = $(g)(fopt.f, dual1, dual2)
            tr = trace(x1, x2)
            out, partials = dual_rewrap(S, duals, tr)
            record!(tr, nothing, (x1, x2), out, partials)
            return out
        end

        # 3 args
        function Base.$(g){F,S,T1,T2,T3,N}(fopt::ForwardOptimize{F},
                                           x1::AbstractArray{TraceReal{S,T1},N},
                                           x2::AbstractArray{TraceReal{S,T2},N},
                                           x3::AbstractArray{TraceReal{S,T3},N})
            dual1 = dual_unwrap(x1, Val{3}, 1)
            dual2 = dual_unwrap(x2, Val{3}, 2)
            dual3 = dual_unwrap(x3, Val{3}, 3)
            duals = $(g)(fopt.f, dual1, dual2, dual3)
            tr = trace(x1, x2, x3)
            out, partials = dual_rewrap(S, duals, tr)
            record!(tr, nothing, (x1, x2, x3), out, partials)
            return out
        end
    end
end

##########################
# Skip Node Optimization #
##########################

immutable SkipOptimize{F}
    f::F
end

macro skip(ex)
    return esc(annotate_func_expr(:SkipOptimize, ex))
end

# fallback #
#----------#

@inline (self::SkipOptimize{F}){F}(args...) = self.f(args...)

# unary #
#-------#

@inline (self::SkipOptimize{F}){F}(a::TraceReal) = self.f(value(a))

# binary #
#--------#

@inline (self::SkipOptimize{F}){F}(a::TraceReal, b::TraceReal) = self.f(value(a), value(b))
@inline (self::SkipOptimize{F}){F}(a, b::TraceReal) = self.f(a, value(b))
@inline (self::SkipOptimize{F}){F}(a::TraceReal, b) = self.f(value(a), b)

for f in SKIP_BINARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(a::TraceReal, b::TraceReal) = SkipOptimize($(f))(a, b)
    for R in REAL_DEF_TYPES
        @eval begin
            @inline Base.$(f)(a::$R, b::TraceReal) = SkipOptimize($(f))(a, b)
            @inline Base.$(f)(a::TraceReal, b::$R) = SkipOptimize($(f))(a, b)
        end
    end
end

#######################
# Array Optimizations #
#######################

# unary #
#-------#

for f in (:-, :inv, :det)
    for A in (:AbstractArray, :AbstractMatrix, :Array, :Matrix)
        @eval function Base.$(f){S,T}(x::$(A){TraceReal{S,T}})
            tr = trace(x)
            out = wrap(S, $(f)(value(x)), tr)
            record!(tr, $(f), x, out)
            return out
        end
    end
end

for A in (:AbstractArray, :Array)
    @eval function Base.sum{S,T}(x::$(A){TraceReal{S,T}})
        result = zero(T)
        for t in x
            result += value(t)
        end
        tr = trace(x)
        out = TraceReal{S}(result, tr)
        record!(tr, sum, x, out)
        return out
    end
end

# binary #
#--------#

for f in (:-, :+, :*,
          :A_mul_Bt, :At_mul_B, :At_mul_Bt,
          :A_mul_Bc, :Ac_mul_B, :Ac_mul_Bc)
    @eval function Base.$(f){S,A,B}(a::AbstractMatrix{TraceReal{S,A}},
                                    b::AbstractMatrix{TraceReal{S,B}})
        tr = trace(a, b)
        out = wrap(S, $(f)(value(a), value(b)), tr)
        record!(tr, $(f), (a, b), out)
        return out
    end
end

# in-place A_mul_B family #
#-------------------------#

for (f!, f) in ((:A_mul_B!, :*),
                (:A_mul_Bt!, :A_mul_Bt), (:At_mul_B!, :At_mul_B), (:At_mul_Bt!, :At_mul_Bt),
                (:A_mul_Bc!, :A_mul_Bc), (:Ac_mul_B!, :Ac_mul_B), (:Ac_mul_Bc!, :Ac_mul_Bc))
    @eval function Base.$(f!){S,Y,A,B}(out::AbstractMatrix{TraceReal{S,Y}},
                                       a::AbstractMatrix{TraceReal{S,A}},
                                       b::AbstractMatrix{TraceReal{S,B}})
        tr = trace(a, b)
        wrap!(out, $(f)(value(a), value(b)), tr)
        record!(tr, $(f), (a, b), out)
        return out
    end
end
