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

@inline function (self::ForwardOptimize{F}){F,S}(t::TraceReal{S})
    dual = self.f(Dual(value(t), one(valtype(t))))
    tr = trace(t)
    out = TraceReal{S}(value(dual), tr)
    record!(tr, nothing, t, out, partials(dual))
    return out
end

# binary #
#--------#

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
