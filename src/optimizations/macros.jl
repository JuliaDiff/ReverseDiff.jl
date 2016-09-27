#=
works for the following formats:
- `@forward(f)(args...)`
- `@forward f(args...) = ...`
- `@forward f = (args...) -> ...`
=#
function annotate_func_expr(typesym, expr)
    if isa(expr, Expr) && expr.head == :(=)
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

@inline function (self::ForwardOptimize{F}){F,S}(t::Tracer{S})
    dual = self.f(Dual(value(t), one(valtype(t))))
    tp = tape(t)
    out = Tracer{S}(value(dual), tp)
    record!(tp, nothing, t, out, partials(dual))
    return out
end

# binary #
#--------#

@inline function (self::ForwardOptimize{F}){F,S}(a::Tracer{S}, b::Tracer{S})
    A, B = valtype(a), valtype(b)
    dual_a = Dual(value(a), one(A), zero(A))
    dual_b = Dual(value(b), zero(B), one(B))
    dual_c = self.f(dual_a, dual_b)
    tp = tape(a, b)
    out = Tracer{S}(value(dual_c), tp)
    record!(tp, nothing, (a, b), out, partials(dual_c))
    return out
end

@inline function (self::ForwardOptimize{F}){F,S}(x::Real, t::Tracer{S})
    dual = self.f(x, Dual(value(t), one(valtype(t))))
    tp = tape(t)
    out = Tracer{S}(value(dual), tp)
    record!(tp, nothing, t, out, partials(dual))
    return out
end

@inline function (self::ForwardOptimize{F}){F,S}(t::Tracer{S}, x::Real)
    dual = self.f(Dual(value(t), one(valtype(t))), x)
    tp = tape(t)
    out = Tracer{S}(value(dual), tp)
    record!(tp, nothing, t, out, partials(dual))
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

@inline (self::SkipOptimize{F}){F}(a::Tracer) = self.f(value(a))

# binary #
#--------#

@inline (self::SkipOptimize{F}){F}(a::Tracer, b::Tracer) = self.f(value(a), value(b))
@inline (self::SkipOptimize{F}){F}(a, b::Tracer) = self.f(a, value(b))
@inline (self::SkipOptimize{F}){F}(a::Tracer, b) = self.f(value(a), b)
