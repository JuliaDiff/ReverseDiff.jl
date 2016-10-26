#=
works for the following formats:
- `@forward(f)(args...)`
- `@forward f(args...) = ...`
- `@forward f = (args...) -> ...`
=#
function annotate_func_expr(typesym, expr)
    if isa(expr, Expr) && (expr.head == :(=) || expr.head == :function)
        lhs = expr.args[1]
        if isa(lhs, Expr) && lhs.head == :call # named function definition site
            name_and_types = lhs.args[1]
            args_signature = lhs.args[2:end]
            old_name_and_types = deepcopy(name_and_types)
            if isa(name_and_types, Expr) && name_and_types.head == :curly
                name = name_and_types.args[1]
                hidden_name = gensym(name)
                name_and_types.args[1] = hidden_name

            elseif isa(name_and_types, Symbol)
                name = name_and_types
                hidden_name = gensym(name)
                lhs.args[1] = hidden_name
            else
                error("potentially malformed function signature: $(signature)")
            end
            return quote
                $expr
                @inline function $(old_name_and_types)($(args_signature...))
                    return ReverseDiffPrototype.$(typesym)($(hidden_name))($(args_signature...))
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

@inline function (self::ForwardOptimize{F}){F,V,A}(t::Tracked{V,A})
    dual = self.f(Dual(value(t), one(V)))
    tp = tape(t)
    out = Tracked(value(dual), A, tp)
    record!(tp, nothing, t, out, ForwardDiff.partials(dual))
    return out
end

# binary #
#--------#

@inline function (self::ForwardOptimize{F}){F,V1,V2,A}(a::Tracked{V1,A}, b::Tracked{V2,A})
    dual_a = Dual(value(a), one(V1), zero(V1))
    dual_b = Dual(value(b), zero(V2), one(V2))
    dual_c = self.f(dual_a, dual_b)
    tp = tape(a, b)
    out = Tracked(value(dual_c), A, tp)
    record!(tp, nothing, (a, b), out, ForwardDiff.partials(dual_c))
    return out
end

@inline function (self::ForwardOptimize{F}){F,V,A}(x::Real, t::Tracked{V,A})
    dual = self.f(x, Dual(value(t), one(V)))
    tp = tape(t)
    out = Tracked(value(dual), A, tp)
    record!(tp, nothing, t, out, ForwardDiff.partials(dual))
    return out
end

@inline function (self::ForwardOptimize{F}){F,V,A}(t::Tracked{V,A}, x::Real)
    dual = self.f(Dual(value(t), one(V)), x)
    tp = tape(t)
    out = Tracked(value(dual), A, tp)
    record!(tp, nothing, t, out, ForwardDiff.partials(dual))
    return out
end

@inline (self::ForwardOptimize{F}){F}(x::Dual, t::Tracked) = invoke(self.f, (Dual, Real), x, t)
@inline (self::ForwardOptimize{F}){F}(t::Tracked, x::Dual) = invoke(self.f, (Real, Dual), t, x)

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

@inline (self::SkipOptimize{F}){F}(a::Tracked) = self.f(value(a))

# binary #
#--------#

@inline (self::SkipOptimize{F}){F}(a::Tracked, b::Tracked) = self.f(value(a), value(b))
@inline (self::SkipOptimize{F}){F}(a, b::Tracked) = self.f(a, value(b))
@inline (self::SkipOptimize{F}){F}(a::Tracked, b) = self.f(value(a), b)
