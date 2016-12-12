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
                const $(old_name_and_types) = ReverseDiff.$(typesym)(
                    ($(args_signature...)) -> begin
                        $(Expr(:meta, :inline))
                        return $(hidden_name)($(args_signature...))
                    end
                )
            end
        elseif isa(lhs, Symbol) # variable assignment site
            expr.args[2] = :(ReverseDiff.$(typesym)($(expr.args[2])))
            return expr
        else
            error("failed to apply $typesym to expression $expr")
        end
    else # call site
        return :(ReverseDiff.$(typesym)($expr))
    end
end

#############################
# Forward-Mode Optimization #
#############################

immutable ForwardOptimize{F}
    f::F
end

@inline ForwardOptimize(f::ForwardOptimize) = f

"""
    ReverseDiff.@forward(f)(args::Real...)
    ReverseDiff.@forward f(args::Real...) = ...
    ReverseDiff.@forward f = (args::Real...) -> ...

Declare that the given function should be differentiated using forward mode automatic
differentiation. Note that the macro can be used at either the definition site or at the
call site of `f`. Currently, only `length(args) <= 2` is supported. **Note that, if `f` is
defined within another function `g`, `f` should not close over any differentiable input of
`g`.** By using this macro, you are providing a guarantee that this property holds true.

This macro can be very beneficial for performance when intermediate functions in your
computation are low dimensional scalar functions, because it minimizes the number of
instructions that must be recorded to the tape. For example, take the function
`sigmoid(n) = 1. / (1. + exp(-n))`. Normally, using ReverseDiff to differentiate this
function would require recording 4 instructions (`-`, `exp`, `+`, and `/`). However, if we
apply the `@forward` macro, only one instruction will be recorded (`sigmoid`). The `sigmoid`
function will then be differentiated using ForwardDiff's `Dual` number type.

This is also beneficial for higher-order elementwise function application. ReverseDiff
overloads `map`/`broadcast` to dispatch on `@forward`-applied functions. For example,
`map(@forward(f), x)` will usually be more performant than `map(f, x)`.

ReverseDiff overloads many Base scalar functions to behave as `@forward` functions by
default. A full list is given by `ReverseDiff.FORWARD_UNARY_SCALAR_FUNCS` and
`ReverseDiff.FORWARD_BINARY_SCALAR_FUNCS`.
"""
macro forward(ex)
    return esc(annotate_func_expr(:ForwardOptimize, ex))
end

# fallback #
#----------#

@inline (self::ForwardOptimize{F}){F}(args...) = self.f(args...)

# unary #
#-------#

@inline function (self::ForwardOptimize{F}){F,V,D}(t::TrackedReal{V,D})
    dual = self.f(Dual(value(t), one(V)))
    tp = tape(t)
    out = track(ForwardDiff.value(dual), D, tp)
    cache = RefValue(ForwardDiff.partials(dual, 1))
    record!(tp, ScalarInstruction, self.f, t, out, cache)
    return out
end

# binary #
#--------#

@inline function (self::ForwardOptimize{F}){F,V1,V2,D}(a::TrackedReal{V1,D}, b::TrackedReal{V2,D})
    dual_a = Dual(value(a), one(V1), zero(V1))
    dual_b = Dual(value(b), zero(V2), one(V2))
    dual_c = self.f(dual_a, dual_b)
    tp = tape(a, b)
    out = track(ForwardDiff.value(dual_c), D, tp)
    cache = RefValue(ForwardDiff.partials(dual_c))
    record!(tp, ScalarInstruction, self.f, (a, b), out, cache)
    return out
end

@inline function (self::ForwardOptimize{F}){F,V,D}(x::Real, t::TrackedReal{V,D})
    dual = self.f(x, Dual(value(t), one(V)))
    tp = tape(t)
    out = track(ForwardDiff.value(dual), D, tp)
    partial = ForwardDiff.partials(dual, 1)
    cache = RefValue(Partials((partial, partial)))
    record!(tp, ScalarInstruction, self.f, (x, t), out, cache)
    return out
end

@inline function (self::ForwardOptimize{F}){F,V,D}(t::TrackedReal{V,D}, x::Real)
    dual = self.f(Dual(value(t), one(V)), x)
    tp = tape(t)
    out = track(ForwardDiff.value(dual), D, tp)
    partial = ForwardDiff.partials(dual, 1)
    cache = RefValue(Partials((partial, partial)))
    record!(tp, ScalarInstruction, self.f, (t, x), out, cache)
    return out
end

@inline (self::ForwardOptimize{F}){F}(x::Dual, t::TrackedReal) = invoke(self.f, (Dual, Real), x, t)
@inline (self::ForwardOptimize{F}){F}(t::TrackedReal, x::Dual) = invoke(self.f, (Real, Dual), t, x)

#################################
# Skip Instruction Optimization #
#################################

immutable SkipOptimize{F}
    f::F
end

@inline SkipOptimize(f::SkipOptimize) = f

"""
    ReverseDiff.@skip(f)(args::Real...)
    ReverseDiff.@skip f(args::Real...) = ...
    ReverseDiff.@skip f = (args::Real...) -> ...

Declare that the given function should be skipped during the instruction-recording phase of
differentiation. Note that the macro can be used at either the definition site or at the
call site of `f`. **Note that, if `f` is defined within another function `g`, `f` should not
close over any differentiable input of `g`.** By using this macro, you are providing a
guarantee that this property holds true.

ReverseDiff overloads many Base scalar functions to behave as `@skip` functions by default.
A full list is given by `ReverseDiff.SKIPPED_UNARY_SCALAR_FUNCS` and
`ReverseDiff.SKIPPED_BINARY_SCALAR_FUNCS`.
"""
macro skip(ex)
    return esc(annotate_func_expr(:SkipOptimize, ex))
end

@inline (self::SkipOptimize{F}){F}(args...) = self.f(map(value, args)...)
@inline (self::SkipOptimize{F}){F}(a) = self.f(value(a))
@inline (self::SkipOptimize{F}){F}(a, b) = self.f(value(a), value(b))
@inline (self::SkipOptimize{F}){F}(a, b, c) = self.f(value(a), value(b), value(c))
