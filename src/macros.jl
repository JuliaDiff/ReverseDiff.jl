#=
works for the following formats:
- `@forward(f)(args...)`
- `@forward f(args...) = ...`
- `@forward f = (args...) -> ...`
=#
function annotate_func_expr(typesym, _module_, expr)
    if isa(expr, Expr) && (expr.head == :(=) || expr.head == :function)
        lhs = expr.args[1]
        if isa(lhs, Expr) && (lhs.head == :call || lhs.head == :where) # named function definition site
            given_name = lhs.head == :where ? lhs.args[1].args[1] : lhs.args[1]
            @assert isa(given_name, Symbol) "potentially malformed function signature for $typesym"
            hidden_name = Symbol("#hidden_$(given_name)")
            if lhs.head == :where
                lhs.args[1].args[1] = hidden_name
            else
                lhs.args[1] = hidden_name
            end
            return quote
                $expr
                if !(isdefined($(_module_), $(Expr(:quote, given_name))))
                    const $(given_name) = ReverseDiff.$(typesym)($(hidden_name))
                end
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

struct ForwardOptimize{F}
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
default. A full list is given by `DiffRules.diffrules()`.
"""
macro forward(ex)
    return esc(annotate_func_expr(:ForwardOptimize, __module__, ex))
end

# fallback #
#----------#

@inline (self::ForwardOptimize{F})(args...) where {F} = self.f(args...)

# unary #
#-------#

@inline function (self::ForwardOptimize{F})(t::TrackedReal{V,D}) where {F,V,D}
    T = promote_type(V, D)
    result = DiffResult(zero(T), zero(T))
    result = ForwardDiff.derivative!(result, self.f, value(t))
    tp = tape(t)
    out = track(DiffResults.value(result), D, tp)
    cache = RefValue(DiffResults.derivative(result))
    record!(tp, ScalarInstruction, self.f, t, out, cache)
    return out
end

# binary #
#--------#

@inline function (self::ForwardOptimize{F})(a::TrackedReal{V1,D}, b::TrackedReal{V2,D}) where {F,V1,V2,D}
    T = promote_type(V1, V2, D)
    result = DiffResults.GradientResult(SVector(zero(T), zero(T)))
    result = ForwardDiff.gradient!(result, x -> self.f(x[1], x[2]), SVector(value(a), value(b)))
    tp = tape(a, b)
    out = track(DiffResults.value(result), D, tp)
    cache = RefValue(DiffResults.gradient(result))
    record!(tp, ScalarInstruction, self.f, (a, b), out, cache)
    return out
end

@inline function (self::ForwardOptimize{F})(x::Real, t::TrackedReal{V,D}) where {F,V,D}
    T = promote_type(typeof(x), V, D)
    result = DiffResult(zero(T), zero(T))
    result = ForwardDiff.derivative!(result, vt -> self.f(x, vt), value(t))
    tp = tape(t)
    out = track(DiffResults.value(result), D, tp)
    dt = DiffResults.derivative(result)
    cache = RefValue(SVector(dt, dt))
    record!(tp, ScalarInstruction, self.f, (x, t), out, cache)
    return out
end

@inline function (self::ForwardOptimize{F})(t::TrackedReal{V,D}, x::Real) where {F,V,D}
    T = promote_type(typeof(x), V, D)
    result = DiffResult(zero(T), zero(T))
    result = ForwardDiff.derivative!(result, vt -> self.f(vt, x), value(t))
    tp = tape(t)
    out = track(DiffResults.value(result), D, tp)
    dt = DiffResults.derivative(result)
    cache = RefValue(SVector(dt, dt))
    record!(tp, ScalarInstruction, self.f, (t, x), out, cache)
    return out
end

@inline (self::ForwardOptimize{F})(x::Dual, t::TrackedReal) where {F} = invoke(self.f, Tuple{Dual,Real}, x, t)
@inline (self::ForwardOptimize{F})(t::TrackedReal, x::Dual) where {F} = invoke(self.f, Tuple{Real,Dual}, t, x)

#################################
# Skip Instruction Optimization #
#################################

struct SkipOptimize{F}
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
    return esc(annotate_func_expr(:SkipOptimize, __module__, ex))
end

@inline (self::SkipOptimize{F})(args...) where {F} = self.f(map(value, args)...)
@inline (self::SkipOptimize{F})(a) where {F} = self.f(value(a))
@inline (self::SkipOptimize{F})(a, b) where {F} = self.f(value(a), value(b))
@inline (self::SkipOptimize{F})(a, b, c) where {F} = self.f(value(a), value(b), value(c))

"""
    f(x) = dot(x, x)
    f(x::ReverseDiff.TrackedVector) = ReverseDiff.track(f, x)
    ReverseDiff.@grad function f(x)
        xv = ReverseDiff.value(x)
        return dot(xv, xv), ∇ -> (∇ * 2 * xv,)
    end

The `@grad` macro provides a way for the users to define custom adjoints for single-output functions wrt to their input numbers or arrays.
"""
macro grad(expr)
    if @capture(expr, 
        (f_(xs__) where {T__} = body_) | 
        (f_(xs__) = body_) | 
        (function f_(xs__) body_ end) | 
        (function f_(xs__) where {T__} body_ end)
    )
        closure = gensym(:f)
        tp = gensym(:tp)
        output_value = gensym(:ov)
        output = gensym(:o)
        back = gensym(:back)
        xsv = remove_tp.(xs)
        T = T == nothing ? [] : T
        return quote
            function ReverseDiff.track(::typeof($f), $(xs...)) where {$(T...),}
                $closure = ($(xs...),) -> $body
                $tp = ReverseDiff.tape($(xsv...),)
                $output_value, $back = $closure($(xsv...),)
                $output = ReverseDiff.track($output_value, $tp)
                ReverseDiff.record!(
                    $tp,
                    ReverseDiff.SpecialInstruction,
                    $f,
                    ($(xsv...),),
                    $output,
                    ($back, $closure),
                )
                return $output
            end

            @static if !hasmethod(
                ReverseDiff.special_reverse_exec!,
                Tuple{ReverseDiff.SpecialInstruction{typeof($f)}},
            )
                @noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof($f)})
                    output = instruction.output
                    input = instruction.input
                    back = instruction.cache[1]
                    input_derivs = back(ReverseDiff.deriv(output))
                    @assert input_derivs isa Tuple
                    ReverseDiff.add_to_deriv!.(input, input_derivs)
                    ReverseDiff.unseed!(output)
                    return nothing
                end
            end

            @static if !hasmethod(
                ReverseDiff.special_forward_exec!,
                Tuple{ReverseDiff.SpecialInstruction{typeof($f)}},
            )
                @noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof($f)})
                    output, input = instruction.output, instruction.input
                    pullback = instruction.cache[2]
                    out_value = pullback(input...)[1]
                    ReverseDiff.value!(output, out_value)
                    return nothing
                end
            end
        end |> esc
    else
        throw("Invalid `ReverseDiff` custom gradient definition.")
    end
end
add_to_deriv!(d1, d2) = nothing
function add_to_deriv!(d1::Union{TrackedReal, TrackedArray}, d2)
    d = ReverseDiff.deriv(d1)
    d .+= d2
end
function remove_tp(t)
    if @capture(t, X_::T_)
        return X
    elseif @capture(t, ::typeof(T_))
        return T
    else
        return t
    end
end
