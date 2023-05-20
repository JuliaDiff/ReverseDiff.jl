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
        return dot(xv, xv), Δ -> (Δ * 2 * xv,)
    end
The `@grad` macro provides a way for the users to define custom adjoints for single-output functions wrt to their input numbers or arrays.
"""
macro grad(expr)
    d = MacroTools.splitdef(expr)
    f = d[:name]
    closure = gensym(f)
    d[:name] = closure
    closure_ex = MacroTools.combinedef(d)

    @gensym tp output_value output back args kwargs
    args_ex = getargs_expr(d[:args])
    kwargs_ex = getkwargs_expr(d[:kwargs])
    return quote
        function $ReverseDiff.track(::typeof($f), $(d[:args]...); $(d[:kwargs]...)) where {$(d[:whereparams]...),}
            $closure_ex
            $args = $args_ex
            $kwargs = $kwargs_ex
            $tp = $ReverseDiff.tape($args...)
            $output_value, $back = $closure($args...; $kwargs...)
            $output = $ReverseDiff.track($output_value, $tp)
            $ReverseDiff.record!(
                $tp,
                $ReverseDiff.SpecialInstruction,
                $f,
                $args,
                $output,
                ($back, $closure, $kwargs),
            )
            return $output
        end

        if !hasmethod(
            $ReverseDiff.special_reverse_exec!,
            Tuple{$ReverseDiff.SpecialInstruction{typeof($f)}},
        )
            @noinline function $ReverseDiff.special_reverse_exec!(instruction::$ReverseDiff.SpecialInstruction{typeof($f)})
                output = instruction.output
                input = instruction.input
                back = instruction.cache[1]
                input_derivs = back($ReverseDiff.deriv(output))
                @assert input_derivs isa Tuple
                $ReverseDiff._add_to_deriv!.(input, input_derivs)
                $ReverseDiff.unseed!(output)
                return nothing
            end
        end

        if !hasmethod(
            $ReverseDiff.special_forward_exec!,
            Tuple{$ReverseDiff.SpecialInstruction{typeof($f)}},
        )
            @noinline function $ReverseDiff.special_forward_exec!(instruction::$ReverseDiff.SpecialInstruction{typeof($f)})
                output, input = instruction.output, instruction.input
                $ReverseDiff.pull_value!.(input)
                pullback = instruction.cache[2]
                kwargs = instruction.cache[3]
                out_value = pullback(input...; kwargs...)[1]
                $ReverseDiff.value!(output, out_value)
                return nothing
            end
        end
    end |> esc
end

"""
    _make_fwd_args(func, arg_list)

Function `_make_fwd_args` accepts a function name and an argument
list, returns a tuple of argument lists whose elements are:
1. the`arg_list` untouched, 2. a new argument list with the function
as its first element and other elements in `arg_list` followed, 3. a
new argument for the definition of function `track`, 4. a new argument
list with all kwargs removed, 5, types of the arguments in the 4th
element, 5 the kwargs name if any otherwise an empty tuple. E.g.:

_make_fwd_args(:f, [:(a::String), :(b::TrackedReal), :(args...)])

returns

([:(a::String), :(b::TrackedReal), :(args...)],
 [:f, :(a::String), :(b::TrackedReal), :(args...)],
 [:(::typeof(f)), :(a::String), :(b::TrackedReal), :(args...)],
 [:(a::String), :(b::TrackedReal), :(args...)],
 [:String, :TrackedReal, :(Vararg{Any})],
 :kwargs)

It also deals with varargs and variable keyword arguments, and ensures
that at least one of the argument is tracked.

"""
function _make_fwd_args(func, args_l)
    kwargs = :(())
    args_r = copy(args_l)
    args_track = copy(args_l)
    if Meta.isexpr(args_r[1], :parameters) # has kw args
        insert!(args_r, 2, func)
        insert!(args_track, 2, :(::typeof($func)))
        kwargs = gensym(:kwargs)
        args_track[1].args = [:($(kwargs)...)]
    else
        insert!(args_r, 1, func)
        insert!(args_track, 1, :(::typeof($func)))
    end

    args_fixed = filter(copy(args_l)) do arg
        !Meta.isexpr(arg, :parameters)
    end

    arg_types = map(args_fixed) do arg
        if Meta.isexpr(arg, :(...))
            Meta.isexpr(arg.args[1], :(::)) ? :(Vararg{$(arg.args[1].args[end])}) : :(Vararg{Any})
        elseif Meta.isexpr(arg, :(::))
            arg.args[end]
        else
            :Any
        end
    end

    return args_l, args_r, args_track, args_fixed, arg_types, kwargs
end

"""
    @grad_from_chainrules f(args...; kwargs...)

The `@grad_from_chainrules` macro provides a way to import
adjoints(rrule) defined in ChainRules to ReverseDiff. One must provide
a method signature to import the corresponding `rrule`. In the
provided method signature, one should replace the types of arguments
to which one wants to take derivatives with respect with
`ReverseDiff.TrackedReal` and `ReverseDiff.TrackedArray`
respectively. For example, we can import `rrule` of `f(x::Real,
y::Array)` like below:

```julia
ReverseDiff.@grad_from_chainrules f(x::TrackedReal, y::TrackedArray)
ReverseDiff.@grad_from_chainrules f(x::TrackedReal, y::Array)
ReverseDiff.@grad_from_chainrules f(x::Real, y::TrackedArray)
```
"""
macro grad_from_chainrules(fcall)
    Meta.isexpr(fcall, :call) && length(fcall.args) >= 2 ||
        error("`@grad_from_chainrules` has to be applied to a function signature")
    f = esc(fcall.args[1])
    xs = map(fcall.args[2:end]) do x
        if x isa Expr && x.head == :(::)
            if length(x.args) == 1 # ::T without var name
                return :($(gensym())::$(esc(x.args[1])))
            else # x::T
                return :($(x.args[1])::$(esc(x.args[2])))
            end
        else
            return x
        end
    end
    args_l, args_r, args_track, args_fixed, arg_types, kwargs = _make_fwd_args(f, xs)
    return quote
        $f($(args_l...)) = ReverseDiff.track($(args_r...))
        function ReverseDiff.track($(args_track...))
            args = ($(args_fixed...),)
            tp = ReverseDiff.tape(args...)
            output_value, back = ChainRulesCore.rrule($f, map(ReverseDiff.value, args)...; $kwargs...)
            output = ReverseDiff.track(output_value, tp)
            closure(cls_args...; cls_kwargs...) = ChainRulesCore.rrule($f, map(ReverseDiff.value, cls_args)...; cls_kwargs...)
            ReverseDiff.record!(
                tp,
                ReverseDiff.SpecialInstruction,
                $f,
                args,
                output,
                (back, closure, $kwargs),
            )
            return output
        end

        @noinline function ReverseDiff.special_reverse_exec!(instruction::ReverseDiff.SpecialInstruction{typeof($f), <:Tuple{$(arg_types...)}})
            output = instruction.output
            input = instruction.input
            back = instruction.cache[1]
            back_output = back(ReverseDiff.deriv(output))
            input_derivs = back_output[2:end]
            @assert input_derivs isa Tuple
            ReverseDiff._add_to_deriv!.(input, input_derivs)
            ReverseDiff.unseed!(output)
            return nothing
        end

        @noinline function ReverseDiff.special_forward_exec!(instruction::ReverseDiff.SpecialInstruction{typeof($f), <:Tuple{$(arg_types...)}})
            output, input = instruction.output, instruction.input
            ReverseDiff.pull_value!.(input)
            pullback = instruction.cache[2]
            kwargs = instruction.cache[3]
            out_value = pullback(input...; kwargs...)[1]
            ReverseDiff.value!(output, out_value)
            return nothing
        end
    end
end

_add_to_deriv!(d1, d2) = nothing
function _add_to_deriv!(d1::Union{TrackedReal, AbstractArray{<:TrackedReal}}, d2::AbstractThunk)
    increment_deriv!(d1, unthunk(d2))
end
function _add_to_deriv!(d1::Union{TrackedReal, AbstractArray{<:TrackedReal}}, d2)
    increment_deriv!(d1, d2)
end
function getargs_expr(args_with_types)
    expr = Expr(:tuple)
    for at in args_with_types
        x, tosplat = remove_tp(at)
        if tosplat
            push!(expr.args, :($x...))
        else
            push!(expr.args, x)
        end
    end
    return expr
end
function getkwargs_expr(kwargs_with_types)
    syms = []
    final = nothing
    for at in kwargs_with_types
        final isa Nothing || throw("Invalid kwargs.")
        x, tosplat = remove_tp(at)
        if tosplat
            final = x
        else
            push!(syms, x)
        end
    end
    expr = length(syms) == 0 ? :(NamedTuple()) : Expr(:tuple, [:($f = $f) for f in syms]...)
    final = final == nothing ? :(NamedTuple()) : final
    return :(Base.merge($expr, $final))
end
function remove_tp(t)
    if @capture(t, X_::T_...)
        return X, true
    elseif @capture(t, X_::T_)
        return X, false
    elseif @capture(t, X_::T_ = V_)
        return X, false
    elseif @capture(t, ::typeof(T_)...)
        return T, true
    elseif @capture(t, ::typeof(T_))
        return T, false
    elseif @capture(t, X_...)
        return X, true
    elseif @capture(t, X_ = V_)
        return X, false
    else
        return t, false
    end
end
