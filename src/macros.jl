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

@inline function (self::ForwardOptimize{F})(t::TrackedReal{V, D}) where {F, V, D}
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

@inline function (self::ForwardOptimize{F})(a::TrackedReal{V1, D}, b::TrackedReal{V2, D}) where {F, V1, V2, D}
    T = promote_type(V1, V2, D)
    result = DiffResults.GradientResult(SVector(zero(T), zero(T)))
    result = ForwardDiff.gradient!(result, x -> self.f(x[1], x[2]), SVector(value(a), value(b)))
    tp = tape(a, b)
    out = track(DiffResults.value(result), D, tp)
    cache = RefValue(DiffResults.gradient(result))
    record!(tp, ScalarInstruction, self.f, (a, b), out, cache)
    return out
end

@inline function (self::ForwardOptimize{F})(x::Real, t::TrackedReal{V, D}) where {F, V, D}
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

@inline function (self::ForwardOptimize{F})(t::TrackedReal{V, D}, x::Real) where {F, V, D}
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

@inline (self::ForwardOptimize{F})(x::Dual, t::TrackedReal) where {F} = invoke(self.f, Tuple{Dual, Real}, x, t)
@inline (self::ForwardOptimize{F})(t::TrackedReal, x::Dual) where {F} = invoke(self.f, Tuple{Real, Dual}, t, x)

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

skipvalue(x) = value(x)
skipvalue(x::TrackedReal) = skipvalue(value(x))
skipvalue(x::AbstractArray{<:TrackedReal}) = skipvalue(value(x))

@inline (self::SkipOptimize{F})(args...) where {F} = self.f(map(skipvalue, args)...)

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
        function $ReverseDiff.track(::typeof($f), $(d[:args]...); $(d[:kwargs]...)) where {$(d[:whereparams]...)}
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
                $ReverseDiff._check_tangents($f, input_derivs, length(input))
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

# `MacroTools.splitarg` with all user-supplied parts escaped, and a name for unnamed arguments
function splitarg_esc(arg)
    name, type, tosplat, default = MacroTools.splitarg(arg)
    if name === nothing
        name = gensym()
    end
    name = esc(name)
    # `combinearg` drops only an unescaped `::Any`, and a keyword splat cannot be typed
    if type !== :Any
        type = esc(type)
    end
    if default !== nothing
        default = esc(default)
    end
    return name, type, tosplat, default
end

# forward an argument by name: an annotation would be a type assert, a default a keyword
function forwardarg((name, _, tosplat, _), iskw)
    if tosplat
        return :($name...)
    elseif iskw
        return Expr(:kw, name, name)
    else
        return name
    end
end

# the `where` parameters that positional types refer to, directly or via the bounds of such parameters,
# since unused type variables cause warnings; a bound can only refer to earlier parameters
function positional_whereparams(whereparams, args)
    types = [MacroTools.splitarg(arg)[2] for arg in args]
    used = []
    for p in Iterators.reverse(whereparams)
        name = wherename(p)
        if any(t -> MacroTools.inexpr(t, name), types) || any(q -> MacroTools.inexpr(q, name), used)
            pushfirst!(used, p)
        end
    end
    return used
end

# `T`, `T <: U`, `T >: L` or `L <: T <: U`
function wherename(p)
    if Meta.isexpr(p, :<:) || Meta.isexpr(p, :>:)
        return p.args[1]
    elseif Meta.isexpr(p, :comparison)
        return p.args[3]
    else
        return p
    end
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

Varargs, keyword arguments and `where` clauses are supported as well:

```julia
ReverseDiff.@grad_from_chainrules f(x::TrackedArray{V}, ys::Int...; k = 1) where {V}
```

Note that every imported signature generates methods that dispatch on the types of
all arguments. If two signatures of the same function overlap, such as
`f(x::TrackedArray, y)` and `f(x, y::TrackedArray)`, then the fully tracked
combination `f(x::TrackedArray, y::TrackedArray)` has to be imported as well,
otherwise a call with two tracked arguments is ambiguous.
"""
macro grad_from_chainrules(fcall)
    if !Meta.isexpr(fcall, :call) && !Meta.isexpr(fcall, :where)
        error("`@grad_from_chainrules` has to be applied to a function signature")
    end
    d = MacroTools.splitdef(:($fcall = nothing))
    if isempty(d[:args])
        error("`@grad_from_chainrules` requires a signature with at least one positional argument")
    end
    f = esc(d[:name])
    whereparams = map(esc, d[:whereparams])
    args = map(splitarg_esc, d[:args])
    kwargs = map(splitarg_esc, d[:kwargs])

    sig_args = [MacroTools.combinearg(arg...) for arg in args]
    sig_kwargs = [MacroTools.combinearg(kw...) for kw in kwargs]
    # positional defaults are handled by the methods of `f`
    track_args = [MacroTools.combinearg(name, type, tosplat, nothing) for (name, type, tosplat, _) in args]
    call_args = [forwardarg(arg, false) for arg in args]
    call_kwargs = [forwardarg(kw, true) for kw in kwargs]
    arg_types = [tosplat ? :(Vararg{$type}) : type for (_, type, tosplat, _) in args]
    # `track` and the exec methods only see positional types
    track_whereparams = map(esc, positional_whereparams(d[:whereparams], d[:args]))

    # argument names are escaped and may shadow `$f` in the bodies, so bind the function via its type
    return quote
        (func::$ReverseDiff._selftype($f))($(sig_args...); $(sig_kwargs...)) where {$(whereparams...)} =
            $ReverseDiff.track(func, $(call_args...); $(call_kwargs...))
        function $ReverseDiff.track(func::$ReverseDiff._selftype($f), $(track_args...); kwargs...) where {$(track_whereparams...)}
            args = ($(call_args...),)
            tp = $ReverseDiff.tape(args...)
            output_value, back = $ChainRulesCore.rrule(func, map($ReverseDiff.value, args)...; kwargs...)
            output = $ReverseDiff.track(output_value, tp)
            closure(cls_args...; cls_kwargs...) = $ChainRulesCore.rrule(func, map($ReverseDiff.value, cls_args)...; cls_kwargs...)
            $ReverseDiff.record!(
                tp,
                $ReverseDiff.SpecialInstruction,
                func,
                args,
                output,
                (back, closure, kwargs),
            )
            return output
        end

        @noinline function $ReverseDiff.special_reverse_exec!(instruction::$ReverseDiff.SpecialInstruction{typeof($f), <:Tuple{$(arg_types...)}}) where {$(track_whereparams...)}
            output = instruction.output
            input = instruction.input
            back = instruction.cache[1]
            back_output = back($ReverseDiff.deriv(output))
            # the first tangent is the one of the function itself
            $ReverseDiff._check_tangents($f, back_output, length(input) + 1)
            input_derivs = Base.tail(back_output)
            $ReverseDiff._add_to_deriv!.(input, input_derivs)
            $ReverseDiff.unseed!(output)
            return nothing
        end

        @noinline function $ReverseDiff.special_forward_exec!(instruction::$ReverseDiff.SpecialInstruction{typeof($f), <:Tuple{$(arg_types...)}}) where {$(track_whereparams...)}
            output, input = instruction.output, instruction.input
            $ReverseDiff.pull_value!.(input)
            pullback = instruction.cache[2]
            kwargs = instruction.cache[3]
            out_value = pullback(input...; kwargs...)[1]
            $ReverseDiff.value!(output, out_value)
            return nothing
        end
    end
end

# like `Core.Typeof`, which Julia 1.14 renames to `Core.TypeEqOf` (JuliaLang/julia#62001)
_selftype(@nospecialize(x)) = x isa Type ? Type{x} : typeof(x)

# a 1-tuple would be silently broadcast over every argument
function _check_tangents(f, tangents, n)
    if !(tangents isa Tuple)
        throw(ArgumentError(lazy"the pullback of `$f` must return a tuple of tangents"))
    elseif length(tangents) != n
        throw(
            ArgumentError(
                lazy"the pullback of `$f` must return a tuple of length $n, not $(length(tangents))"
            ),
        )
    else
        return nothing
    end
end

_add_to_deriv!(d1, d2) = nothing
_add_to_deriv!(d1::Union{TrackedReal, AbstractArray}, ::Union{AbstractZero, Nothing}) = nothing
function _add_to_deriv!(d1::Union{TrackedReal, AbstractArray}, d2::AbstractThunk)
    istracked(d1) && increment_deriv!(d1, unthunk(d2))
    return nothing
end
function _add_to_deriv!(d1::Union{TrackedReal, AbstractArray}, d2)
    istracked(d1) && increment_deriv!(d1, d2)
    return nothing
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
