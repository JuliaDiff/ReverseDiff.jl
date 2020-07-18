##################
## Broadcasting ##
##################

using Base.Broadcast: BroadcastStyle, ArrayStyle, Broadcasted, broadcasted
using ForwardDiff: ForwardDiff, Dual
import Base.Broadcast: materialize
const RDBroadcasted{F, T} = Broadcasted{<:Any, <:Any, F, T}

"""
    NotTracked(f::Function)

A struct that can be used to wrap around closures, structs and arrays of structs declaring that they do not contain tracked variables. This enables a more efficient broadcasting of such functions and structs when doing automatic differentiation with `ReverseDiff` producing a `TrackedArray` instead of an `Array{<:TrackedReal}`.
"""
struct NotTracked{F} <: Function
    f::F
end
(f::NotTracked{<:Union{Function, Type}})(args...; kwargs...) = f.f(args...; kwargs...)

istypeorclosure(::F) where {F} = _istypeorclosure(F)
istypeorclosure(::AbstractArray{F}) where {F} = _istypeorclosure(F)
istypeorclosure(::Base.RefValue{F}) where {F} = _istypeorclosure(F)
istypeorclosure(::AbstractArray{<:Real}) = false
istypeorclosure(::TrackedArray) = false
istypeorclosure(::AbstractArray{<:TrackedReal}) = true
istypeorclosure(::Real) = false
@generated _istypeorclosure(::Type{F}) where {F} = :($(fieldcount(F) > 0))

mayhavetracked(b) = istypeorclosure(b)
mayhavetracked(b::Type) = false
mayhavetracked(b::NotTracked) = false
mayhavetracked(b::Base.RefValue{<:NotTracked}) = false
mayhavetracked(b::Broadcasted) = mayhavetracked(b.f) || any(mayhavetracked, b.args)

struct TrackedStyle <: BroadcastStyle end

Broadcast.BroadcastStyle(::Type{<:Union{TrackedArray, TrackedReal}}) = TrackedStyle()
Broadcast.BroadcastStyle(::TrackedStyle, b::BroadcastStyle) = TrackedStyle()

# We have to re-build the original broadcast struct to get the appropriate array
# style. We need this primarily to support CuArrays' broadcasting fixes.
broadcast_rebuild(xs) = recur_value(xs)
recur_value(xs) = xs
recur_value(xs::Union{TrackedReal, TrackedArray}) = recur_value(value(xs))

function broadcast_rebuild(bc::Broadcasted)
    broadcasted(bc.f, broadcast_rebuild.(bc.args)...)
end

getstyle(::Broadcasted{Style}) where {Style} = Style
remove_not_tracked(f) = f
remove_not_tracked(f::NotTracked) = f.f
remove_not_tracked(f::Base.RefValue{<:NotTracked}) = Ref(remove_not_tracked(f[]))
remove_not_tracked(f::Base.RefValue{<:NotTracked{<:AbstractArray}}) = remove_not_tracked(f[])
function remove_not_tracked(b::Broadcasted{style}) where {style}
    return Broadcasted{style}(remove_not_tracked(b.f), remove_not_tracked.(b.args), b.axes)
end

onlyrealarrays(args::Tuple) = onlyrealarray(first(args)) && onlyrealarrays(Base.tail(args))
onlyrealarrays(::Tuple{}) = true
onlyrealarray(::AbstractArray{<:Real}) = true
onlyrealarray(::AbstractArray) = false
onlyrealarray(::Any) = true

anyreals(args::Tuple) = first(args) isa Real || anyreals(Base.tail(args))
anyreals(args::Tuple{}) = false

function get_implementation(bc, f, T, args)
    outputisreal = (T <: AbstractArray{<:Real}) && (T !== Union{})
    # Each arg is either a real number, an array of untraked reals, a tracked array of reals or an array of untracked non-reals,
    # Output is real, and
    # No tracked closure or arguments, except TrackedReal and TrackedArray.
    if !mayhavetracked(bc) && outputisreal && (anyreals(args) || !onlyrealarrays(args))
        return Val(:tracker)
    # No arg is a real number and array args must be arrays of untracked reals or tracked arrays of reals,
    # Output is real, and
    # No tracked closure or arguments, except TrackedReal and TrackedArray.
    elseif !mayhavetracked(bc) && outputisreal
        return Val(:reversediff)
    # Function or any arg is possibly a tracked non-real or an array of tracked reals/non-reals,
    # Or output is not an array of reals
    else
        return Val(:fallback)
    end
end
function Base.copy(_bc::Broadcasted{TrackedStyle})
    bc = remove_not_tracked(_bc)
    flattened_bc = Broadcast.flatten(bc)
    untracked_bc = broadcast_rebuild(bc)
    flattened_untracked_bc = Broadcast.flatten(untracked_bc)
    T = Core.Compiler.return_type(copy, Tuple{typeof(untracked_bc)})
    f, args = flattened_untracked_bc.f, flattened_bc.args
    implementation = get_implementation(_bc, f, T, args)
    if implementation isa Val{:reversediff}
        return ∇broadcast(f, args...)
    elseif implementation isa Val{:tracker}
        return tracker_∇broadcast(f, args...)
    else
        style, axes = getstyle(flattened_untracked_bc), flattened_bc.axes
        return copy(Broadcasted{style, typeof(axes), typeof(f), typeof(args)}(f, args, axes))
    end
end

# https://github.com/FluxML/Flux.jl/issues/353
if VERSION < v"1.1.0-DEV.548"
    @eval Base.Broadcast begin
        function flatten(bc::Broadcasted{Style}) where {Style}
            isflat(bc) && return bc
            args = cat_nested(bc)
            let makeargs = make_makeargs(bc), f = bc.f
                newf = @inline function(args::Vararg{Any,N}) where N
                f(makeargs(args...)...)
                end
                return Broadcasted{Style}(newf, args, bc.axes)
            end
        end
        @inline function make_makeargs(makeargs, t::Tuple{<:Broadcasted,Vararg{Any}})
            bc = t[1]
            let makeargs = make_makeargs(makeargs, tail(t)), f = bc.f
                let makeargs = make_makeargs(makeargs, bc.args)
                    headargs, tailargs = make_headargs(bc.args), make_tailargs(bc.args)
                    return @inline function(args::Vararg{Any,N}) where N
                        args1 = makeargs(args...)
                        a, b = headargs(args1...), tailargs(args1...)
                        (f(a...), b...)
                    end
                end
            end
        end
    end
end

getouttype(::TrackedReal{<:Any, D}) where {D} = D
getouttype(::TrackedArray{<:Any, D}) where {D} = D
getouttype(::Any) = Union{}

deref(x) = x
deref(x::Base.RefValue) = x[]

@generated function splatcall(f, x::SVector{N}, utargs::T, ::Val{tinds}) where {N, T <: Tuple, tinds}
    args = []
    ti = 1
    uti = 1
    for i in 1:(N + length(T.types))
        if i in tinds
            push!(args, :(deref(x[$ti])))
            ti += 1
        else
            push!(args, :(deref(utargs[$uti])))
            uti += 1
        end
    end
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:call, :f, args...))
    end
end

@generated function splitargs(args::T) where {T <: Tuple}
    N = length(T.types)
    RealOrArray = Union{Real, AbstractArray}
    inds = [i for i in 1:N if T.types[i] <: RealOrArray]
    indsval = :(Val{$(Expr(:tuple, [:($i) for i in inds]...))}())
    maybetracked = Expr(:tuple, [:(args[$i]) for i in inds]...)
    untracked = Expr(:tuple, [:(args[$i]) for i in 1:N if !(i in inds)]...)
    return :($indsval, $maybetracked, $untracked)
end

## A generalization of the broadcasting approach in ReverseDiff for general functions

@inline function ∇broadcast(f::F, args::Vararg{<:Any}) where {F}
    inds, targs, untracked = splitargs(args)
    N = length(targs)
    D = promote_type(getouttype.(targs)...)
    result = DiffResults.GradientResult(zero(SVector{N, D}))
    function df(x...)
        return ForwardDiff.gradient!(
            result,
            s -> splatcall(f, s, untracked, inds),
            SVector(x),
        )
    end
    results = broadcast(df, value.(targs)...)
    tp = tape(targs...)
    out_value = DiffResults.value.(results)
    eltype(out_value) == Bool && return out_value
    out = track(out_value, D, tp)
	cache = (results, df, index_bound.(targs, (out,)))
	record!(tp, SpecialInstruction, ∇broadcast, targs, out, cache)
    return out
end
@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(∇broadcast)})
    input = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    results, _, bounds = instruction.cache
    N = length(input)
    if N == 1 || all(isequal(size(input[1])), size.(Base.tail(input)))
        _add_to_deriv!(input, output_deriv, results)
    else
        _add_to_deriv!(input, output_deriv, results, bounds)
    end
    unseed!(output)
    return nothing
end

@generated function _add_to_deriv!(xs::T, o, r) where {T <: Tuple}
    N = length(T.types)
    return Expr(:block, [:(_add_to_deriv!(xs[$i], o, r, Val($i))) for i in 1:N]...)
end
_add_to_deriv!(_, _, _, _) = nothing
function _add_to_deriv!(x::Union{TrackedReal, TrackedArray}, out_deriv, results, ::Val{i}) where {i}
    return istracked(x) && diffresult_increment_deriv!(x, out_deriv, results, i)
end

@generated function _add_to_deriv!(xs::T, o, r, bounds) where {T <: Tuple}
    N = length(T.types)
    return Expr(:block, [:(_add_to_deriv!(xs[$i], o, r, Val($i), bounds[$i])) for i in 1:N]...)
end
_add_to_deriv!(_, _, _, _, _) = nothing
function _add_to_deriv!(x::Union{TrackedReal,TrackedArray}, out_deriv, results, ::Val{i}, bound) where {i}
    return istracked(x) && diffresult_increment_deriv!(x, out_deriv, results, i, bound)
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(∇broadcast)})
    input, output = instruction.input, instruction.output
    results, df, _ = instruction.cache
    pull_value!.(input)
    broadcast!(df, results, value.(input)...)
    output_value = value(output)
    output_value .= DiffResults.value.(results)
    return nothing
end

## Tracker style broadcasting
## Good for broadcasting real numbers or arrays of non-tracked structs

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

unbroadcast(x::AbstractArray, Δ) =
  size(x) == size(Δ) ? Δ :
  length(x) == length(Δ) ? trim(x, Δ) :
    trim(x, sum(Δ, dims = ntuple(i -> size(x, i) == 1 ? i : ndims(Δ)+1, Val(ndims(Δ)))))

unbroadcast(x::Number, Δ) = sum(Δ)
unbroadcast(x::Base.RefValue, _) = nothing

dual(x, p) = x
dual(x::Real, p) = Dual(x, p)

function _deriv(f, G, ::Val{i}, args::Vararg{Any, N}) where {N, i}
    dargs = ntuple(j -> dual(args[j], i==j), Val(N))
    return f(dargs...).partials[1] * G
end
@generated function _derivs(f, G, args::Vararg{Any, N}) where {N}
    return Expr(:tuple, [:(_deriv.(f, G, Val($i), args...)) for i in 1:N]...)
end
@inline function tracker_∇broadcast(f, args::Vararg{Any, N}) where {N}
    args_values = map(value, args)
    out_value = broadcast(f, args_values...)
    tp = tape(args...)
    eltype(out_value) == Bool && return out_value
	out = track(out_value, tp)
    cache = (f,)
	record!(tp, SpecialInstruction, tracker_∇broadcast, args, out, cache)
    return out
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(tracker_∇broadcast)})
    input, output = instruction.input, instruction.output
    f = instruction.cache[1]
    output_value = value(output)
    pull_value!.(input)
    broadcast!(f, output_value, value.(input)...)
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(tracker_∇broadcast)})
    input = instruction.input
    output = instruction.output
    f = instruction.cache[1]
    output_deriv = deriv(output)
    N = length(input)
    Δargs = _derivs(f, output_deriv, value.(input)...)
    dxs = map(unbroadcast, input, Δargs)
    map(_add_to_deriv!, input, dxs)
    unseed!(output)
    return nothing
end

## Limited ReverseDiff broadcasting
## Efficient broadcasting for specific functions, e.g. +, -

@inline _materialize(f, args) = broadcast(f, args...)

for (M, f, arity) in DiffRules.diffrules()
    isdefined(ReverseDiff, M) || continue
    if arity == 1
        @eval @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{TrackedArray}}) = _materialize(bc.f, bc.args)
    elseif arity == 2
        @eval begin
            @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{TrackedArray,TrackedArray}}) = _materialize(bc.f, bc.args)
            @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{TrackedArray,TrackedReal}}) = _materialize(bc.f, bc.args)
            @noinline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{TrackedReal,TrackedArray}}) = _materialize(bc.f, bc.args)
        end
        for A in ARRAY_TYPES
            @eval begin
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{$A,TrackedArray}}) = _materialize(bc.f, bc.args)
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{TrackedArray, $A}}) = _materialize(bc.f, bc.args)
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{$A, TrackedReal}}) = _materialize(bc.f, bc.args)
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{TrackedReal,$A}}) = _materialize(bc.f, bc.args)
            end
        end
        for R in REAL_TYPES
            @eval begin
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{$R,TrackedArray}}) = _materialize(bc.f, bc.args)
                @inline materialize(bc::RDBroadcasted{typeof($M.$f), <:Tuple{TrackedArray,$R}}) = _materialize(bc.f, bc.args)
            end
        end
    end
end
