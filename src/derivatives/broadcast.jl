##################
## Broadcasting ##
##################

using Base.Broadcast: AbstractArrayStyle, BroadcastStyle, Broadcasted, DefaultArrayStyle
using ForwardDiff: ForwardDiff, Dual

"""
    NotTracked(f::Function)

A struct that can be used to wrap around closures, structs and arrays of structs declaring that they do not contain tracked variables. This enables a more efficient broadcasting of such functions and structs when doing automatic differentiation with `ReverseDiff` producing a `TrackedArray` instead of an `Array{<:TrackedReal}`.
"""
struct NotTracked{F} <: Function
    f::F
end
(f::NotTracked{<:Union{Function, Type}})(args...; kwargs...) = f.f(args...; kwargs...)

# can `f` receive tracked values that `∇broadcast` does not seed?
mayhidetracked(::F) where {F} = _mayhidetracked(F)
mayhidetracked(::AbstractArray{F}) where {F} = _mayhidetracked(F)
mayhidetracked(::Base.RefValue{F}) where {F} = _mayhidetracked(F)
mayhidetracked(::AbstractArray{<:Real}) = false
mayhidetracked(::Real) = false
mayhidetracked(::Type) = false
mayhidetracked(t::Tuple) = any(x -> istracked(x) || mayhidetracked(x), t)
mayhidetracked(b::ForwardOptimize) = mayhidetracked(b.f)
mayhidetracked(b::SkipOptimize) = mayhidetracked(b.f)
mayhidetracked(b::Broadcasted) = mayhidetracked(b.f) || any(mayhidetracked, b.args)

# below the argument nothing is seeded, so a tracked value is hidden wherever it sits: recurse
# into what a container holds rather than ask about the container
_mayhidetracked(::Type{<:NotTracked}) = false
_mayhidetracked(::Type{<:AbstractArray{F}}) where {F} = _mayhidetracked(F)
# a type that is not concrete, such as `Type{T}` or an abstract type, may have fields
_mayhidetracked(::Type{F}) where {F} = !isconcretetype(F) || fieldcount(F) > 0

struct TrackedStyle{N} <: AbstractArrayStyle{N} end

(::Type{<:TrackedStyle})(::Val{N}) where {N} = TrackedStyle{N}()

Broadcast.BroadcastStyle(::Type{<:TrackedReal}) = TrackedStyle{0}()
Broadcast.BroadcastStyle(::Type{<:AbstractArray{<:TrackedReal{<:Any,D},N}}) where {D,N} =
    TrackedStyle{N}()

# `AbstractArrayStyle{Any}` carries `Any` as its dimension, which `max` cannot compare
_maxdim(M::Int, N::Int) = max(M, N)
_maxdim(_, _) = Any

# tracked values must stay tracked, so take precedence over every other array style
Broadcast.BroadcastStyle(::TrackedStyle{M}, ::AbstractArrayStyle{N}) where {M,N} =
    TrackedStyle{_maxdim(M, N)}()

# resolve the overlap with `Base`'s three `DefaultArrayStyle` rules, preserving their results
Broadcast.BroadcastStyle(::TrackedStyle{M}, ::DefaultArrayStyle{N}) where {M,N} =
    TrackedStyle{_maxdim(M, N)}()
Broadcast.BroadcastStyle(::TrackedStyle{N}, ::DefaultArrayStyle{N}) where {N} = TrackedStyle{N}()
Broadcast.BroadcastStyle(::TrackedStyle{Any}, ::DefaultArrayStyle) = TrackedStyle{Any}()

# the style the arguments have untracked, which picks the fallback's output container
untrackedstyle(x) = Broadcast.BroadcastStyle(typeof(x))
untrackedstyle(x::Union{TrackedReal, AbstractArray{<:TrackedReal}}) = untrackedstyle(value(x))
# unwrapping would copy
untrackedstyle(::Array{<:TrackedReal,N}) where {N} = DefaultArrayStyle{N}()

# a static value's elements, as a static array that `StaticArrays` broadcasts over itself
staticelements(x) = x
function staticelements(t::TrackedArray{<:Any,<:Any,<:Any,VA}) where {VA<:StaticArray}
    return similar_type(VA, eltype(t))(ntuple(i -> t[i], Val(length(VA))))
end

remove_not_tracked(f) = f
remove_not_tracked(f::NotTracked) = f.f
remove_not_tracked(f::Base.RefValue{<:NotTracked}) = Ref(remove_not_tracked(f[]))
remove_not_tracked(f::Base.RefValue{<:NotTracked{<:AbstractArray}}) = remove_not_tracked(f[])
function remove_not_tracked(b::Broadcasted{style}) where {style}
    return Broadcasted{style}(remove_not_tracked(b.f), remove_not_tracked.(b.args), b.axes)
end

function Base.copy(_bc::Broadcasted{<:TrackedStyle})
    # scalars take the scalar derivative rules; ask the axes, since `TrackedStyle{Any}` carries
    # no dimension and `LinearAlgebra` may pass an uninstantiated `Broadcasted`
    if axes(_bc) isa Tuple{}
        return _bc[CartesianIndex()]
    end
    flattened_bc = Base.Broadcast.flatten(remove_not_tracked(_bc))
    f, args = flattened_bc.f, flattened_bc.args
    # only the arguments are seeded, not e.g. a closure's captures, and a `TrackedArray`
    # holds only `Real`s
    if !mayhidetracked(_bc)
        vals = map(value, args)
        if Broadcast.combine_eltypes(f, vals) <: Real
            return ∇broadcast(f, args, vals)
        end
    end
    elargs = map(staticelements, args)
    style = typeof(reduce(Broadcast.result_style, map(untrackedstyle, elargs)))
    return copy(Broadcast.instantiate(Broadcasted{style}(f, elargs)))
end

_no_tracked_dest() = throw(ArgumentError("`TrackedArray`s do not support `setindex!` and cannot be used as a broadcast destination. Use `y = f.(x)` instead."))

Base.copyto!(::TrackedArray, ::Broadcasted{<:TrackedStyle}) = _no_tracked_dest()
Base.copyto!(::TrackedArray, ::Broadcasted{<:DefaultArrayStyle}) = _no_tracked_dest()

getouttype(::TrackedReal{<:Any, D}) where {D} = D
getouttype(::AbstractArray{<:TrackedReal{<:Any, D}}) where {D} = D
getouttype(::Any) = Union{}

deref(x) = x
deref(x::Base.RefValue) = x[]

@generated function splatcall(f, x::NTuple{N,Any}, utargs::T, ::Val{tinds}) where {N, T <: Tuple, tinds}
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
    inds = [i for i in 1:N if T.types[i] <: Union{Real, AbstractArray, Tuple}]
    indsval = :(Val{$(Expr(:tuple, [:($i) for i in inds]...))}())
    maybetracked = Expr(:tuple, [:(args[$i]) for i in inds]...)
    untracked = Expr(:tuple, [:(args[$i]) for i in 1:N if !(i in inds)]...)
    return :($indsval, $maybetracked, $untracked)
end

## A generalization of the broadcasting approach in ReverseDiff for general functions

@inline incr(::Val{k}) where {k} = Val(k + 1)

# argument `i` has partial `slots[i]` (`Val(0)` if untracked), partial `k` belongs to argument
# `positions[k]`. Built on the way out of the recursion to stay constant-folded.
@inline trackedslots(::Tuple{}) = (), (), Val(0)
@inline function trackedslots(args::Tuple)
    slots, positions, n = trackedslots(Base.front(args))
    if istracked(last(args))
        k = incr(n)
        return (slots..., k), (positions..., Val(length(args))), k
    else
        return (slots..., Val(0)), positions, n
    end
end

@inline getat(xs::Tuple, ::Val{i}) where {i} = xs[i]

# `DiffRules` leaves partials such as the order of `besselj` as `NaN`, which `NaN * 0`
# would spread to every slot, so untracked arguments are not dualized
@inline dualize(::Type, ::Val{0}, ::Val, x) = x
@inline function dualize(::Type{T}, ::Val{k}, valP::Val, x) where {T, k}
    return Dual{T}(x, ntuple(j -> j == k, valP))
end

# `f`'s partials in closed form, one per tracked argument
struct KnownPartials{E<:Tuple}
    entries::E
end

const RealOrArray = Union{Real, AbstractArray{<:Real}}

# an argument read at the output's own index has to span the whole output
ifsameshape(c::Contract, x, y) = (x isa Real || y isa Real || axes(x) == axes(y)) ? c : nothing

# partial with respect to argument `i`, or `nothing`. Requiring every argument to be
# `RealOrArray` makes positions in `args` positions in the arguments `splitargs` keeps.
knownpartial(f, ::Val, args) = nothing

knownpartial(::Union{typeof(+), typeof(identity)}, ::Val, ::Tuple{Vararg{RealOrArray}}) =
    Contract(identity, ())
knownpartial(::typeof(-), ::Val{1}, ::Tuple{RealOrArray}) = Contract(-, ())
knownpartial(::typeof(-), ::Val{1}, ::Tuple{RealOrArray,RealOrArray}) = Contract(identity, ())
knownpartial(::typeof(-), ::Val{2}, ::Tuple{RealOrArray,RealOrArray}) = Contract(-, ())

knownpartial(::typeof(*), ::Val{1}, (x, y)::Tuple{RealOrArray,RealOrArray}) =
    ifsameshape(Contract(*, (Val(2),)), x, y)
knownpartial(::typeof(*), ::Val{2}, (x, y)::Tuple{RealOrArray,RealOrArray}) =
    ifsameshape(Contract(*, (Val(1),)), x, y)
# a denominator's partial `-x/y^2` is no argument of the broadcast
knownpartial(::typeof(/), ::Val{1}, (x, y)::Tuple{RealOrArray,RealOrArray}) =
    ifsameshape(Contract(/, (Val(2),)), x, y)
knownpartial(::typeof(\), ::Val{2}, (x, y)::Tuple{RealOrArray,RealOrArray}) =
    ifsameshape(Contract(/, (Val(1),)), x, y)

# `nothing` unless every tracked argument's partial is known
knownpartials(f, args, ::Tuple{}) = ()
function knownpartials(f, args, positions::Tuple)
    rest = knownpartials(f, args, Base.tail(positions))
    if rest === nothing
        return nothing
    end
    p = knownpartial(f, first(positions), args)
    if p === nothing
        return nothing
    else
        return (p, rest...)
    end
end

# marks the `Dual`s `∇broadcast` seeds, so `@skip` can drop them as it drops tracking
struct BroadcastTag{F} end

skipvalue(x::Dual{T}) where {T<:ForwardDiff.Tag{<:BroadcastTag}} = skipvalue(ForwardDiff.value(T, x))

# at least one argument has to be a non-0-dimensional array: `copy` sends the scalar and
# 0-dimensional cases onto the scalar rules instead
@inline function ∇broadcast(f::F, args::Tuple, argvals::Tuple) where {F}
    inds, targs, untracked = splitargs(args)
    _, vals, _ = splitargs(argvals)
    slots, positions, valP = trackedslots(targs)
    # keyed on every argument's type, so an enclosing differentiation's `Dual` makes a newer tag
    tag = ForwardDiff.Tag(BroadcastTag{F}(), typeof(argvals))
    # `broadcast` calls `df` elementwise, so it receives one scalar per argument
    function df(x::Vararg{Any,N}) where {N}
        dx = map((slot, xi) -> dualize(typeof(tag), slot, valP, xi), slots, x)
        return splatcall(f, dx, untracked, inds)
    end
    # known partials leave nothing to read off a `Dual`, so `f` is evaluated undualized
    vf(x::Vararg{Any,N}) where {N} = splatcall(f, x, untracked, inds)
    entries = knownpartials(f, args, positions)
    if entries === nothing
        return trackresults(typeof(tag), broadcast(df, vals...), df, vf, targs, vals)
    else
        return recordresults(typeof(tag), KnownPartials(entries), df, vf, targs, vals)
    end
end

# the cache carries what the replay needs: `df` to recompute the stored partials, or, where
# they are known already, `vf` for the values alone
replaycache(::Type{T}, results::AbstractArray, df, _, _) where {T} =
    (df, map(y -> ForwardDiff.value(T, y), results))
replaycache(::Type, ::KnownPartials, _, vf, vals) = (vf, broadcast(vf, vals...))

# a perturbation riding on an argument ends up in the derivative, so `D` has to be able to hold it
@inline function checkargtags(::Type{D}, ::Type{E}) where {D, E}
    if ForwardDiff.tagtype(E) !== Nothing && !(promote_type(D, E) <: D)
        throw(ArgumentError(LazyString("a broadcast argument with element type ", E,
                                       " carries a perturbation that a derivative of type ", D,
                                       " cannot hold")))
    end
    return nothing
end

# an abstract element type such as `Real` can hide a perturbation
function checkargvalues(::Type{D}, v) where {D}
    if isconcretetype(eltype(v))
        checkargtags(D, eltype(v))
    else
        for x in v
            if x isa Dual
                checkargtags(D, typeof(x))
            elseif !(x isa Real)
                checkargvalues(D, x)
            end
        end
    end
    return nothing
end

@inline function recordresults(::Type{T}, results, df, vf, targs, vals) where {T}
    D = mapreduce(getouttype, promote_type, targs)
    foreach(v -> checkargvalues(D, v), vals)
    g, outvalue = replaycache(T, results, df, vf, vals)
    tp = tape(targs...)
    out = track(outvalue, D, tp)
    _, positions, n = trackedslots(targs)
    bounds = map(p -> index_bound(getat(targs, p), out), positions)
    cache = (results, g, T(), map(tuple, positions, ntuple(Val, n), bounds))
    record!(tp, SpecialInstruction, ∇broadcast, targs, out, cache)
    return out
end

# an enclosing differentiation's tag is constant in our arguments, while one nested inside
# `f` buries our partial; a widened element type is a `Union`, so every member is checked
@inline function checktags(::Type{T}, ::Type{E}) where {T, E}
    if E isa Union
        checktags(T, E.a)
        checktags(T, E.b)
    else
        S = ForwardDiff.tagtype(E)
        if S !== Nothing && !ForwardDiff.:≺(S, T)
            throw(ForwardDiff.DualMismatchError(T, S))
        end
    end
    return nothing
end

# inferred, not read off the results, since a replay can take other branches and writes into
# the results
@inline function trackresults(::Type{T}, results::AbstractArray, df, vf, targs, vals) where {T}
    E = Broadcast.combine_eltypes(df, vals)
    if typeintersect(E, Dual{T}) === Union{}
        checktags(T, E)
        return results
    else
        return recordresults(T, writable(results, E), df, vf, targs, vals)
    end
end

# the replay writes into the results, which a static array may reject
writable(results::AbstractArray, ::Type{E}) where {E} = convert(AbstractArray{E}, results)
writable(results::StaticArray, ::Type{E}) where {E} = copyto!(similar(results, E), results)

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(∇broadcast)})
    input = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    results, _, tag, targets = instruction.cache
    T = typeof(tag)
    partials = reversepartials(results, input)
    foreach(targets) do (p, k, bound)
        x = getat(input, p)
        istracked(x) && _br_add_to_deriv!(T, x, k, output_deriv, partials, bound)
    end
    unseed!(output)
    return nothing
end

struct PartialsWithArgs{E<:Tuple,A<:Tuple}
    entries::E
    args::A
end

# the partials of one reverse pass. Only known partials read argument values, and only of the
# arguments they name, since unwrapping an array of `TrackedReal`s copies it.
reversepartials(results::AbstractArray, _) = results
function reversepartials(p::KnownPartials, input)
    args = map(e -> map(j -> value(getat(input, j)), e.args), p.entries)
    return PartialsWithArgs(p.entries, args)
end

# an argument broadcast to the full output shape needs no index clamping
function _br_add_to_deriv!(::Type{T}, x, slot::Val, out_deriv, results,
                           bound::CartesianIndex) where {T}
    if bound == CartesianIndex(size(out_deriv))
        return _increment_deriv!(T, x, out_deriv, results, slot)
    else
        return _increment_deriv!(T, x, out_deriv, results, slot, bound)
    end
end

_br_add_to_deriv!(::Type{T}, x, slot::Val, out_deriv, results, ::Nothing) where {T} =
    _increment_deriv!(T, x, out_deriv, results, slot, nothing)

# a per-element cache is indexed by the slot, a closed-form one is picked out by it
_increment_deriv!(::Type{T}, x, out_deriv, results, ::Val{k}, bound...) where {T, k} =
    diffresult_increment_deriv!(T, x, out_deriv, results, k, bound...)
_increment_deriv!(::Type, x, out_deriv, p::PartialsWithArgs, ::Val{k}, bound...) where {k} =
    contract_increment_deriv!(x, out_deriv, p.entries[k], p.args[k], bound...)

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(∇broadcast)})
    input, output = instruction.input, instruction.output
    results, df, tag, _ = instruction.cache
    foreach(pull_value!, input)
    _replay!(typeof(tag), value(output), results, df, map(value, input))
    return nothing
end

function _replay!(::Type{T}, out_value, results::AbstractArray, df, vals) where {T}
    broadcast!(df, results, vals...)
    map!(y -> ForwardDiff.value(T, y), out_value, results)
    return nothing
end

# known partials stay valid, so only the values are recomputed
function _replay!(::Type, out_value, results::KnownPartials, vf, vals)
    broadcast!(vf, out_value, vals...)
    return nothing
end
