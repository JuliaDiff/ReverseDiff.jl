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

# an argument is seeded itself, or elementwise, so its own tracked values are in plain sight; a
# `Ref` is handed to `f` whole, so nothing inside one is
mayhidetracked(b::F) where {F} = _mayhidetracked(F)
mayhidetracked(::AbstractArray{F}) where {F} = _mayhidetracked(F)
mayhidetracked(::Base.RefValue{F}) where {F} = _mayhidetracked(F)
mayhidetracked(::AbstractArray{<:Real}) = false
mayhidetracked(::Real) = false
mayhidetracked(b::Type) = false
mayhidetracked(b::ForwardOptimize) = mayhidetracked(b.f)
mayhidetracked(b::SkipOptimize) = mayhidetracked(b.f)
mayhidetracked(b::Broadcasted) = mayhidetracked(b.f) || any(mayhidetracked, b.args)

# below the argument nothing is seeded, so a tracked value is hidden wherever it sits: recurse
# into what a container holds rather than ask about the container
_mayhidetracked(::Type{<:NotTracked}) = false
_mayhidetracked(::Type{<:AbstractArray{F}}) where {F} = _mayhidetracked(F)
@generated function _mayhidetracked(::Type{F}) where {F}
    # `fieldcount` errors for types without a definite number of fields, such as
    # `Type{T}` and abstract types; be conservative in that case.
    hasfields = try
        fieldcount(F) > 0
    catch
        true
    end
    return :($hasfields)
end

struct TrackedStyle{N} <: AbstractArrayStyle{N} end

(::Type{<:TrackedStyle})(::Val{N}) where {N} = TrackedStyle{N}()

Broadcast.BroadcastStyle(::Type{<:TrackedArray{V,D,N}}) where {V,D,N} = TrackedStyle{N}()
Broadcast.BroadcastStyle(::Type{<:TrackedReal}) = TrackedStyle{0}()
Broadcast.BroadcastStyle(::Type{<:AbstractArray{<:TrackedReal,N}}) where {N} = TrackedStyle{N}()

# `AbstractArrayStyle{Any}` carries `Any` as its dimension, which `max` cannot compare
_maxdim(M::Int, N::Int) = max(M, N)
_maxdim(::Type{Any}, ::Int) = Any
_maxdim(::Int, ::Type{Any}) = Any
_maxdim(::Type{Any}, ::Type{Any}) = Any

# tracked values must stay tracked, so take precedence over every other array style
Broadcast.BroadcastStyle(::TrackedStyle{M}, ::AbstractArrayStyle{N}) where {M,N} =
    TrackedStyle{_maxdim(M, N)}()

# resolve the overlap with `Base`'s three `DefaultArrayStyle` rules, preserving their results
Broadcast.BroadcastStyle(::TrackedStyle{M}, ::DefaultArrayStyle{N}) where {M,N} =
    TrackedStyle{_maxdim(M, N)}()
Broadcast.BroadcastStyle(::TrackedStyle{N}, ::DefaultArrayStyle{N}) where {N} = TrackedStyle{N}()
Broadcast.BroadcastStyle(::TrackedStyle{Any}, ::DefaultArrayStyle) = TrackedStyle{Any}()

# the untracked style decides where the fallback re-dispatches, and a `CuArray` backing a
# `TrackedArray` has to keep its own
recur_value(xs) = xs
recur_value(xs::Union{TrackedReal, TrackedArray, AbstractArray{<:TrackedReal}}) = recur_value(value(xs))

remove_not_tracked(f) = f
remove_not_tracked(f::NotTracked) = f.f
remove_not_tracked(f::Base.RefValue{<:NotTracked}) = Ref(remove_not_tracked(f[]))
remove_not_tracked(f::Base.RefValue{<:NotTracked{<:AbstractArray}}) = remove_not_tracked(f[])
function remove_not_tracked(b::Broadcasted{style}) where {style}
    return Broadcasted{style}(remove_not_tracked(b.f), remove_not_tracked.(b.args), b.axes)
end

# scalars take `Base`'s 0-dimensional route onto the scalar derivative rules; `instantiate`
# leaves an `AbstractArrayStyle{0}` without axes, so 0 dimensions shows up either way
Base.copy(bc::Broadcasted{<:TrackedStyle, <:Union{Nothing, Tuple{}}}) = bc[CartesianIndex()]

function Base.copy(_bc::Broadcasted{<:TrackedStyle})
    bc = remove_not_tracked(_bc)
    flattened_bc = Base.Broadcast.flatten(bc)
    f, args = flattened_bc.f, flattened_bc.args
    # only the arguments are seeded, so a tracked value reaching `f` by another route, such
    # as a closure capturing one, has to be traced scalar-wise
    if mayhidetracked(_bc)
        axs = flattened_bc.axes
        style = typeof(Broadcast.combine_styles(map(recur_value, args)...))
        return copy(Broadcasted{style, typeof(axs), typeof(f), typeof(args)}(f, args, axs))
    else
        return ∇broadcast(f, args...)
    end
end

_no_tracked_dest() = throw(ArgumentError("`TrackedArray`s do not support `setindex!` and cannot be used as a broadcast destination. Use `y = f.(x)` instead."))

Base.copyto!(::TrackedArray, ::Broadcasted{<:TrackedStyle}) = _no_tracked_dest()
Base.copyto!(::TrackedArray, ::Broadcasted{<:DefaultArrayStyle}) = _no_tracked_dest()

getouttype(::TrackedReal{<:Any, D}) where {D} = D
getouttype(::TrackedArray{<:Any, D}) where {D} = D
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
    RealOrArray = Union{Real, AbstractArray}
    inds = [i for i in 1:N if T.types[i] <: RealOrArray]
    indsval = :(Val{$(Expr(:tuple, [:($i) for i in inds]...))}())
    maybetracked = Expr(:tuple, [:(args[$i]) for i in inds]...)
    untracked = Expr(:tuple, [:(args[$i]) for i in 1:N if !(i in inds)]...)
    return :($indsval, $maybetracked, $untracked)
end

## A generalization of the broadcasting approach in ReverseDiff for general functions

@inline incr(::Val{k}) where {k} = Val(k + 1)

# `slots[i]` indexes argument `i`'s partial and is `Val(0)` where the argument is untracked;
# the second value counts the tracked arguments. Both are built on the way out of the
# recursion: a count passed into it would stop being constant-folded after a few arguments,
# leaving the partials a runtime-sized tuple.
@inline trackedslots(::Tuple{}) = (), Val(0)
@inline function trackedslots(args::Tuple)
    slots, n = trackedslots(Base.tail(args))
    istracked(first(args)) || return (Val(0), slots...), n
    k = incr(n)
    return (k, slots...), k
end

# `DiffRules` leaves partials such as the order of `besselj` as `NaN`, which `NaN * 0`
# would spread to every slot, so untracked arguments are not dualized
@inline dualize(::Type, ::Val{0}, ::Val, x) = x
@inline function dualize(::Type{T}, ::Val{k}, valP::Val, x) where {T, k}
    return Dual{T}(x, ntuple(j -> j == k, valP))
end

# only seeded arguments make up the `Dual` value type, so the tag follows `slots`
@inline dualvaltype(::Val{0}, v) = Union{}
@inline dualvaltype(::Val, v) = eltype(v)

# `f`'s partials in closed form, one entry per tracked argument in slot order
struct KnownPartials{E<:Tuple}
    entries::E
end

# `trackedslots` numbers the tracked arguments from the last, so drop the untracked ones and
# reverse what remains
trackedentries(::Tuple{}, ::Tuple{}) = ()
trackedentries(slots::Tuple{Val{0},Vararg{Any}}, entries::Tuple) =
    trackedentries(Base.tail(slots), Base.tail(entries))
trackedentries(slots::Tuple, entries::Tuple) =
    (trackedentries(Base.tail(slots), Base.tail(entries))..., first(entries))

# an argument read at the output's own index has to span the whole output
_sameshape(x, y) = x isa Real || y isa Real || axes(x) == axes(y)

# one entry per argument, or `nothing` where `f`'s partials depend on the point. Annotating
# every argument is what lets an entry name one: those are exactly the ones `splitargs` keeps.
knownpartials(f, args...) = nothing

# `+` and `-` are affine in their arguments jointly
knownpartials(::Union{typeof(+), typeof(identity)},
              args::Vararg{Union{Real, AbstractArray{<:Real}}}) =
    map(_ -> Contract(identity, ()), args)

knownpartials(::typeof(-), arg::Union{Real, AbstractArray{<:Real}}) = (Contract(-, ()),)

knownpartials(::typeof(-), x::Union{Real, AbstractArray{<:Real}},
              y::Union{Real, AbstractArray{<:Real}}) =
    (Contract(identity, ()), Contract(-, ()))

# `*`, `/` and `\` are multilinear, so a partial is another argument, which the reverse pass
# reads from the instruction's own input where `record!` has captured it. A tracked
# denominator is excluded, its partial `-x/y^2` being no argument of the broadcast.
function knownpartials(::typeof(*), x::Union{Real, AbstractArray{<:Real}},
                       y::Union{Real, AbstractArray{<:Real}})
    if _sameshape(x, y)
        return (Contract(*, (Val(2),)), Contract(*, (Val(1),)))
    else
        return nothing
    end
end

function knownpartials(::typeof(/), x::Union{Real, AbstractArray{<:Real}},
                       y::Union{Real, AbstractArray{<:Real}})
    if !istracked(y) && _sameshape(x, y)
        return (Contract(/, (Val(2),)), nothing)
    else
        return nothing
    end
end

function knownpartials(::typeof(\), x::Union{Real, AbstractArray{<:Real}},
                       y::Union{Real, AbstractArray{<:Real}})
    if !istracked(x) && _sameshape(x, y)
        return (nothing, Contract(/, (Val(1),)))
    else
        return nothing
    end
end

broadcastresults(::Nothing, slots, df, vals) = broadcast(df, vals...)
broadcastresults(entries::Tuple, slots, df, vals) =
    KnownPartials(trackedentries(slots, entries))

# at least one argument has to be a non-0-dimensional array: `copy` sends the scalar and
# 0-dimensional cases down `Base`'s `TrackedStyle{0}` route onto the scalar rules instead
@inline function ∇broadcast(f::F, args::Vararg{Any}) where {F}
    inds, targs, untracked = splitargs(args)
    D = mapreduce(getouttype, promote_type, targs)
    slots, valP = trackedslots(targs)
    vals = map(value, targs)
    # one tag for the whole broadcast keeps `results` concretely typed
    T = typeof(ForwardDiff.Tag(f, reduce(promote_type, map(dualvaltype, slots, vals))))
    # `broadcast` calls `df` elementwise, so it receives one scalar per argument
    function df(x::Vararg{Any,N}) where {N}
        dx = map((slot, xi) -> dualize(T, slot, valP, xi), slots, x)
        return splatcall(f, dx, untracked, inds)
    end
    # known partials leave nothing to read off a `Dual`, so `f` is evaluated undualized
    vf(x::Vararg{Any,N}) where {N} = splatcall(f, x, untracked, inds)
    entries = knownpartials(f, args...)
    return trackresults(T, broadcastresults(entries, slots, df, vals), df, vf, targs, D)
end

# the cache carries what the replay needs: `df` to recompute the stored partials, or, where
# they are known already, `vf` for the values alone
replaycache(::Type{T}, results::AbstractArray, df, vf, targs) where {T} =
    (df, map(y -> ForwardDiff.value(T, y), results))
replaycache(::Type, ::KnownPartials, df, vf, targs) =
    (vf, broadcast(vf, map(value, targs)...))

# the seed is contracted with an argument, so a perturbation riding on one ends up in the
# derivative and `D` has to be able to hold it; a widened element type is a `Union`
@inline function checkargtags(::Type{D}, ::Type{E}) where {D, E}
    if E isa Union
        checkargtags(D, E.a)
        checkargtags(D, E.b)
    elseif ForwardDiff.tagtype(E) !== Nothing && !(promote_type(D, E) <: D)
        throw(ArgumentError(LazyString("a broadcast argument with element type ", E,
                                       " carries a perturbation that a derivative of type ", D,
                                       " cannot hold")))
    end
    return nothing
end

# `df` hands back `f`'s own result, so `results` keeps the element type `Base` would produce
@inline function recordresults(::Type{T}, results, df, vf, targs, ::Type{D}) where {T, D}
    foreach(t -> checkargtags(D, eltype(value(t))), targs)
    g, outvalue = replaycache(T, results, df, vf, targs)
    tp = tape(targs...)
    out = track(outvalue, D, tp)
    cache = (results, g, T(), map(t -> index_bound(t, out), targs))
    record!(tp, SpecialInstruction, ∇broadcast, targs, out, cache)
    return out
end

@inline trackresults(::Type{T}, results::AbstractArray{<:Dual{T}}, df, vf, targs,
                     ::Type{D}) where {T, D} = recordresults(T, results, df, vf, targs, D)

@inline trackresults(::Type{T}, results::KnownPartials, df, vf, targs,
                     ::Type{D}) where {T, D} = recordresults(T, results, df, vf, targs, D)

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

# a type-unstable `f`, such as one with an integer literal branch, leaves an abstract
# element type that can still hide a `Dual{T}`
@inline function trackresults(::Type{T}, results::AbstractArray, df, vf, targs,
                              ::Type{D}) where {T, D}
    if typeintersect(eltype(results), Dual{T}) === Union{}
        checktags(T, eltype(results))
        return results
    else
        return recordresults(T, results, df, vf, targs, D)
    end
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(∇broadcast)})
    input = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    results, _, tag, bounds = instruction.cache
    T = typeof(tag)
    slots, _ = trackedslots(input)
    map((x, slot, bound) ->
            _br_add_to_deriv!(T, x, slot, output_deriv, results, bound, input),
        input, slots, bounds)
    unseed!(output)
    return nothing
end

# a per-element cache is indexed by the slot, a closed-form one is picked out by it
selectpartial(results, ::Val{k}, args) where {k} = (results, k)
selectpartial(p::KnownPartials, ::Val{k}, args) where {k} = (p.entries[k], args)

_br_add_to_deriv!(::Type, _, ::Val{0}, _, _, ::CartesianIndex, _) = nothing
_br_add_to_deriv!(::Type, _, ::Val{0}, _, _, ::Nothing, _) = nothing

# an argument broadcast to the full output shape needs no index clamping
function _br_add_to_deriv!(::Type{T}, x, slot::Val{k}, out_deriv, results,
                           bound::CartesianIndex, args) where {T, k}
    results, sel = selectpartial(results, slot, args)
    if bound == CartesianIndex(size(out_deriv))
        return diffresult_increment_deriv!(T, x, out_deriv, results, sel)
    else
        return diffresult_increment_deriv!(T, x, out_deriv, results, sel, bound)
    end
end

function _br_add_to_deriv!(::Type{T}, x, slot::Val{k}, out_deriv, results, ::Nothing,
                           args) where {T, k}
    results, sel = selectpartial(results, slot, args)
    return diffresult_increment_deriv!(T, x, out_deriv, results, sel, nothing)
end

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
