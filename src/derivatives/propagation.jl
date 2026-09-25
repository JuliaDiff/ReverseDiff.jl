
#=
The code here is mainly deals with propagating derivative information between input and
output values. Usually, this means incrementing/decrementing the input's derivative(s) by an
amount scaled by the output's derivative(s). Sometimes, extra partial information (in the
form of normal scalars/arrays, or `Dual` numbers) needs to accounted for as well.

Often, a function's input and output values are not similarly shaped. To account for these
cases, `diffresult` and `contract` versions of the propagation functions have been implemented.
Accumulations into a scalar derivative go through `sum`, whose pairwise reduction is both more
accurate and faster than a sequential loop.

A lot of the code here is pretty repetitive, and only covers the use cases that actually
arise from the derivative definitions implemented elsewhere in ReverseDiff. At some point,
we should figure out a cleaner, more general implementation pattern that doesn't sacrifice
efficiency.
=#

#############
# utilities #
#############

index_bound(x::Any, ::AbstractArray{T,N}) where {T,N} = nothing

index_bound(x::AbstractArray, ::AbstractArray{T,N}) where {T,N} = CartesianIndex{N}(ntuple(i -> size(x, i), Val(N)))

###################
# increment_deriv #
###################

@inline increment_deriv!(t::TrackedArray, x::AbstractArray, i) = (t.deriv[i] += x[i]; nothing)
@inline increment_deriv!(t::TrackedArray, x::Real, i) = (t.deriv[i] += x; nothing)

# `istracked(t)` only promises that an element *may* be tracked
@inline function increment_deriv!(t::AbstractArray, x::AbstractArray, i)
    ti = t[i]
    istracked(ti) && increment_deriv!(ti, x[i])
    return nothing
end

@inline function increment_deriv!(t::AbstractArray, x::Real, i)
    ti = t[i]
    istracked(ti) && increment_deriv!(ti, x)
    return nothing
end

function increment_deriv!(t::AbstractArray, x)
    for i in eachindex(t)
        increment_deriv!(t, x, i)
    end
    return nothing
end

function increment_deriv!(t::TrackedReal, x::Real)
    pull_deriv!(t)
    t.deriv += _convert(typeof(t.deriv), x)
    push_deriv!(t)
    return nothing
end

###################
# decrement_deriv #
###################

@inline decrement_deriv!(t::TrackedArray, x::AbstractArray, i) = (t.deriv[i] -= x[i]; nothing)
@inline decrement_deriv!(t::TrackedArray, x::Real, i) = (t.deriv[i] -= x; nothing)

# `istracked(t)` only promises that an element *may* be tracked
@inline function decrement_deriv!(t::AbstractArray, x::AbstractArray, i)
    ti = t[i]
    istracked(ti) && decrement_deriv!(ti, x[i])
    return nothing
end

@inline function decrement_deriv!(t::AbstractArray, x::Real, i)
    ti = t[i]
    istracked(ti) && decrement_deriv!(ti, x)
    return nothing
end

function decrement_deriv!(t::AbstractArray, x)
    for i in eachindex(t)
        decrement_deriv!(t, x, i)
    end
    return nothing
end

function decrement_deriv!(t::TrackedReal, x::Real)
    pull_deriv!(t)
    t.deriv -= _convert(typeof(t.deriv), x)
    push_deriv!(t)
    return nothing
end

###############################
# diffresult_increment_deriv! #
###############################

@inline getpartial(::Type, r::DiffResults.ImmutableDiffResult{1,V,Tuple{D}}, p) where {V,D<:AbstractArray} = DiffResults.derivative(r)[p]
@inline getpartial(::Type, r::DiffResults.ImmutableDiffResult{1,V,Tuple{D}}, p) where {V,D<:Number} = DiffResults.derivative(r)
@inline getpartial(::Type{T}, d::ForwardDiff.Dual, p) where {T} = ForwardDiff.partials(T, d, p)
@inline getpartial(::Type, x::Real, p) = zero(x)

# a partial known in closed form: the argument collects `op(seed, args...)` per element, with
# `args` naming broadcast arguments. `op` meets the seed, so `/` forms no reciprocal.
struct Contract{Op,A<:Tuple}
    op::Op
    args::A
end

_at(::Val{j}, i, args) where {j} = _elem(args[j], i)
_elem(v::Real, i) = v
_elem(v::AbstractArray, i) = v[i]

_contract(e::Contract, seed, i, args) = e.op(seed, map(a -> _at(a, i, args), e.args)...)

function diffresult_increment_deriv!(::Type{T}, input::AbstractArray, x::AbstractArray,
                                     results::AbstractArray, p::Int) where {T}
    for i in eachindex(x, results)
        increment_deriv!(input, x[i] * getpartial(T, results[i], p), i)
    end
    return nothing
end

function diffresult_increment_deriv!(::Type{T}, input::AbstractArray, x::AbstractArray,
                                     results::AbstractArray, p::Int,
                                     bound::CartesianIndex) where {T}
    axes(x) == axes(results) ||
        throw(DimensionMismatch("`x` and `results` must have the same indices"))
    for (xi, r, i) in zip(x, results, CartesianIndices(size(x)))
        increment_deriv!(input, xi * getpartial(T, r, p), min(bound, i))
    end
    return nothing
end

function diffresult_increment_deriv!(::Type{T}, input::TrackedReal, x::AbstractArray,
                                     results::AbstractArray, p::Int, ::Nothing) where {T}
    inds = eachindex(x, results)
    isempty(inds) && return nothing
    pull_deriv!(input)
    input.deriv += sum(i -> x[i] * getpartial(T, results[i], p), inds)
    push_deriv!(input)
    return nothing
end

#############################
# contract_increment_deriv! #
#############################

function contract_increment_deriv!(input::AbstractArray, x::AbstractArray, e::Contract,
                                   args::Tuple)
    for i in eachindex(input, x)
        increment_deriv!(input, _contract(e, x[i], i, args), i)
    end
    return nothing
end

function contract_increment_deriv!(input::AbstractArray, x::AbstractArray, e::Contract,
                                   args::Tuple, bound::CartesianIndex)
    for i in CartesianIndices(size(x))
        increment_deriv!(input, _contract(e, x[i], i, args), min(bound, i))
    end
    return nothing
end

function contract_increment_deriv!(input::TrackedReal, x::AbstractArray, e::Contract,
                                   args::Tuple, ::Nothing)
    isempty(x) && return nothing
    pull_deriv!(input)
    input.deriv += sum(i -> _contract(e, x[i], i, args), eachindex(x))
    push_deriv!(input)
    return nothing
end

##############################
# reduction_increment_deriv! #
##############################

function reduction_increment_deriv!(input::AbstractArray, x::AbstractArray,
                                    bound::CartesianIndex)
    for i in CartesianIndices(size(input))
        increment_deriv!(input, x[min(bound, i)], i)
    end
    return nothing
end
