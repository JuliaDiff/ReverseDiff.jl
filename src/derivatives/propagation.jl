
#=
The code here is mainly deals with propagating derivative information between input and
output values. Usually, this means incrementing/decrementing the input's derivative(s) by an
amount scaled by the output's derivative(s). Sometimes, extra partial information (in the
form of normal scalars/arrays, or `Dual` numbers) needs to accounted for as well.

Often, a function's input and output values are not similarly shaped. To account for these
cases, `broadcast` and `reduce` versions of the propagation functions have been implemented.
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

@inline getpartial(r::DiffResults.ImmutableDiffResult{1,V,Tuple{D}}, p) where {V,D<:AbstractArray} = DiffResults.derivative(r)[p]
@inline getpartial(r::DiffResults.ImmutableDiffResult{1,V,Tuple{D}}, p) where {V,D<:Number} = DiffResults.derivative(r)

function diffresult_increment_deriv!(input::AbstractArray, x::AbstractArray,
                                     results, p::Int)
    for i in eachindex(x, results)
        increment_deriv!(input, x[i] * getpartial(results[i], p), i)
    end
    return nothing
end

function diffresult_increment_deriv!(input::AbstractArray, x::AbstractArray,
                                     results, p::Int, bound::CartesianIndex)
    axes(x) == axes(results) ||
        throw(DimensionMismatch("`x` and `results` must have the same indices"))
    for (xi, r, i) in zip(x, results, CartesianIndices(size(x)))
        increment_deriv!(input, xi * getpartial(r, p), min(bound, i))
    end
    return nothing
end

function diffresult_increment_deriv!(input::TrackedReal, x::AbstractArray,
                                     results, p::Int, ::Nothing)
    inds = eachindex(x, results)
    isempty(inds) && return nothing
    pull_deriv!(input)
    input.deriv += sum(i -> x[i] * getpartial(results[i], p), inds)
    push_deriv!(input)
    return nothing
end

##############################
# broadcast_increment_deriv! #
##############################

# without partials #
#------------------#

function broadcast_increment_deriv!(input::AbstractArray, x::AbstractArray,
                                    bound::CartesianIndex)
    for (xi, i) in zip(x, CartesianIndices(size(x)))
        increment_deriv!(input, xi, min(bound, i))
    end
    return nothing
end

function broadcast_increment_deriv!(input::TrackedReal, x::AbstractArray, ::Nothing)
    isempty(x) && return nothing
    pull_deriv!(input)
    input.deriv += sum(x)
    push_deriv!(input)
    return nothing
end

# with partials #
#---------------#

@inline broadcast_increment_deriv!(input, x, partials, input_bound, partials_bound) =
    _broadcast_increment_deriv!(*, input, x, partials, input_bound, partials_bound)

@inline broadcast_increment_div_deriv!(input, x, partials, input_bound, partials_bound) =
    _broadcast_increment_deriv!(/, input, x, partials, input_bound, partials_bound)

# with partial array #
#--------------------#

function _broadcast_increment_deriv!(op::F, input::AbstractArray, x::AbstractArray,
                                     partials::AbstractArray,
                                     input_bound::CartesianIndex,
                                     partials_bound::CartesianIndex) where {F}
    for (xi, i) in zip(x, CartesianIndices(size(x)))
        current_deriv = op(xi, partials[min(partials_bound, i)])
        increment_deriv!(input, current_deriv, min(input_bound, i))
    end
    return nothing
end

function _broadcast_increment_deriv!(op::F, input::TrackedReal, x::AbstractArray,
                                     partials::AbstractArray, ::Nothing,
                                     ::CartesianIndex) where {F}
    inds = eachindex(x, partials)
    isempty(inds) && return nothing
    pull_deriv!(input)
    input.deriv += sum(i -> op(x[i], partials[i]), inds)
    push_deriv!(input)
    return nothing
end

# with partial scalar #
#---------------------#

function _broadcast_increment_deriv!(op::F, input::AbstractArray, x::AbstractArray,
                                     partial::Real, input_bound::CartesianIndex,
                                     ::Nothing) where {F}
    for (xi, i) in zip(x, CartesianIndices(size(x)))
        increment_deriv!(input, op(xi, partial), min(input_bound, i))
    end
    return nothing
end

##############################
# broadcast_decrement_deriv! #
##############################

function broadcast_decrement_deriv!(input::AbstractArray, x::AbstractArray,
                                    bound::CartesianIndex)
    for (xi, i) in zip(x, CartesianIndices(size(x)))
        decrement_deriv!(input, xi, min(bound, i))
    end
    return nothing
end

function broadcast_decrement_deriv!(input::TrackedReal, x::AbstractArray, ::Nothing)
    isempty(x) && return nothing
    pull_deriv!(input)
    input.deriv -= sum(x)
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
