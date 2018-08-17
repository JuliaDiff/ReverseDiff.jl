
#=
The code here is mainly deals with propagating derivative information between input and
output values. Usually, this means incrementing/decrementing the input's derivative(s) by an
amount scaled by the output's derivative(s). Sometimes, extra partial information (in the
form of normal scalars/arrays, or `Dual` numbers) needs to accounted for as well.

Often, a function's input and output values are not similarly shaped. To account for these
cases, `broadcast` and `reduce` versions of the propagation functions have been implemented.
These special cases should be decently fast, but some might be prone to numerical error if
the accumulated derivative becomes too large compared to the individual terms being added to
it. This can be overcome by using the divide-and-conquer strategy from Base.mapreducedim,
but that strategy is less cache efficient and more complicated to implement.

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
@inline increment_deriv!(t::AbstractArray, x::AbstractArray, i) = increment_deriv!(t[i], x[i])
@inline increment_deriv!(t::AbstractArray, x::Real, i) = increment_deriv!(t[i], x)

function increment_deriv!(t::AbstractArray, x)
    for i in eachindex(t)
        increment_deriv!(t, x, i)
    end
    return nothing
end

function increment_deriv!(t::TrackedReal, x::Real)
    pull_deriv!(t)
    t.deriv += x
    push_deriv!(t)
    return nothing
end

###################
# decrement_deriv #
###################

@inline decrement_deriv!(t::TrackedArray, x::AbstractArray, i) = (t.deriv[i] -= x[i]; nothing)
@inline decrement_deriv!(t::TrackedArray, x::Real, i) = (t.deriv[i] -= x; nothing)
@inline decrement_deriv!(t::AbstractArray, x::AbstractArray, i) = decrement_deriv!(t[i], x[i])
@inline decrement_deriv!(t::AbstractArray, x::Real, i) = decrement_deriv!(t[i], x)

function decrement_deriv!(t::AbstractArray, x)
    for i in eachindex(t)
        decrement_deriv!(t, x, i)
    end
    return nothing
end

function decrement_deriv!(t::TrackedReal, x::Real)
    pull_deriv!(t)
    t.deriv -= x
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
    for i in eachindex(x)
        increment_deriv!(input, x[i] * getpartial(results[i], p), i)
    end
    return nothing
end

function diffresult_increment_deriv!(input::AbstractArray, x::AbstractArray,
                                     results, p::Int, bound::CartesianIndex)
    for i in CartesianRange(size(x))
        increment_deriv!(input, x[i] * getpartial(results[i], p), min(bound, i))
    end
    return nothing
end

function diffresult_increment_deriv!(input::TrackedReal, x::AbstractArray,
                                     results, p::Int, ::Nothing)
    pull_deriv!(input)
    input_deriv = input.deriv
    for i in eachindex(results)
        input_deriv += x[i] * getpartial(results[i], p)
    end
    input.deriv = input_deriv
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
    for i in CartesianRange(size(x))
        increment_deriv!(input, x[i], min(bound, i))
    end
    return nothing
end

function broadcast_increment_deriv!(input::TrackedReal, x::AbstractArray, ::Nothing)
    pull_deriv!(input)
    input_deriv = input.deriv
    for i in x
        input_deriv += i
    end
    input.deriv = input_deriv
    push_deriv!(input)
    return nothing
end

# with partial array #
#--------------------#

function broadcast_increment_deriv!(input::AbstractArray, x::AbstractArray,
                                    partials::AbstractArray,
                                    input_bound::CartesianIndex,
                                    partials_bound::CartesianIndex)
    for i in CartesianRange(size(x))
        current_deriv = x[i] * partials[min(partials_bound, i)]
        increment_deriv!(input, current_deriv, min(input_bound, i))
    end
    return nothing
end

function broadcast_increment_deriv!(input::TrackedReal, x::AbstractArray,
                                    partials::AbstractArray, ::Nothing, ::CartesianIndex)
    pull_deriv!(input)
    input_deriv = input.deriv
    for i in eachindex(x)
        input_deriv += x[i] * partials[i]
    end
    input.deriv = input_deriv
    push_deriv!(input)
    return nothing
end

# with partial scalar #
#---------------------#

@inline function broadcast_increment_deriv!(input, x, partial::Ref, i, j)
    return broadcast_increment_deriv!(input, x, partial[], i, j)
end

function broadcast_increment_deriv!(input::AbstractArray, x::AbstractArray,
                                    partial::Real, input_bound::CartesianIndex,
                                    ::Nothing)
    for i in CartesianRange(size(x))
        increment_deriv!(input, x[i] * partial, min(input_bound, i))
    end
    return nothing
end

function broadcast_increment_deriv!(input::TrackedReal, x::AbstractArray,
                                    partial::Real, ::Nothing, ::Nothing)
    pull_deriv!(input)
    input_deriv = input.deriv
    for i in eachindex(x)
        input_deriv += x[i] * partial
    end
    input.deriv = input_deriv
    push_deriv!(input)
    return nothing
end

##############################
# broadcast_decrement_deriv! #
##############################

function broadcast_decrement_deriv!(input::AbstractArray, x::AbstractArray,
                                    bound::CartesianIndex)
    for i in CartesianRange(size(x))
        decrement_deriv!(input, x[i], min(bound, i))
    end
    return nothing
end

function broadcast_decrement_deriv!(input::TrackedReal, x::AbstractArray, ::Nothing)
    pull_deriv!(input)
    input_deriv = input.deriv
    for i in x
        input_deriv -= i
    end
    input.deriv = input_deriv
    push_deriv!(input)
    return nothing
end

##############################
# reduction_increment_deriv! #
##############################

function reduction_increment_deriv!(input::AbstractArray, x::AbstractArray,
                                    bound::CartesianIndex)
    for i in CartesianRange(size(input))
        increment_deriv!(input, x[min(bound, i)], i)
    end
    return nothing
end
