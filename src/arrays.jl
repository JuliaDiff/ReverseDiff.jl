#############
# accessors #
#############

ForwardDiff.value{T}(arr::AbstractArray{T}) = value!(similar(arr, valtype(T)), arr)

function value!(out, arr)
    for i in eachindex(out)
        out[i] = value(arr[i])
    end
    return out
end

adjoint{T}(arr::AbstractArray{T}) = adjoint!(similar(arr, adjtype(T)), arr)

function adjoint!(out, arr)
    for i in eachindex(out)
        out[i] = adjoint(arr[i])
    end
    return out
end

###################
# tape selection #
###################

function tape(arr::AbstractArray)
    for t in arr
        !(isnull(tape(t))) && return tape(t)
    end
    return Nullable{Tape}()
end

tape(a, b::AbstractArray) = isnull(tape(a)) ? tape(b) : tape(a)
