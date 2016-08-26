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
# trace selection #
###################

function trace(arr::AbstractArray)
    for t in arr
        !(isnull(trace(t))) && return trace(t)
    end
    return Nullable{Trace}()
end

trace(a::AbstractArray, b::AbstractArray) = isnull(trace(a)) ? trace(b) : trace(a)
trace(a::AbstractArray, b::AbstractArray, c::AbstractArray) = isnull(trace(a)) ? trace(b, c) : trace(a)
