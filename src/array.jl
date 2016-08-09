#############
# utilities #
#############

function value!(out, arr)
    for i in eachindex(out)
        out[i] = value(arr[i])
    end
    return out
end

value{tag,S,N,T}(arr::AbstractArray{TraceReal{tag,S,N,T}}) = value!(similar(arr, T), arr)

function adjoint!(out, arr)
    for i in eachindex(out)
        out[i] = arr[i].adjoint[]
    end
    return out
end

adjoint{tag,S,N,T}(arr::AbstractArray{TraceReal{tag,S,N,T}}) = adjoint!(similar(arr, S), arr)

######################
# overloaded methods #
######################

# derivatives calculated in forward pass #
#----------------------------------------#

function Base.map{tag,S,N,T}(f, arr::AbstractArray{TraceReal{tag,S,N,T}})
    R = Base.promote_op(f, T)
    out = similar(arr, TraceReal{tag,S,1,R})
    out_partials = one(Partials{1,T})
    for i in eachindex(out)
        out[i] = TraceReal{tag,S,1,R}(f(Dual{1,R}(value(arr[i]), out_partials)))
    end
    record!(tag, arr, out)
    return out
end

# derivatives calculated in reverse pass #
#----------------------------------------#

for f in (:*, :-, :+)
    @eval function Base.$(f){A<:TraceReal,B<:TraceReal}(a::AbstractMatrix{A}, b::AbstractMatrix{B})
        tag = tagtype(A)
        result_value = $(f)(value(a), value(b))
        out = map(TraceReal{tag,adjtype(A),0,eltype(result_value)}, result_value)
        record!(tag, $(f), (a, b), out)
        return out
    end
end
