#############
# utilities #
#############

function value{T<:TraceReal}(arr::AbstractArray{T})
    out = similar(arr, valtype(T))
    for i in eachindex(out)
        out[i] = value(arr[i])
    end
    return out
end

function adjoint{T<:TraceReal}(arr::AbstractArray{T})
    out = similar(arr, adjtype(T))
    for i in eachindex(out)
        out[i] = arr[i].adjoint[]
    end
    return out
end

function trace_array{tag,S,T}(::Type{tag}, ::Type{S}, arr::AbstractArray{T})
    out = similar(arr, TraceReal{tag,S,T})
    for i in eachindex(out)
        out[i] = TraceReal{tag,S,T}(arr[i])
    end
    return out
end

function dual_array{R<:TraceReal,N}(arr::AbstractArray{R}, ::Type{Val{N}}, i)
    tag, S, T = tagtype(R), adjtype(R), valtype(R)
    out_partials = Partials{N,T}((z = zeros(T,N); z[i] = one(T); (z...)))
    out = similar(arr, Dual{N,T})
    for j in eachindex(out)
        out[j] = Dual{N,T}(value(arr[j]), out_partials)
    end
    return out
end

######################
# overloaded methods #
######################

for g in (:broadcast, :map)
    @eval begin
        function Base.$(g){tag,S,T,N}(f, x::AbstractArray{TraceReal{tag,S,T},N})
            dual = $(g)(f, dual_array(x, Val{1}, 1))
            out = trace_array(tag, S, dual)
            record!(tag, x, out)
            return out
        end

        function Base.$(g){tag,S,T1,T2,N}(f,
                                          x1::AbstractArray{TraceReal{tag,S,T1},N},
                                          x2::AbstractArray{TraceReal{tag,S,T2},N})
            dual1 = dual_array(x1, Val{2}, 1)
            dual2 = dual_array(x2, Val{2}, 2)
            out = trace_array(tag, S, $(g)(f, dual1, dual2))
            record!(tag, (x1, x2), out)
            return out
        end

        function Base.$(g){tag,S,T1,T2,T3,N}(f,
                                             x1::AbstractArray{TraceReal{tag,S,T1},N},
                                             x2::AbstractArray{TraceReal{tag,S,T2},N},
                                             x3::AbstractArray{TraceReal{tag,S,T3},N})
            dual1 = dual_array(x1, Val{3}, 1)
            dual2 = dual_array(x2, Val{3}, 2)
            dual3 = dual_array(x2, Val{3}, 3)
            out = trace_array(tag, S, $(g)(f, dual1, dual2, dual3))
            record!(tag, (x1, x2, x3), out)
            return out
        end
    end
end

for f in (:-, :+, :*)
    @eval function Base.$(f){tag,S,A,B}(a::AbstractMatrix{TraceReal{tag,S,A}},
                                        b::AbstractMatrix{TraceReal{tag,S,B}})
        out = trace_array(tag, S, $(f)(value(a), value(b)))
        record!(tag, $(f), (a, b), out)
        return out
    end
end

function Base.:-{tag,S,T}(x::AbstractArray{TraceReal{tag,S,T}})
    out = trace_array(tag, S, -value(x))
    record!(tag, -, x, out)
    return out
end
