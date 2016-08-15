#############
# utilities #
#############

value{T<:TraceReal}(arr::AbstractArray{T}) = value!(similar(arr, valtype(T)), arr)

function value!(out, arr)
    for i in eachindex(out)
        out[i] = value(arr[i])
    end
    return out
end

adjoint{T<:TraceReal}(arr::AbstractArray{T}) = adjoint!(similar(arr, adjtype(T)), arr)

function adjoint!(out, arr)
    for i in eachindex(out)
        out[i] = adjoint(arr[i])
    end
    return out
end

trace_array{tag,S,T}(::tag, ::Type{S}, arr::AbstractArray{T}) = trace_array(tag, S, arr)

function trace_array{tag,S,T}(::Type{tag}, ::Type{S}, arr::AbstractArray{T})
    return copy!(similar(arr, TraceReal{tag,S,T}), arr)
end

trace_output{tag,S}(::tag, ::Type{S}, item) = trace_output(tag, S, item)
trace_output{tag,S}(::Type{tag}, ::Type{S}, arr::AbstractArray) = trace_array(tag, S, arr)
trace_output{tag,S}(::Type{tag}, ::Type{S}, n::Number) = TraceReal{tag,S}(n)

function dual_array{R<:TraceReal,N}(arr::AbstractArray{R}, ::Type{Val{N}}, i)
    tag, S, T = tagtype(R), adjtype(R), valtype(R)
    out_partials = Partials{N,T}((z = zeros(T,N); z[i] = one(T); (z...)))
    out = similar(arr, Dual{N,T})
    for j in eachindex(out)
        out[j] = Dual{N,T}(value(arr[j]), out_partials)
    end
    return out
end

###########################
# optimized array methods #
###########################

# broadcast/map #
#---------------#

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

# unary functions #
#-----------------#

for f in (:-, :inv, :det)
    for A in (:AbstractArray, :AbstractMatrix, :Array, :Matrix)
        @eval function Base.$(f){tag,S,T}(x::$(A){TraceReal{tag,S,T}})
            out = trace_output(tag, S, $(f)(value(x)))
            record!(tag, $(f), x, out)
            return out
        end
    end
end

for A in (:AbstractArray, :Array)
    @eval function Base.sum{tag,S,T}(x::$(A){TraceReal{tag,S,T}})
        result = zero(T)
        for t in x
            result += value(t)
        end
        out = TraceReal{tag,S}(result)
        record!(tag, sum, x, out)
        return out
    end
end

# binary functions #
#------------------#

for f in (:-, :+, :*,
          :A_mul_Bt, :At_mul_B, :At_mul_Bt,
          :A_mul_Bc, :Ac_mul_B, :Ac_mul_Bc)
    @eval function Base.$(f){tag,S,A,B}(a::AbstractMatrix{TraceReal{tag,S,A}},
                                        b::AbstractMatrix{TraceReal{tag,S,B}})
        out = trace_array(tag, S, $(f)(value(a), value(b)))
        record!(tag, $(f), (a, b), out)
        return out
    end
end

# in-place A_mul_B family #
#-------------------------#

for (f!, f) in ((:A_mul_B!, :*),
                (:A_mul_Bt!, :A_mul_Bt), (:At_mul_B!, :At_mul_B), (:At_mul_Bt!, :At_mul_Bt),
                (:A_mul_Bc!, :A_mul_Bc), (:Ac_mul_B!, :Ac_mul_B), (:Ac_mul_Bc!, :Ac_mul_Bc))
    @eval function Base.$(f!){tag,S,Y,A,B}(out::AbstractMatrix{TraceReal{tag,S,Y}},
                                           a::AbstractMatrix{TraceReal{tag,S,A}},
                                           b::AbstractMatrix{TraceReal{tag,S,B}})
        copy!(out, $(f)(value(a), value(b)))
        record!(tag, $(f), (a, b), out)
        return out
    end
end
