function Base.map{F,S,T,N}(f, arr::AbstractArray{TraceReal{F,S,T,N}})
    R = promote_op(f, T)
    out = similar(arr, TraceReal{F,S,1,R})
    out_partials = one(Partials{1,T})
    for i in eachindex(out)
        out[i] = TraceReal{F,S,1,R}(f(Dual{1,R}(value(arr[i]), out_partials)))
    end
    record!(F, arr, out)
    return out
end

function duals{F,S,A,B,M,N}(a::AbstractArray{TraceReal{F,S,A,M}}, b::AbstractArray{TraceReal{F,S,B,N}})
    x = similar(a, Dual{2,A})
    y = similar(b, Dual{2,B})
    x_partials = Partials{2,A}((one(A), zero(A)))
    y_partials = Partials{2,B}((zero(B), one(B)))
    for i in eachindex(x)
        x[i] = Dual{2,A}(value(a[i]), x_partials)
    end
    for i in eachindex(y)
        y[i] = Dual{2,B}(value(b[i]), y_partials)
    end
    return x, y
end

function Base.:*{F,S,A,B,M,N}(a::Array{TraceReal{F,S,A,M}}, b::Array{TraceReal{F,S,B,N}})
    x, y = duals(a, b)
    z = x * y
    R = numtype(eltype(z))
    out = similar(z, TraceReal{F,S,2,R})
    for i in eachindex(out)
        out[i] = TraceReal{F,S}(z[i])
    end
    record!(F, (a, b), out)
    return out
end
