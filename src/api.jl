function trace_input_array{F,S}(::F, x, ::Type{S})
    T = eltype(x)
    arr = similar(x, TraceReal{F,S,0,T})
    for i in eachindex(arr)
        arr[i] = TraceReal{F,S,0,T}(x[i])
    end
    return arr
end

function gradient!{F}(out, f::F, x, tracex = trace_input_array(f, x, eltype(out)))
    trace = reset_trace!(F)
    f(tracex)
    backprop!(trace)
    for i in eachindex(out)
        out[i] = tracex[i].adjoint[]
    end
    return out
end
