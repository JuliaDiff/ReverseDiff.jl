function gradient!{F}(out, f::F, x, tracex = trace_array(F, eltype(out), x))
    trace = reset_trace!(F)
    f(tracex)
    backprop!(trace)
    for i in eachindex(out)
        out[i] = tracex[i].adjoint[]
    end
    return out
end
