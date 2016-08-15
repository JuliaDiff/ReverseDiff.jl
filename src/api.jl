function gradient{F}(f::F, x, tracex = trace_array(F, eltype(x), x))
    trace = reset_trace!(F)
    f(tracex)
    backprop!(trace)
    return adjoint(tracex)
end

function gradient!{F}(out, f::F, x, tracex = trace_array(F, eltype(out), x))
    trace = reset_trace!(F)
    f(tracex)
    backprop!(trace)
    return adjoint!(out, tracex)
end
