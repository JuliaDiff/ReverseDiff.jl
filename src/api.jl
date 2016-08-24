######################
# gradient/gradient! #
######################

function gradient{F}(f::F, x, tracex = trace_array(F, eltype(x), x))
    trace = reset_trace!(F)
    seed!(f(tracex))
    backprop!(trace)
    return adjoint(tracex)
end

function gradient!{F}(out, f::F, x, tracex = trace_array(F, eltype(out), x))
    trace = reset_trace!(F)
    seed!(f(tracex))
    backprop!(trace)
    return adjoint!(out, tracex)
end

######################
# jacobian/jacobian! #
######################

function load_jacobian!(out, tracex, y, trace)
    for i in eachindex(y)
        n = y[i]
        seed!(n)
        backprop!(trace)
        for j in eachindex(tracex)
            m = tracex[j]
            out[i, j] = adjoint(m)
            unseed!(m)
        end
        unseed!(n)
    end
    return out
end

function jacobian{F}(f::F, x, tracex = trace_array(F, eltype(x), x))
    trace = reset_trace!(F)
    y = f(tracex)
    out = similar(y, eltype(x), length(y), length(x))
    return load_jacobian!(out, tracex, y, trace)
end

function jacobian!{F}(out, f::F, x, tracex = trace_array(F, eltype(out), x))
    trace = reset_trace!(F)
    y = f(tracex)
    load_jacobian!(reshape(out, length(y), length(x)), tracex, y, trace)
    return out
end
