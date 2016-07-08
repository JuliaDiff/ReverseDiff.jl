function backprop_step!{T,S,P}(t::TraceReal{T,S,P})
    parent_adjoints = t.parent_adjoints
    dual::Dual{P,S} = t.adjoint[] * t.dual
    for i in 1:P
        parent_adjoints[i][] += partials(dual, i)
    end
    return nothing
end

function backprop!(t::TraceReal)
    queue = TraceReal[t]
    while !(isempty(queue))
        current = pop!(queue)
        backprop_step!(current)
        for node in current.parents
            unshift!(queue, node)
        end
    end
end

function trace_input_array{S}(x, ::Type{S})
    T = eltype(x)
    arr = similar(x, TraceReal{T,S,0})
    for i in eachindex(arr)
        arr[i] = TraceReal{T,S,0}(x[i])
    end
    return arr
end

function gradient!(out, f, x, tracex = trace_input_array(x, eltype(out)))
    result = f(tracex)
    result.adjoint[] = one(result.adjoint[])
    backprop!(result)
    for i in eachindex(out)
        out[i] = tracex[i].adjoint[]
    end
    return out
end
