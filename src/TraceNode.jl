immutable TraceNode{F,I,O,M}
    func::F
    inputs::I
    outputs::O
    revdata::M # holds data used in reverse pass (gradients, dual arrays, value arrays, etc.)
end

function Base.show(io::IO, node::TraceNode, pad = "")
    println(io, pad, "TraceNode($(node.func)):")
    println(io, pad, "\tinputs:  ", node.inputs)
    println(io, pad, "\toutputs: ", node.outputs)
    print(io,   pad, "\trevdata:    ", node.revdata)
end

typealias Trace Vector{TraceNode}

function record!(tr::Nullable{Trace}, func, inputs, outputs, revdata = nothing)
    !(isnull(tr)) && push!(get(tr), TraceNode(func, inputs, outputs, revdata))
    return nothing
end
