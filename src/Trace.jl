immutable TraceNode{F,I,O,P}
    func::F
    inputs::I
    outputs::O
    partials::P
end

function Base.show(io::IO, node::TraceNode, pad = "")
    println(io, pad, "TraceNode($(node.func)):")
    println(io, pad, "\tinputs:   ", node.inputs)
    println(io, pad, "\toutputs:  ", node.outputs)
    print(io,   pad, "\tpartials: ", node.partials)
end

typealias Trace Vector{TraceNode}

function record!(tr::Nullable{Trace}, func, inputs, outputs, partials = nothing)
    !(isnull(tr)) && push!(get(tr), TraceNode(func, inputs, outputs, partials))
    return nothing
end
