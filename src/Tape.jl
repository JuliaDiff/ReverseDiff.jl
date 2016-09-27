immutable TapeNode{F,I,O,M}
    func::F
    inputs::I
    outputs::O
    cache::M # holds data used in reverse pass (gradients, dual arrays, value arrays, etc.)
end

function Base.show(io::IO, node::TapeNode, pad = "")
    println(io, pad, "TapeNode($(node.func)):")
    println(io, pad, "\tinputs:  ", node.inputs)
    println(io, pad, "\toutputs: ", node.outputs)
    print(io,   pad, "\tcache:    ", node.cache)
end

typealias Tape Vector{TapeNode}

function record!(tp::Nullable{Tape}, func, inputs, outputs, cache = nothing)
    !(isnull(tp)) && push!(get(tp), TapeNode(func, inputs, outputs, cache))
    return nothing
end
