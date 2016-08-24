#############
# TraceNode #
#############

immutable TraceNode{F,I,O,P}
    func::F
    inputs::I
    outputs::O
    partials::P
end

function Base.show(io::IO, node::TraceNode, pad = "")
    println(io, pad, "TraceNode($(node.func)):")
    println(io, pad, "\tout: ", node.outputs)
    print(io,   pad, "\tin:  ", node.inputs)
end

############
# NodeList #
############

immutable NodeList
    isinit::Bool
    head::TraceNode
    tail::NodeList
    NodeList() = new(true)
    NodeList(head::TraceNode, tail::NodeList) = new(false, head, tail)
end

Base.start(list::NodeList) = list
Base.next(list::NodeList, state::NodeList) = (state.head, state.tail)
Base.done(list::NodeList, state::NodeList) = state.isinit

#########
# Trace #
#########

type Trace
    list::NodeList
    Trace() = new(NodeList())
end

function Base.empty!(tr::Trace)
    tr.list = NodeList()
    return tr
end

function Base.push!(tr::Trace, node::TraceNode)
    tr.list = NodeList(node, tr.list)
    return tr
end

function Base.pop!(tr::Trace)
    node = tr.list.head
    tr.list = tr.list.tail
    return node
end

Base.start(tr::Trace) = start(tr.list)
Base.next(tr::Trace, state) = next(tr.list, state)
Base.done(tr::Trace, state) = done(tr.list, state)

function Base.show(io::IO, tr::Trace)
    println(io, "Trace (lower numbers correspond to later nodes): ")
    count = 1
    for node in tr
        show(io, node, "$count ")
        count += 1
        println(io)
    end
end

###########
# record! #
###########

function record!(tr::Nullable{Trace}, func, inputs, outputs, partials = nothing)
    !(isnull(tr)) && push!(get(tr), TraceNode(func, inputs, outputs, partials))
    return nothing
end
