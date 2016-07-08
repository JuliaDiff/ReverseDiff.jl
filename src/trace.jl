#############
# TraceNode #
#############

immutable TraceNode{I,O}
    inputs::I
    outputs::O
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

function clear!(trace::Trace)
    trace.list = NodeList()
    return trace
end

function Base.push!(trace::Trace, node::TraceNode)
    trace.list = NodeList(node, trace.list)
    return trace
end

Base.start(trace::Trace) = start(trace.list)
Base.next(trace::Trace, state) = next(trace.list, state)
Base.done(trace::Trace, state) = done(trace.list, state)

function seed!(trace::Trace)
    for t in trace.list.head.outputs
        t.adjoint[] = one(adjtype(t))
    end
    return trace
end

#######################
# trace pointer cache #
#######################

const TRACE_CACHE = ObjectIdDict()

reset_trace!{F}(::F) = reset_trace!(F)
reset_trace!{F}(::Type{F}) = clear!(get!(TRACE_CACHE, F, Trace()))::Trace

######################################
# recording the trace (forward pass) #
######################################

@generated function record!{F}(::Type{F}, inputs, outputs)
    return quote
        push!($(TRACE_CACHE[F])::Trace, TraceNode(inputs, outputs))
        return nothing
    end
end

#################################################
# backpropagation over the trace (reverse pass) #
#################################################

function backprop!(trace::Trace)
    seed!(trace)
    for node in trace
        backprop_step!(node)
    end
    return nothing
end

backprop_step!(node::TraceNode) = backprop_step!(node.inputs, node.outputs)

# f(::Number)::Number
function backprop_step!{F,S}(input::TraceReal{F,S}, output::TraceReal{F,S,1})
    input.adjoint[] += output.adjoint[] * partials(output, 1)
    return nothing
end

# f(::Number...)::Number
function backprop_step!{F,S,N}(inputs::Tuple, output::TraceReal{F,S,N})
    dual::Dual{N,S} = output.adjoint[] * output.dual
    for i in 1:N
        inputs[i].adjoint[] += partials(dual, i)
    end
    return nothing
end

# f(::AbstractArray)::AbstractArray
function backprop_step!(input::AbstractArray, output::AbstractArray)
    for i in eachindex(input)
        backprop_step!(input[i], output[i])
    end
    return nothing
end

# f(::AbstractArray, ::AbstractArray)::AbstractArray
function backprop_step!{A,B}(inputs::Tuple{A,B}, output::AbstractArray)
    a, b = inputs
    for i in eachindex(input)
        backprop_step!((a[i], b[i]), output[i])
    end
    return nothing
end
