#############
# TraceNode #
#############

immutable TraceNode{F,I,O}
    func::F
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

reset_trace!{tag}(::tag) = reset_trace!(tag)
reset_trace!{tag}(::Type{tag}) = clear!(get!(TRACE_CACHE, tag, Trace()))::Trace

######################################
# recording the trace (forward pass) #
######################################

# placeholder type for functions nodes whose derivatives
# were calculated during the forward pass
immutable SkipDiffType end

const SKIP_DIFF = SkipDiffType()

@inline record!{tag}(::Type{tag}, inputs, outputs) = record!(tag, SKIP_DIFF, inputs, outputs)

@generated function record!{tag}(::Type{tag}, func, inputs, outputs)
    return quote
        push!($(TRACE_CACHE[tag])::Trace, TraceNode(func, inputs, outputs))
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

# backprop when derivatives are available via the forward pass #
#--------------------------------------------------------------#

backprop_step!(node::TraceNode{SkipDiffType}) = skipdiff_backprop_step!(node.inputs, node.outputs)

# f(::Number)::Number
function skipdiff_backprop_step!{F,S}(input::TraceReal{F,S}, output::TraceReal{F,S,1})
    input.adjoint[] += output.adjoint[] * partials(output, 1)
    return nothing
end

# f(::Number...)::Number
function skipdiff_backprop_step!{F,S,N}(inputs::Tuple, output::TraceReal{F,S,N})
    dual::Dual{N,S} = output.adjoint[] * output.dual
    for i in 1:N
        inputs[i].adjoint[] += partials(dual, i)
    end
    return nothing
end

# f(::AbstractArray)::AbstractArray
function skipdiff_backprop_step!(input::AbstractArray, output::AbstractArray)
    for i in eachindex(input)
        skipdiff_backprop_step!(input[i], output[i])
    end
    return nothing
end

# backprop when derivatives need to be calculated in the reverse pass #
#---------------------------------------------------------------------#

backprop_step!{F}(node::TraceNode{F}) = diff_backprop_step!(node.func, node.inputs, node.outputs)

function increment_adjoint!(output, derivs)
    for i in eachindex(output)
        output[i].adjoint[] += derivs[i]
    end
    return output
end

function diff_backprop_step!{A,B}(::typeof(*), inputs::Tuple{A,B}, output::AbstractArray)
    adj, a, b = adjoint(output), inputs[1], inputs[2]
    increment_adjoint!(a, adj * value(b)')
    increment_adjoint!(b, value(a)' * adj)
    return nothing
end

function diff_backprop_step!{A,B}(::typeof(+), inputs::Tuple{A,B}, output::AbstractArray)
    adj, a, b = adjoint(output), inputs[1], inputs[2]
    increment_adjoint!(a, adj)
    increment_adjoint!(b, adj)
    return nothing
end

function diff_backprop_step!{A,B}(::typeof(-), inputs::Tuple{A,B}, output::AbstractArray)
    adj, a, b = adjoint(output), inputs[1], inputs[2]
    increment_adjoint!(a, adj)
    increment_adjoint!(b, -adj)
    return nothing
end
