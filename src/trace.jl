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

# placeholder for functions whose derivatives
# were calculated during the forward pass
immutable UseDual end

const USE_DUAL = UseDual()

@inline record!{tag}(::Type{tag}, inputs, outputs) = record!(tag, USE_DUAL, inputs, outputs)

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

backprop_step!(node::TraceNode{UseDual}) = dual_backprop_step!(node.inputs, node.outputs)
backprop_step!{F}(node::TraceNode{F}) = no_dual_backprop_step!(node.func, node.inputs, node.outputs)

# scalar functions #
#------------------#

# f(::Number)::Number
function dual_backprop_step!{tag,S,T}(input::TraceReal{tag,S}, output::TraceReal{tag,S,Dual{1,T}})
    input.adjoint[] += output.adjoint[] * partials(output.value, 1)
    return nothing
end

# f(::Number...)::Number
function dual_backprop_step!{tag,S,N,T}(inputs::Tuple, output::TraceReal{tag,S,Dual{N,T}})
    dual::Dual{N,S} = output.adjoint[] * output.value
    for i in 1:N
        inputs[i].adjoint[] += partials(dual, i)
    end
    return nothing
end

# elementwise functions (e.g. broadcast, map) #
#---------------------------------------------#

# f.(::AbstractArray)::AbstractArray
function dual_backprop_step!(input::AbstractArray, output::AbstractArray)
    for i in eachindex(output)
        dual_backprop_step!(input[i], output[i])
    end
    return nothing
end

# f.(::AbstractArray, ::AbstractArray)::AbstractArray
function dual_backprop_step!(inputs::NTuple{2}, output::AbstractArray)
    a, b = inputs
    for i in eachindex(output)
        dual_backprop_step!((a[i], b[i]), output[i])
    end
    return nothing
end

# f.(::AbstractArray, ::AbstractArray, ::AbstractArray)::AbstractArray
function dual_backprop_step!(inputs::NTuple{3}, output::AbstractArray)
    a, b, c = inputs
    for i in eachindex(output)
        dual_backprop_step!((a[i], b[i], c[i]), output[i])
    end
    return nothing
end

# functions whose derivatives didn't get calculated in the forward pass #
#-----------------------------------------------------------------------#

function no_dual_backprop_step!{A,B}(::typeof(*), inputs::Tuple{A,B}, output::AbstractArray)
    adj, a, b = adjoint(output), inputs[1], inputs[2]
    increment_adjoint!(a, adj * value(b)')
    increment_adjoint!(b, value(a)' * adj)
    return nothing
end

function no_dual_backprop_step!{A,B}(::typeof(+), inputs::Tuple{A,B}, output::AbstractArray)
    increment_adjoint!(inputs[1], output)
    increment_adjoint!(inputs[2], output)
    return nothing
end

function no_dual_backprop_step!{A,B}(::typeof(-), inputs::Tuple{A,B}, output::AbstractArray)
    increment_adjoint!(inputs[1], output)
    decrement_adjoint!(inputs[2], output)
    return nothing
end

function no_dual_backprop_step!(::typeof(-), input::AbstractArray, output::AbstractArray)
    decrement_adjoint!(input, output)
    return nothing
end

# utilities #
#-----------#

function decrement_adjoint!{T<:TraceReal}(input, output::AbstractArray{T})
    for i in eachindex(input)
        input[i].adjoint[] -= output[i].adjoint[]
    end
    return input
end

function increment_adjoint!{T<:TraceReal}(input, output::AbstractArray{T})
    for i in eachindex(input)
        input[i].adjoint[] += output[i].adjoint[]
    end
    return input
end

function increment_adjoint!(input, derivs)
    for i in eachindex(input)
        input[i].adjoint[] += derivs[i]
    end
    return input
end
