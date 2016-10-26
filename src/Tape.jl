#################
# TapeNode/Tape #
#################

immutable TapeNode{F,I,O,M}
    func::F
    inputs::I
    outputs::O
    cache::M # holds data used in reverse pass (gradients, dual arrays, value arrays, etc.)
end

typealias Tape Vector{TapeNode}

function record!(tp::Nullable{Tape}, func, inputs, outputs, cache = nothing)
    if !(isnull(tp))
        node = TapeNode(func, capture(inputs), capture(outputs), cache)
        push!(get(tp), node)
    end
    return nothing
end

# Ensure that the external state is "captured" so that external
# reference-breaking (e.g. destructive assignment) doesn't break
# internal TapeNode state.
@inline capture(state) = state
@inline capture(state::AbstractArray) = copy(state)
@inline capture(state::Tuple{Vararg{Number}}) = state
@inline capture(state::Tuple) = map(capture, state)

function Base.:(==)(a::TapeNode, b::TapeNode)
    return (a.func == b.func &&
            a.inputs == b.inputs &&
            a.outputs == b.outputs &&
            a.cache == b.cache)
end

################################################
# reverse pass (backpropagation over the tape) #
################################################

function reverse_pass!(tape::Tape)
    for i in length(tape):-1:1
        reverse_step!(tape[i])
    end
    return nothing
end

# The *_reverse_step! functions are implemented for relevant functions in the `derivatives` folder
reverse_step!(node::TapeNode{Void}) = scalar_reverse_step!(node.inputs, node.outputs, node.cache)
reverse_step!(node::TapeNode) = special_reverse_step!(node.func, node.inputs, node.outputs, node.cache)

###################
# Pretty Printing #
###################

compactrepr(x, _...) = repr(x)

function compactrepr(x::AbstractArray, _...)
    xrepr = repr(x)
    m = match(r"\[.*?\]", xrepr)
    return isa(m, Void) ? xrepr : m.match
end

function compactrepr(t::Tuple, pad = "")
    io = IOBuffer()
    print(io, "(")
    print(io, compactrepr(t[1]))
    for i in drop(t, 1)
        println(io, ",")
        print(io, " ", pad, compactrepr(i))
    end
    print(io, ")")
    return takebuf_string(io)
end

function Base.show(io::IO, node::TapeNode, pad = "")
    println(io, pad, "TapeNode($(node.func)):")
    # length of the prefix strings below (e.g. "  inputs:  ")
    # plus whatever extra padding was passed in
    valpad = repeat(" ", 11 + length(pad))
    println(io, pad, "  inputs:  ", compactrepr(node.inputs, valpad))
    println(io, pad, "  outputs: ", compactrepr(node.outputs, valpad))
    print(io,   pad, "  cache:   ", compactrepr(node.cache, valpad))
end

Base.display(tp::Tape) = show(STDOUT, tp)

function Base.show(io::IO, tp::Tape)
    println("$(length(tp))-element Vector{TapeNode}:")
    for node in tp
        println("↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓")
        show(io, node)
        println()
    end
end
