#################
# TapeNode/Tape #
#################

abstract NodeKind
abstract Scalar <: NodeKind
abstract Special <: NodeKind

immutable TapeNode{K<:NodeKind,F,I,O,C}
    func::F
    inputs::I
    outputs::O
    cache::C # holds data/buffers used in forward/reverse passes
end

@inline function TapeNode{K<:NodeKind,F,I,O,C}(::Type{K}, func::F, inputs::I, outputs::O, cache::C)
    return TapeNode{K,F,I,O,C}(func, inputs, outputs, cache)
end

typealias Tape Vector{TapeNode}

function record_node!{K<:NodeKind}(tp::Nullable{Tape}, ::Type{K}, func, inputs, outputs, cache = nothing)
    if !(isnull(tp))
        node = TapeNode(K, func, capture(inputs), capture(outputs), cache)
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

function Base.:(==){A,B}(a::TapeNode{A}, b::TapeNode{B})
    return (A === B &&
            a.func == b.func &&
            a.inputs == b.inputs &&
            a.outputs == b.outputs &&
            a.cache == b.cache)
end

########################
# forward/reverse pass #
########################

function forward_pass!(tape::Tape)
    for node in tape
        forward_step!(node)
    end
    return nothing
end

function reverse_pass!(tape::Tape)
    for i in length(tape):-1:1
        reverse_step!(tape[i])
    end
    return nothing
end

# *_forward_step! and *_reverse_step! methods are implemented in the `derivatives` folder
function forward_step!(node::TapeNode{Scalar})
    scalar_forward_step!(node.func, node.inputs, node.outputs, node.cache)
    unseed!(node.outputs)
end

function forward_step!(node::TapeNode{Special})
    special_forward_step!(node.func, node.inputs, node.outputs, node.cache)
    unseed!(node.outputs)
end

reverse_step!(node::TapeNode{Scalar}) = scalar_reverse_step!(node.inputs, node.outputs, node.cache)
reverse_step!(node::TapeNode{Special}) = special_reverse_step!(node.func, node.inputs, node.outputs, node.cache)

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
    if length(t) > 1
        for i in t[2:end]
            println(io, ",")
            print(io, " ", pad, compactrepr(i))
        end
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
