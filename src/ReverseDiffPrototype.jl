module ReverseDiffPrototype

using ForwardDiff
import Calculus

######################
# Tape-related Types #
######################

# value types #
#-------------#

type TapeReal{T<:Real} <: Real
    val::T
    adj::T
end

TapeReal(val::Real) = TapeReal(val, zero(val))

type TapeArray{T,N,A} <: AbstractArray{T,N}
    val::A
    adj::A
end

TapeArray(val) = TapeArray(val, zeros(val))

typealias TapeValue Union{TapeReal, TapeArray}

@inline tapeval(val::Real) = TapeReal(val)
@inline tapeval(val) = TapeArray(val)

@inline value(n::TapeValue) = n.val
@inline adjoint(n::TapeValue) = n.adj

# tape storage #
#--------------#

immutable TapeNode{op,I,O}
    inputs::I
    outputs::O
end

TapeNode{op,I,O}(::Type{Val{op}}, inputs::I, output::O) = TapeNode{op,I,O}(inputs, output)

const TAPE = Vector{TapeNode}()

function record!{op}(::Type{Val{op}}, inputs, outputs)
    push!(TAPE, TapeNode(Val{op}, inputs, outputs))
    return outputs
end

########################
# Function Overloading #
########################
# Strategy for supporting a function `f`
#   1. overload `f` to store itself, it's inputs, and outputs to the TAPE, or add it to the
#      no-op list if it's a `backprop!` no-op
#   2. overload `backprop!` to propogate the derivative from the outputs to the inputs

# unary number functions
for (op, _) in Calculus.symbolic_derivatives_1arg()
    @eval begin
        @inline Base.$(op)(n::TapeValue) = record!(Val{$(op)}, n, tapeval($(op)(value(n))))
    end
end

function backprop!{op,T<:TapeReal,S<:TapeReal}(node::TapeNode{op,T,S})
    node.inputs.adj += adjoint(node.outputs) * ForwardDiff.derivative(op, value(node.inputs))
    return node
end

# binary functions
for op in (:(Base.:*), :(Base.:/), :(Base.:+), :(Base.:-))
    @eval begin
        @inline function $(op)(a::TapeReal, b::TapeReal)
            out = tapeval($(op)(value(a), value(b)))
            return record!(Val{$(op)}, tuple(a, b), out)
        end
    end
end

function backprop!{T1<:TapeReal,T2<:TapeReal,S<:TapeReal}(node::TapeNode{+,Tuple{T1,T2},S})
    adj = adjoint(node.outputs)
    node.inputs[1].adj += adj
    node.inputs[2].adj += adj
    return node
end

function backprop!{T1<:TapeReal,T2<:TapeReal,S<:TapeReal}(node::TapeNode{-,Tuple{T1,T2},S})
    adj = adjoint(node.outputs)
    node.inputs[1].adj += adj
    node.inputs[2].adj -= adj
    return node
end

function backprop!{T1<:TapeReal,T2<:TapeReal,S<:TapeReal}(node::TapeNode{*,Tuple{T1,T2},S})
    adj = adjoint(node.outputs)
    node.inputs[1].adj += adj * value(node.inputs[2])
    node.inputs[2].adj += adj * value(node.inputs[1])
    return node
end

function backprop!{T1<:TapeReal,T2<:TapeReal,S<:TapeReal}(node::TapeNode{/,Tuple{T1,T2},S})
    adj = adjoint(node.outputs)
    x, y = value(node.inputs[1]), value(node.inputs[2])
    node.inputs[1].adj += adj / y
    node.inputs[2].adj += (-adj * x) / (y*y)
    return node
end

# no-ops w.r.t. back-propagation
for op in (:(Base.:<), :(Base.:>), :(Base.:(==)), :(Base.:(<=)), :(Base.:(>=)))
    @eval begin
        @inline $(op)(a::TapeReal, b::TapeReal) = tapeval($(op)(value(a), value(b)))
    end
end

#######
# API #
#######

seed!(n::TapeReal) = (n.adj = (one(n.adj)); return n)
seed!(node::TapeNode) = (seed!(node.outputs); return node)
seed!(tape::Vector{TapeNode}) = (seed!(last(tape)); return tape)

function backprop!(tape::Vector{TapeNode})
    while !(isempty(tape))
        backprop!(pop!(tape))
    end
    return nothing
end

function gradient!(out, f, x, tapevec = Vector{TapeReal{eltype(x)}}(length(x)))
    for i in eachindex(x)
        tapevec[i] = TapeReal(x[i])
    end
    f(tapevec)
    seed!(TAPE)
    backprop!(TAPE)
    for i in eachindex(out)
        out[i] = adjoint(tapevec[i])
    end
    return out
end

end # module
