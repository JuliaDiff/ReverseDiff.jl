module ReverseDiffPrototype

using ForwardDiff
import Calculus

abstract Input{T,N}
abstract Tag{F,T,N}

export Input

######################
# Tape-related Types #
######################

# value types #
#-------------#

type TapeReal{tag,T<:Real} <: Real
    val::T
    adj::T
end

TapeReal{tag<:Tag,T}(::Type{tag}, val::T) = TapeReal{tag,T}(val)

@inline value(n::TapeReal) = n.val
@inline adjoint(n::TapeReal) = n.adj

Base.convert{tag,T<:Real}(::Type{TapeReal{tag,T}}, n::Real) = TapeReal{tag,T}(T(n), zero(T))
Base.convert{tag,T<:Real}(::Type{TapeReal{tag,T}}, n::TapeReal) = TapeReal{tag,T}(value(n), adjoint(n))
Base.convert{tag,T<:Real}(::Type{TapeReal{tag,T}}, n::TapeReal{tag,T}) = n

Base.promote_rule{tag,A<:Real,B<:Real}(::Type{A}, ::Type{TapeReal{tag,B}}) = TapeReal{tag,promote_type(A,B)}
Base.promote_rule{tag,A<:Real,B<:Real}(::Type{TapeReal{tag,A}}, ::Type{TapeReal{tag,B}}) = TapeReal{tag,promote_type(A,B)}

# tape storage #
#--------------#

immutable TapeNode{F,I,O}
    f::F
    inputs::I
    outputs::O
end

record!(f, inputs, outputs) = error("no tape defined")

####################
# Math Overloading #
####################
# Strategy for supporting a function `f`
#   1. overload `f` to store itself, it's inputs, and outputs to the TAPE, or add it to the
#      no-op list if it's a `backprop!` no-op
#   2. overload `backprop!` to propogate the derivative from the outputs to the inputs

# unary number functions
for (f, _) in Calculus.symbolic_derivatives_1arg()
    @eval begin
        @inline Base.$(f){tag}(n::TapeReal{tag}) = record!($(f), n, TapeReal(tag, $(f)(value(n))))
    end
end

function backprop!{F,T<:TapeReal,S<:TapeReal}(node::TapeNode{F,T,S})
    node.inputs.adj += adjoint(node.outputs) * ForwardDiff.derivative(node.f, value(node.inputs))
    return node
end

# binary functions
for f in (:(Base.:*), :(Base.:/), :(Base.:+), :(Base.:-))
    @eval begin
        @inline function $(f){tag}(a::TapeReal{tag}, b::TapeReal{tag})
            out = TapeReal(tag, $(f)(value(a), value(b)))
            return record!($(f), tuple(a, b), out)
        end
    end
end

function backprop!{T1<:TapeReal,T2<:TapeReal,S<:TapeReal}(node::TapeNode{typeof(+),Tuple{T1,T2},S})
    adj = adjoint(node.outputs)
    node.inputs[1].adj += adj
    node.inputs[2].adj += adj
    return node
end

function backprop!{T1<:TapeReal,T2<:TapeReal,S<:TapeReal}(node::TapeNode{typeof(-),Tuple{T1,T2},S})
    adj = adjoint(node.outputs)
    node.inputs[1].adj += adj
    node.inputs[2].adj -= adj
    return node
end

function backprop!{T1<:TapeReal,T2<:TapeReal,S<:TapeReal}(node::TapeNode{typeof(*),Tuple{T1,T2},S})
    adj = adjoint(node.outputs)
    node.inputs[1].adj += adj * value(node.inputs[2])
    node.inputs[2].adj += adj * value(node.inputs[1])
    return node
end

function backprop!{T1<:TapeReal,T2<:TapeReal,S<:TapeReal}(node::TapeNode{typeof(/),Tuple{T1,T2},S})
    adj = adjoint(node.outputs)
    x, y = value(node.inputs[1]), value(node.inputs[2])
    node.inputs[1].adj += adj / y
    node.inputs[2].adj += (-adj * x) / (y*y)
    return node
end

# no-ops w.r.t. back-propagation
for f in (:(Base.:<), :(Base.:>), :(Base.:(==)), :(Base.:(<=)), :(Base.:(>=)))
    @eval begin
        @inline $(f){tag}(a::TapeReal{tag}, b::TapeReal{tag}) = TapeReal(tag, $(f)(value(a), value(b)))
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

gradient(f, x) = gradient(f, Input{eltype(x),length(x)})

function gradient{F,T,N}(f::F, ::Type{Input{T,N}})
    tape = Vector{TapeNode}()
    tapevec = Vector{TapeReal{Tag{F,T,N},T}}(N)
    eval(quote
        function ReverseDiffPrototype.record!(f, inputs, output::TapeReal{Tag{$F,$T,$N}})
            push!($tape, TapeNode(f, inputs, output))
            return output
        end
        (out, x) -> begin
            tape = $tape
            tapevec = $tapevec
            copy!(tapevec, x)
            $(f)(tapevec)
            ReverseDiffPrototype.backprop!(ReverseDiffPrototype.seed!(tape))
            for i in eachindex(out)
                out[i] = adjoint(tapevec[i])
            end
            return out
        end
    end)
end

end # module
