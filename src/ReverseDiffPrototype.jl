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

typealias TapeValue{tag,T} Union{TapeReal{tag,T}, Array{TapeReal{tag,T}}}

TapeReal{tag<:Tag,T}(::Type{tag}, val::T) = TapeReal{tag,T}(val)

@inline value(n::TapeReal) = n.val
@inline adjoint(n::TapeReal) = n.adj

# we can't use map here because we'll overload it below to record all ops
@inline function value{tag,T<:Real}(A::Array{TapeReal{tag,T}})
    out = similar(A, T)
    for i in eachindex(A)
        out[i] = value(A[i])
    end
    return out
end
@inline function adjoint{tag,T<:Real}(A::Array{TapeReal{tag,T}})
    out = similar(A, T)
    for i in eachindex(A)
        out[i] = adjoint(A[i])
    end
    return out
end

@inline numtype{tag,T}(::Type{TapeReal{tag,T}}) = T
@inline tagtype{tag,T}(::Type{TapeReal{tag,T}}) = tag

Base.convert{tag,T<:Real}(::Type{TapeReal{tag,T}}, n::Real) = TapeReal{tag,T}(T(n), zero(T))
Base.convert{tag,T<:Real}(::Type{TapeReal{tag,T}}, n::TapeReal) = TapeReal{tag,T}(value(n), adjoint(n))
Base.convert{tag,T<:Real}(::Type{TapeReal{tag,T}}, n::TapeReal{tag,T}) = n

Base.promote_rule{tag,A<:Real,B<:Real}(::Type{A}, ::Type{TapeReal{tag,B}}) = TapeReal{tag,promote_type(A,B)}
Base.promote_rule{tag,A<:Real,B<:Real}(::Type{TapeReal{tag,A}}, ::Type{TapeReal{tag,B}}) = TapeReal{tag,promote_type(A,B)}

# tape storage #
#--------------#

abstract AbstractTapeNode

immutable InitialNode <: AbstractTapeNode end

immutable TapeNode{F,I,O} <: AbstractTapeNode
    f::F
    inputs::I
    outputs::O
    parent::AbstractTapeNode
end

type TapeHead
    node::AbstractTapeNode
end

record!(f, inputs, outputs) = error("no tape defined")

function seed!(head::TapeHead)
    head.node.outputs.adj = one(head.node.outputs.adj)
    return nothing
end

function backprop!(head::TapeHead)
    init_node = InitialNode()
    seed!(head)
    current_node = head.node
    while current_node !== init_node
        backprop_rule!(current_node)
        current_node = current_node.parent
    end
    return nothing
end

function increment_adjoint!{T<:TapeReal, S<:Real}(x::Array{T}, y::Array{S})
    for i in eachindex(x)
        x[i].adj += numtype(T)(y[i])
    end
end

####################
# Math Overloading #
####################

# unary functions #
#-----------------#

function backprop_rule!{F,T<:TapeReal,S<:TapeReal}(node::TapeNode{F,T,S})
    node.inputs.adj += adjoint(node.outputs) * ForwardDiff.derivative(node.f, value(node.inputs))
    return nothing
end

for (f, _) in Calculus.symbolic_derivatives_1arg()
    @eval begin
        @inline Base.$(f){tag}(n::TapeReal{tag}) = record!($(f), n, TapeReal(tag, $(f)(value(n))))
    end
end

Base.:-{tag}(n::TapeReal{tag}) = record!(-, n, TapeReal(tag, -value(n)))

function backprop_rule!{T<:TapeReal,S<:TapeReal}(node::TapeNode{typeof(-),T,S})
    node.inputs.adj += -adjoint(node.outputs)
    return nothing
end

Base.abs{tag}(n::TapeReal{tag}) = record!(abs, n, TapeReal(tag, abs(value(n))))

function backprop_rule!{T<:TapeReal,S<:TapeReal}(node::TapeNode{typeof(abs),T,S})
    node.inputs.adj += adjoint(node.outputs) * sign(value(node.inputs))
    return nothing
end

# binary functions #
#------------------#

function backprop_rule!{F,T<:Tuple,S<:TapeReal}(node::TapeNode{F,T,S})
    outadj, invals = adjoint(node.outputs), map(value, node.inputs)
    unary_f = x -> node.f(x...)
    grad = Vector{numtype(S)}(length(invals))
    ForwardDiff.gradient!(grad, unary_f, invals)
    for i in eachindex(invals)
        node.inputs[i].adj += outadj * grad[i]
    end
    return nothing
end

for f in (:*, :/, :+, :-)
    grad = Calculus.differentiate(:($f(x, y)), [:x, :y])
    @eval begin
        @inline function Base.$(f){tag}(a::TapeReal{tag}, b::TapeReal{tag})
            out = TapeReal(tag, $(f)(value(a), value(b)))
            return record!($(f), tuple(a, b), out)
        end

        function backprop_rule!{T1<:TapeReal,T2<:TapeReal,S<:TapeReal}(node::TapeNode{typeof($f),Tuple{T1,T2},S})
            adj, x, y = adjoint(node.outputs), value(node.inputs[1]), value(node.inputs[2])
            node.inputs[1].adj += adj * $(grad[1])
            node.inputs[2].adj += adj * $(grad[2])
            return nothing
        end
    end
end

# backprop! no-ops #
#------------------#

for f in (:(Base.:<), :(Base.:>), :(Base.:(==)), :(Base.:(<=)), :(Base.:(>=)))
    @eval begin
        @inline $(f){tag}(a::TapeReal{tag}, b::TapeReal{tag}) = TapeReal(tag, $(f)(value(a), value(b)))
    end
end

function Base.map{tag,T<:Real}(f, A::Array{TapeReal{tag,T}})
    out = similar(A)
    for i in eachindex(A)
        out[i] = TapeReal{tag,T}(f(A[i].val))
    end
    record!(map, tuple(f, A), out)
end

function backprop_rule!{F<:Function,T<:Array,S<:Array}(node::TapeNode{typeof(map),Tuple{F,T},S})
    adj = adjoint(node.outputs)
    f = node.inputs[1]
    df = x -> ForwardDiff.derivative(f, x)
    A = node.inputs[2]
    increment_adjoint!(A, adj .* map(df, value(A)))
    return nothing
end

#######
# API #
#######

gradient(f, x) = gradient(f, Input{eltype(x),length(x)})

function gradient{F,T,N}(f::F, ::Type{Input{T,N}})
    head = TapeHead(InitialNode())
    tapevec = Vector{TapeReal{Tag{F,T,N},T}}(N)
    eval(quote
        function ReverseDiffPrototype.record!{T}(f, inputs, output::TapeValue{Tag{$F,$T,$N},T})
            head = $head
            head.node = TapeNode(f, inputs, output, head.node)
            return output
        end
    end)
    return (out, x) -> begin
        load_tapevec!(tapevec, x)
        f(tapevec)
        backprop!(head)
        head.node = InitialNode()
        load_adjoint!(out, tapevec)
        return out
    end
end

function load_tapevec!{F,T,N}(tapevec::Vector{TapeReal{Tag{F,T,N},T}}, x)
    for i in eachindex(x)
        tapevec[i] = ReverseDiffPrototype.TapeReal{Tag{F,T,N},T}(x[i], zero(T))
    end
    return tapevec
end

function load_adjoint!(out, tapevec)
    for i in eachindex(out)
        out[i] = adjoint(tapevec[i])
    end
    return out
end

end # module
