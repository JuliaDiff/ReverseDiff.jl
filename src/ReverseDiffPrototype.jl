module ReverseDiffPrototype

using ForwardDiff
import Calculus

######################
# Tape-related Types #
######################

# tape storage #
#--------------#

immutable TapeNode{F,I,O}
    f::F
    input::I
    output::O
    parent
end

type TapeHead
    node::TapeNode
end

const INITIAL_NODE = TapeNode(nothing, nothing, nothing, nothing)

isinit(node::TapeNode) = node === INITIAL_NODE
isinit(head::TapeHead) = isinit(head.node)

# value types #
#-------------#

type TapeReal{F,T<:Real} <: Real
    val::T
    adj::T
end

# always provide type parameters when using these aliases for overloading
typealias TapeArray{F,T,N} AbstractArray{TapeReal{F,T},N}

@inline numtype{F,T}(::Type{TapeReal{F,T}}) = T
@inline value(n::TapeReal) = n.val
@inline adjoint(n::TapeReal) = n.adj
@inline numtype{F,T}(::TapeArray{F,T}) = T
function value{F,T}(a::TapeArray{F,T})
    out = similar(a, T)
    for idx in eachindex(a)
        out[idx] = value(a[idx])
    end
    return out
end
function adjoint{F,T}(a::TapeArray{F,T})
    out = similar(a, T)
    for idx in eachindex(a)
        out[idx] = adjoint(a[idx])
    end
    return out
end

Base.convert{F,A<:Real,B<:Real}(::Type{TapeReal{F}}, val::A, adj::B) = TapeReal{F,promote_type(A,B)}(val, adj)
Base.convert{F,T<:Real}(::Type{TapeReal{F}}, val::T) = TapeReal{F,T}(val, zero(val))
Base.convert{F,T<:Real}(::Type{TapeReal{F,T}}, val::Real) = TapeReal{F,T}(T(val), zero(T))
Base.convert{F,T<:Real}(::Type{TapeReal{F,T}}, n::TapeReal) = TapeReal{F,T}(value(n), adjoint(n))
Base.convert{F,T<:Real}(::Type{TapeReal{F,T}}, n::TapeReal{F,T}) = n

Base.promote_rule{F,A<:Real,B}(::Type{A}, ::Type{TapeReal{F,B}}) = TapeReal{F,promote_type(A,B)}
Base.promote_rule{F,A,B}(::Type{TapeReal{F,A}}, ::Type{TapeReal{F,B}}) = TapeReal{F,promote_type(A,B)}

function seed!(head::TapeHead)
    head.node.output.adj = one(head.node.output.adj)
    return nothing
end

function backprop!(head::TapeHead)
    seed!(head)
    current_node = head.node
    while !(isinit(current_node))
        backprop_rule!(current_node)
        current_node = current_node.parent
    end
    return nothing
end

tape_array{F}(f::F, x) = similar(x, TapeReal{F,eltype(x)})

function load_tape_array!{F,T,N}(tapearr::TapeArray{F,T,N}, arr)
    for i in eachindex(tapearr)
        tapearr[i] = TapeReal{F}(arr[i])
    end
    return tapearr
end

function load_adjoint_array!(out, tapearr)
    for i in eachindex(out)
        out[i] = adjoint(tapearr[i])
    end
    return out
end

function increment_adjoint!(out, arr)
    for i in eachindex(out)
        out[i].adj += arr[i]
    end
    return out
end

# recording mechanisms #
#----------------------#

const TAPE_HEAD_CACHE = ObjectIdDict()

function reset_head!{F}(cache, ::F)
    if haskey(cache, F)
        head = cache[F]::TapeHead
        head.node = INITIAL_NODE
    else
        head = TapeHead(INITIAL_NODE)
        cache[F] = head
    end
    return head::TapeHead
end

@generated function record!{F,T}(f, inputs, output::TapeReal{F,T})
    head = TAPE_HEAD_CACHE[F]::TapeHead
    return quote
        $(head).node = TapeNode(f, inputs, output, $(head).node)
        return output
    end
end

@generated function record!{F,T,N}(f, inputs, output::TapeArray{F,T,N})
    head = TAPE_HEAD_CACHE[F]::TapeHead
    return quote
        $(head).node = TapeNode(f, inputs, output, $(head).node)
        return output
    end
end

####################
# Math Overloading #
####################

# unary functions #
#-----------------#

for (f, _) in Calculus.symbolic_derivatives_1arg()
    @eval begin
        @inline Base.$(f){F}(n::TapeReal{F}) = record!($(f), n, TapeReal{F}($(f)(value(n))))
    end
end

Base.:-{F}(n::TapeReal{F}) = record!(-, n, TapeReal{F}(-value(n)))
Base.abs{F}(n::TapeReal{F}) = record!(abs, n, TapeReal{F}(abs(value(n))))

function backprop_rule!{T<:TapeReal,S<:TapeReal}(node::TapeNode{typeof(-),T,S})
    node.input.adj += -adjoint(node.output)
    return nothing
end

function backprop_rule!{T<:TapeReal,S<:TapeReal}(node::TapeNode{typeof(abs),T,S})
    node.input.adj += adjoint(node.output) * sign(value(node.input))
    return nothing
end

# binary functions #
#------------------#

for f in (:*, :/, :+, :-)
    grad = Calculus.differentiate(:($f(x, y)), [:x, :y])
    @eval begin
        @inline Base.$(f){F}(a::TapeReal{F}, b::TapeReal{F}) = record!($(f), tuple(a, b), TapeReal{F}($(f)(value(a), value(b))))

        @inline Base.$(f){F,T<:Real}(A::TapeArray{F,T,2}, B::TapeArray{F,T,2}) = record!($(f), tuple(A, B), TapeArray{F,T}($(f)(value(A), value(B))))

        function backprop_rule!{T1<:TapeReal,T2<:TapeReal,S<:TapeReal}(node::TapeNode{typeof($f),Tuple{T1,T2},S})
            adj, x, y = adjoint(node.output), value(node.input[1]), value(node.input[2])
            node.input[1].adj += adj * $(grad[1])
            node.input[2].adj += adj * $(grad[2])
            return nothing
        end
    end
end

function backprop_rule!{T1<:AbstractArray,T2<:AbstractArray,S<:AbstractArray}(node::TapeNode{typeof(*),Tuple{T1,T2},S})
    adj, x, y = adjoint(node.output), value(node.input[1]), value(node.input[2])
    increment_adjoint!(node.input[1], adj * y')
    increment_adjoint!(node.input[2], x' * adj)
    return nothing
end
function backprop_rule!{T1<:TapeArray,T2<:TapeArray,S<:TapeArray}(node::TapeNode{typeof(+),Tuple{T1,T2},S})
    adj, x, y = adjoint(node.output), value(node.input[1]), value(node.input[2])
    increment_adjoint!(node.input[1], adj)
    increment_adjoint!(node.input[2], adj)
    return nothing
end

# ForwardDiff fallbacks #
#-----------------------#

# f(x) -> n
function backprop_rule!{F,T<:TapeReal,S<:TapeReal}(node::TapeNode{F,T,S})
    adj, x = adjoint(node.output), value(node.input)
    node.input.adj += adj * ForwardDiff.derivative(node.f, x)
    return nothing
end

# f(x,y) -> n
function backprop_rule!{F,T1<:TapeReal,T2<:TapeReal,S<:TapeReal}(node::TapeNode{F,Tuple{T1,T2},S})
    adj, x, y = adjoint(node.output), value(node.input[1]), value(node.input[2])
    dualx = ForwardDiff.Dual(x, one(x), zero(x))
    dualy = ForwardDiff.Dual(y, zero(y), one(y))
    grad = node.f(dualx, dualy)
    node.input[1].adj += adj * ForwardDiff.partials(grad, 1)
    node.input[2].adj += adj * ForwardDiff.partials(grad, 2)
    return nothing
end

# f(x,y,z) -> n
function backprop_rule!{F,T1<:TapeReal,T2<:TapeReal,T3<:TapeReal,S<:TapeReal}(node::TapeNode{F,Tuple{T1,T2,T3},S})
    adj, x, y, z = adjoint(node.output), value(node.input[1]), value(node.input[2]), value(node.input[3])
    dualx = ForwardDiff.Dual(x, one(x), zero(x), zero(z))
    dualy = ForwardDiff.Dual(y, zero(y), one(y), zero(z))
    dualz = ForwardDiff.Dual(z, zero(z), zero(z), one(z))
    grad = node.f(dualx, dualy, dualz)
    node.input[1].adj += adj * ForwardDiff.partials(grad, 1)
    node.input[2].adj += adj * ForwardDiff.partials(grad, 2)
    node.input[3].adj += adj * ForwardDiff.partials(grad, 3)
    return nothing
end

# f(x...) -> n
function backprop_rule!{F,T<:Tuple,S<:TapeReal}(node::TapeNode{F,T,S})
    adj, inputs = adjoint(node.output), map(value, node.input)
    grad = Vector{numtype(S)}(length(inputs))
    ForwardDiff.gradient!(grad, x -> node.f(x...), inputs)
    for i in eachindex(grad)
        node.input[i].adj += adj * grad[i]
    end
    return nothing
end

#############################
# Special Cases Overloading #
#############################

# no-ops
for f in (:(Base.:<), :(Base.:>), :(Base.:(==)), :(Base.:(<=)), :(Base.:(>=)))
    @eval begin
        @inline $(f){F}(a::TapeReal{F}, b::TapeReal{F}) = TapeReal{F}($(f)(value(a), value(b)))
    end
end

# map(f, arr) -> out
function Base.map{F,T,N}(f, arr::TapeArray{F,T,N})
    out = similar(arr)
    for i in eachindex(out)
        out[i] = TapeReal{F,T}(f(value(arr[i])))
    end
    record!(map, tuple(f, arr), out)
    return out
end

function backprop_rule!{F<:Function,T<:AbstractArray,S<:AbstractArray}(node::TapeNode{typeof(map),Tuple{F,T},S})
    out, f, arr = node.output, node.input[1], node.input[2]
    df = x -> ForwardDiff.derivative(f, x)
    for i in eachindex(arr)
        arr[i].adj += adjoint(out[i]) * df(value(arr[i]))
    end
    return nothing
end

#######
# API #
#######

function gradient{F}(f::F)
    head = reset_head!(TAPE_HEAD_CACHE, f)::TapeHead
    return (out, x, tapearr = tape_array(f, x)) -> begin
        load_tape_array!(tapearr, x)
        f(tapearr)
        backprop!(head)
        head.node = INITIAL_NODE
        load_adjoint_array!(out, tapearr)
        return out
    end
end

end # module
