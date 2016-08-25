#################################################
# backpropagation over the trace (reverse pass) #
#################################################

seed!(tr::Trace) = (seed!(tr.list.head.outputs); return tr)
seed!(t::TraceReal) = (t.adjoint[] = one(adjtype(t)); return t)

unseed!(tr::Trace) = (unseed!(tr.list.head.outputs); return tr)
unseed!(t::TraceReal) = (t.adjoint[] = zero(adjtype(t)); return t)
unseed!(items) = for t in items; unseed!(t); end

function backprop!(trace::Trace)
    for node in trace
        backprop_step!(node)
    end
    return nothing
end

backprop_step!(node::TraceNode{Void}) = partials_backprop_step!(node.inputs, node.outputs, node.partials)
backprop_step!(node::TraceNode) = func_backprop_step!(node.func, node.inputs, node.outputs)

####################
# scalar functions #
####################

# f(::Number)::Number
function partials_backprop_step!(input::TraceReal, output::TraceReal, partials::Partials)
    input.adjoint[] += adjoint(output) * first(partials)
    unseed!(output)
    return nothing
end

# f(::Number...)::Number
function partials_backprop_step!{N}(inputs::Tuple, output::TraceReal, partials::Partials{N})
    for i in 1:N
        inputs[i].adjoint[] += adjoint(output) * partials[i]
    end
    unseed!(output)
    return nothing
end

###############################################
# elementwise functions (e.g. broadcast, map) #
###############################################

# f.(::AbstractArray)::AbstractArray
function partials_backprop_step!(input::AbstractArray, output::AbstractArray, partials::AbstractArray)
    for i in eachindex(output)
        partials_backprop_step!(input[i], output[i], partials[i])
    end
    return nothing
end

# f.(::AbstractArray, ::AbstractArray)::AbstractArray
function partials_backprop_step!(inputs::NTuple{2}, output::AbstractArray, partials::AbstractArray)
    a, b = inputs
    for i in eachindex(output)
        partials_backprop_step!((a[i], b[i]), output[i], partials[i])
    end
    return nothing
end

# f.(::AbstractArray, ::AbstractArray, ::AbstractArray)::AbstractArray
function partials_backprop_step!(inputs::NTuple{3}, output::AbstractArray, partials::AbstractArray)
    a, b, c = inputs
    for i in eachindex(output)
        partials_backprop_step!((a[i], b[i], c[i]), output[i], partials[i])
    end
    return nothing
end

#########################################################################
# functions whose derivatives didn't get calculated in the forward pass #
#########################################################################

# unary functions #
#-----------------#

function func_backprop_step!(::typeof(-), input::AbstractArray, output::AbstractArray)
    decrement_adjoint!(input, output)
    unseed!(output)
    return nothing
end

function func_backprop_step!(::typeof(inv), input::AbstractArray, output::AbstractArray)
    output_value = value(output)
    increment_adjoint!(input, negate!(output_value' * adjoint(output)) * output_value')
    unseed!(output)
    return nothing
end

function func_backprop_step!(::typeof(det), input::AbstractArray, output::TraceReal)
    increment_adjoint!(input, (adjoint(output) * value(output)) * inv(value(input))')
    unseed!(output)
    return nothing
end

function func_backprop_step!(::typeof(sum), input::AbstractArray, output::TraceReal)
    increment_adjoint!(input)
    unseed!(output)
    return nothing
end

# binary functions #
#------------------#

function func_backprop_step!{A,B}(::typeof(+), inputs::Tuple{A,B}, output::AbstractArray)
    increment_adjoint!(inputs[1], output)
    increment_adjoint!(inputs[2], output)
    unseed!(output)
    return nothing
end

function func_backprop_step!{A,B}(::typeof(-), inputs::Tuple{A,B}, output::AbstractArray)
    increment_adjoint!(inputs[1], output)
    decrement_adjoint!(inputs[2], output)
    unseed!(output)
    return nothing
end

# A_mul_B family #
#----------------#

function func_backprop_step!{A,B}(::typeof(*), inputs::Tuple{A,B}, output::AbstractArray)
    a, b = inputs
    output_adjoint = adjoint(output)
    increment_adjoint!(a, output_adjoint * value(b)')
    increment_adjoint!(b, value(a)' * output_adjoint)
    unseed!(output)
    return nothing
end

function func_backprop_step!{A,B}(::typeof(A_mul_Bt), inputs::Tuple{A,B}, output::AbstractArray)
    a, b = inputs
    output_adjoint = adjoint(output)
    increment_adjoint!(a, output_adjoint   * value(b))
    increment_adjoint!(b, output_adjoint.' * value(a))
    unseed!(output)
    return nothing
end

function func_backprop_step!{A,B}(::typeof(At_mul_B), inputs::Tuple{A,B}, output::AbstractArray)
    a, b = inputs
    output_adjoint = adjoint(output)
    increment_adjoint!(a, value(b) * output_adjoint.')
    increment_adjoint!(b, value(a) * output_adjoint)
    unseed!(output)
    return nothing
end

function func_backprop_step!{A,B}(::typeof(At_mul_Bt), inputs::Tuple{A,B}, output::AbstractArray)
    a, b = inputs
    output_adjoint = adjoint(output)
    increment_adjoint!(a, (output_adjoint * value(b)).')
    increment_adjoint!(b, (value(a) * output_adjoint).')
    unseed!(output)
    return nothing
end

function func_backprop_step!{A,B}(::typeof(A_mul_Bc), inputs::Tuple{A,B}, output::AbstractArray)
    a, b = inputs
    output_adjoint = adjoint(output)
    increment_adjoint!(a, output_adjoint  * value(b))
    increment_adjoint!(b, output_adjoint' * value(a))
    unseed!(output)
    return nothing
end

function func_backprop_step!{A,B}(::typeof(Ac_mul_B), inputs::Tuple{A,B}, output::AbstractArray)
    a, b = inputs
    output_adjoint = adjoint(output)
    increment_adjoint!(a, value(b) * output_adjoint')
    increment_adjoint!(b, value(a) * output_adjoint)
    unseed!(output)
    return nothing
end

function func_backprop_step!{A,B}(::typeof(Ac_mul_Bc), inputs::Tuple{A,B}, output::AbstractArray)
    a, b = inputs
    output_adjoint = adjoint(output)
    increment_adjoint!(a, (output_adjoint * value(b))')
    increment_adjoint!(b, (value(a) * output_adjoint)')
    unseed!(output)
    return nothing
end

# utilities #
#-----------#

negate!(A) = scale!(-one(eltype(A)), A)

function decrement_adjoint!{T<:TraceReal}(input, output::AbstractArray{T})
    for i in eachindex(input)
        input[i].adjoint[] -= adjoint(output[i])
    end
    return input
end

function increment_adjoint!{T<:TraceReal}(input, output::AbstractArray{T})
    for i in eachindex(input)
        input[i].adjoint[] += adjoint(output[i])
    end
    return input
end

function increment_adjoint!{T<:TraceReal}(input::AbstractArray{T})
    x = one(adjtype(T))
    for i in eachindex(input)
        input[i].adjoint[] += x
    end
    return input
end

function increment_adjoint!(input, derivs)
    for i in eachindex(input)
        input[i].adjoint[] += derivs[i]
    end
    return input
end
