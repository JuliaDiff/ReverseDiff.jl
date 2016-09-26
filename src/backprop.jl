#################################################
# backpropagation over the tape (reverse pass) #
#################################################

seed!(t::Tracer) = (t.adjoint = one(adjtype(t)); return t)
seed!(t::TapeNode) = (seed!(t.outputs); return t)

unseed!(t::Tracer) = (t.adjoint = zero(adjtype(t)); return t)
unseed!(t::TapeNode) = (unseed!(t.inputs); unseed!(t.outputs); return t)
unseed!(ts) = for t in ts; unseed!(t); end

function backprop!(tape::Tape)
    for i in length(tape):-1:1
        backprop_step!(tape[i])
    end
    return nothing
end

backprop_step!(node::TapeNode{Void}) = scalar_backprop_step!(node.inputs, node.outputs, node.cache)
backprop_step!(node::TapeNode) = special_backprop_step!(node.func, node.inputs, node.outputs, node.cache)

####################
# scalar functions #
####################

# f(::Number)::Number
function scalar_backprop_step!(input::Tracer, output::Tracer, deriv::Partials{1})
    input.adjoint += adjoint(output) * deriv[1]
    return nothing
end

# f(::Number...)::Number
function scalar_backprop_step!{N}(inputs::Tuple, output::Tracer, grad::Partials{N})
    for i in 1:N
        inputs[i].adjoint += adjoint(output) * grad[i]
    end
    return nothing
end

#####################
# special functions #
#####################

# map #
#-----#

function special_backprop_step!(::typeof(map), input, output, duals)
    for i in eachindex(output)
        scalar_backprop_step!(input[i], output[i], partials(duals[i]))
    end
    return nothing
end

function special_backprop_step!{A,B}(::typeof(map), inputs::Tuple{A,B}, output, duals)
    a, b = inputs
    for i in eachindex(output)
        scalar_backprop_step!((a[i], b[i]), output[i], partials(duals[i]))
    end
    return nothing
end

# broadcast #
#-----------#

function special_backprop_step!(::typeof(broadcast), input::AbstractArray, output, duals)
    return special_backprop_step!(map, input, output, duals)
end

function special_backprop_step!{A,B}(::typeof(broadcast), inputs::Tuple{A,B}, output, duals)
    a, b = inputs
    if size(a) == size(b)
        special_backprop_step!(map, inputs, output, duals)
    else
        propagate_adjoint!(duals, output)
        increment_adjoint!(a, sumover(1, a, duals))
        increment_adjoint!(b, sumover(2, b, duals))
    end
    return nothing
end

function propagate_adjoint!(duals, output)
    for i in eachindex(duals)
        duals[i] *= adjoint(output[i])
    end
    return nothing
end

function sumover(p, x::AbstractArray, duals)
    dims = (i for i in 1:ndims(duals) if size(x,i) != size(duals,i))
    return sum(d -> partials(d, p), duals, dims)
end

sumover(p, x::Real, duals) = sum(d -> partials(d, p), duals)

# addition/subtraction #
#----------------------#

function special_backprop_step!(::typeof(sum), input, _, __)
    increment_adjoint!(input)
    return nothing
end

function special_backprop_step!{A,B}(::typeof(+), inputs::Tuple{A,B}, output::AbstractArray, _)
    increment_adjoint!(inputs[1], output)
    increment_adjoint!(inputs[2], output)
    return nothing
end

function special_backprop_step!(::typeof(-), input, output, _)
    decrement_adjoint!(input, output)
    return nothing
end

function special_backprop_step!{A,B}(::typeof(-), inputs::Tuple{A,B}, output::AbstractArray, _)
    increment_adjoint!(inputs[1], output)
    decrement_adjoint!(inputs[2], output)
    return nothing
end

# A_mul_B family #
#----------------#

function special_backprop_step!{A,B}(::typeof(*), inputs::Tuple{A,B}, output, vals)
    a, b = inputs
    aval, bval = vals
    output_adjoint = adjoint(output)
    increment_adjoint!(a, output_adjoint * value(b)')
    increment_adjoint!(b, value(a)' * output_adjoint)
    return nothing
end

function special_backprop_step!{A,B}(::typeof(A_mul_Bt), inputs::Tuple{A,B}, output, vals)
    a, b = inputs
    aval, bval = vals
    output_adjoint = adjoint(output)
    increment_adjoint!(a, output_adjoint   * bval)
    increment_adjoint!(b, output_adjoint.' * aval)
    return nothing
end

function special_backprop_step!{A,B}(::typeof(At_mul_B), inputs::Tuple{A,B}, output, vals)
    a, b = inputs
    aval, bval = vals
    output_adjoint = adjoint(output)
    increment_adjoint!(a, bval * output_adjoint.')
    increment_adjoint!(b, aval * output_adjoint)
    return nothing
end

function special_backprop_step!{A,B}(::typeof(At_mul_Bt), inputs::Tuple{A,B}, output, vals)
    a, b = inputs
    aval, bval = vals
    output_adjoint = adjoint(output)
    increment_adjoint!(a, (output_adjoint * bval).')
    increment_adjoint!(b, (aval * output_adjoint).')
    return nothing
end

function special_backprop_step!{A,B}(::typeof(A_mul_Bc), inputs::Tuple{A,B}, output, vals)
    a, b = inputs
    aval, bval = vals
    output_adjoint = adjoint(output)
    increment_adjoint!(a, output_adjoint  * bval)
    increment_adjoint!(b, output_adjoint' * aval)
    return nothing
end

function special_backprop_step!{A,B}(::typeof(Ac_mul_B), inputs::Tuple{A,B}, output, vals)
    a, b = inputs
    aval, bval = vals
    output_adjoint = adjoint(output)
    increment_adjoint!(a, bval * output_adjoint')
    increment_adjoint!(b, aval * output_adjoint)
    return nothing
end

function special_backprop_step!{A,B}(::typeof(Ac_mul_Bc), inputs::Tuple{A,B}, output, vals)
    a, b = inputs
    aval, bval = vals
    output_adjoint = adjoint(output)
    increment_adjoint!(a, (output_adjoint * bval)')
    increment_adjoint!(b, (aval * output_adjoint)')
    return nothing
end

# linear algebra #
#----------------#

function special_backprop_step!(::typeof(inv), input, output, output_value)
    increment_adjoint!(input, negate!(output_value' * adjoint(output)) * output_value')
    return nothing
end

function special_backprop_step!(::typeof(det), input, output, inv_input_value)
    increment_adjoint!(input, (adjoint(output) * value(output)) * inv_input_value')
    return nothing
end

# utilities #
#-----------#

negate!(A) = scale!(-one(eltype(A)), A)

function decrement_adjoint!{T<:Tracer}(input::AbstractArray, output::AbstractArray{T})
    for i in eachindex(input)
        input[i].adjoint -= adjoint(output[i])
    end
    return input
end

function increment_adjoint!{T<:Tracer}(input::AbstractArray, output::AbstractArray{T})
    for i in eachindex(input)
        input[i].adjoint += adjoint(output[i])
    end
    return input
end

function increment_adjoint!{T<:Tracer}(input::AbstractArray{T})
    x = one(adjtype(T))
    for i in eachindex(input)
        input[i].adjoint += x
    end
    return input
end

function increment_adjoint!(input::AbstractArray, derivs::AbstractArray)
    for i in eachindex(input)
        input[i].adjoint += derivs[i]
    end
    return input
end

increment_adjoint!(input::Tracer, deriv::Real) = (input.adjoint += deriv)
