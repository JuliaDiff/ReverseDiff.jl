########################
# addition/subtraction #
########################

# sum #
#-----#

for A in ARRAY_TYPES
    @eval function Base.sum{V,S}(x::$(A){Tracked{V,S}})
        result = zero(V)
        for t in x
            result += value(t)
        end
        tp = tape(x)
        out = Tracked(result, S, tp)
        record_node!(tp, Special, sum, x, out)
        return out
    end
end

function special_reverse_step!(::typeof(sum), input, output, __)
    increment_adjoint!(input, adjoint(output))
    return nothing
end

function special_forward_step!(::typeof(sum), input, output, __)
    result = zero(valtype(eltype(input)))
    for t in input
        result += value(t)
    end
    setvalue!(output, result)
    return nothing
end

# + #
#---#

for A in ARRAY_TYPES
    @eval function Base.:+{X,Y,S}(x::$(A){Tracked{X,S}}, y::$(A){Tracked{Y,S}})
        tp = tape(x, y)
        out = track(value(x) + value(y), S, tp)
        record_node!(tp, Special, +, (x, y), out)
        return out
    end

    @eval function Base.:+{X,S}(x::$(A){Tracked{X,S}}, y::$(A))
        tp = tape(x)
        out = track(value(x) + y, S, tp)
        record_node!(tp, Special, +, (x, y), out)
        return out
    end

    @eval function Base.:+{Y,S}(x::$(A), y::$(A){Tracked{Y,S}})
        tp = tape(y)
        out = track(x + value(y), S, tp)
        record_node!(tp, Special, +, (x, y), out)
        return out
    end
end

function special_reverse_step!{A,B}(::typeof(+), inputs::Tuple{A,B}, output::AbstractArray, _)
    a, b = inputs
    if eltype(A) <: Tracked && eltype(B) <: Tracked
        extract_and_increment_adjoint!(a, output)
        extract_and_increment_adjoint!(b, output)
    elseif eltype(A) <: Tracked
        extract_and_increment_adjoint!(a, output)
    else
        extract_and_increment_adjoint!(b, output)
    end
    return nothing
end

function special_forward_step!{A,B}(::typeof(+), inputs::Tuple{A,B}, output::AbstractArray, _)
    a, b = inputs
    if eltype(A) <: Tracked && eltype(B) <: Tracked
        for i in eachindex(output)
            setvalue!(output[i], value(a[i]) + value(b[i]))
        end
    elseif eltype(A) <: Tracked
        for i in eachindex(output)
            setvalue!(output[i], value(a[i]) + b[i])
        end
    else
        for i in eachindex(output)
            setvalue!(output[i], a[i] + value(b[i]))
        end
    end
    return nothing
end

# - #
#---#

for A in ARRAY_TYPES
    @eval function Base.:-{X,Y,S}(x::$(A){Tracked{X,S}}, y::$(A){Tracked{Y,S}})
        tp = tape(x, y)
        out = track(value(x) - value(y), S, tp)
        record_node!(tp, Special, -, (x, y), out)
        return out
    end

    @eval function Base.:-{X,S}(x::$(A){Tracked{X,S}}, y::$(A))
        tp = tape(x)
        out = track(value(x) - y, S, tp)
        record_node!(tp, Special, -, (x, y), out)
        return out
    end

    @eval function Base.:-{Y,S}(x::$(A), y::$(A){Tracked{Y,S}})
        tp = tape(y)
        out = track(x - value(y), S, tp)
        record_node!(tp, Special, -, (x, y), out)
        return out
    end

    @eval function Base.:-{V,S}(x::$(A){Tracked{V,S}})
        tp = tape(x)
        out = track(-(value(x)), S, tp)
        record_node!(tp, Special, -, x, out)
        return out
    end
end

function special_reverse_step!(::typeof(-), input, output, _)
    extract_and_decrement_adjoint!(input, output)
    return nothing
end

function special_reverse_step!{A,B}(::typeof(-), inputs::Tuple{A,B}, output::AbstractArray, _)
    a, b = inputs
    if eltype(A) <: Tracked && eltype(B) <: Tracked
        extract_and_increment_adjoint!(a, output)
        extract_and_decrement_adjoint!(b, output)
    elseif eltype(A) <: Tracked
        extract_and_increment_adjoint!(a, output)
    else
        extract_and_decrement_adjoint!(b, output)
    end
    return nothing
end

function special_forward_step!(::typeof(-), input, output, _)
    for i in eachindex(output)
        setvalue!(output[i], -value(input[i]))
    end
    return nothing
end

function special_forward_step!{A,B}(::typeof(-), inputs::Tuple{A,B}, output::AbstractArray, _)
    a, b = inputs
    if eltype(A) <: Tracked && eltype(B) <: Tracked
        for i in eachindex(output)
            setvalue!(output[i], value(a[i]) - value(b[i]))
        end
    elseif eltype(A) <: Tracked
        for i in eachindex(output)
            setvalue!(output[i], value(a[i]) - b[i])
        end
    else
        for i in eachindex(output)
            setvalue!(output[i], a[i] - value(b[i]))
        end
    end
    return nothing
end

##################
# multiplication #
##################

const A_MUL_B_FUNCS = ((:A_mul_B!, :*),
                       (:A_mul_Bt!, :A_mul_Bt), (:At_mul_B!, :At_mul_B), (:At_mul_Bt!, :At_mul_Bt),
                       (:A_mul_Bc!, :A_mul_Bc), (:Ac_mul_B!, :Ac_mul_B), (:Ac_mul_Bc!, :Ac_mul_Bc))

# recording pass #
#----------------#

for A in ARRAY_TYPES, (f!, f) in A_MUL_B_FUNCS
    @eval function Base.$(f){X,Y,S}(x::$(A){Tracked{X,S}}, y::$(A){Tracked{Y,S}})
        tp = tape(x, y)
        x_val, y_val = value(x), value(y)
        out_val = $(f)(x_val, y_val)
        out = track(out_val, S, tp)
        cache = (similar(out, S), out_val, similar(x, S), similar(y, S), x_val, y_val)
        record_node!(tp, Special, $(f), (x, y), out, cache)
        return out
    end

    @eval function Base.$(f){X,S}(x::$(A){Tracked{X,S}}, y::$(A))
        tp = tape(x)
        x_val = value(x)
        out_val = $(f)(x_val, y)
        out = track(out_val, S, tp)
        cache = (similar(out, S), out_val, similar(x, S), x_val, y)
        record_node!(tp, Special, $(f), (x, y), out, cache)
        return out
    end

    @eval function Base.$(f){Y,S}(x::$(A), y::$(A){Tracked{Y,S}})
        tp = tape(y)
        y_val = value(y)
        out_val = $(f)(x, y_val)
        out = track(out_val, S, tp)
        cache = (similar(out, S), out_val, similar(y, S), x, y_val)
        record_node!(tp, Special, $(f), (x, y), out, cache)
        return out
    end

    @eval function Base.$(f!){V,X,Y,S}(out::$(A){Tracked{V,S}},
                                       x::$(A){Tracked{X,S}},
                                       y::$(A){Tracked{Y,S}})
        tp = tape(x, y)
        x_val, y_val = value(x), value(y)
        out_val = $(f)(x_val, y_val)
        track!(out, out_val, tp)
        cache = (similar(out, S), out_val, similar(x, S), similar(y, S), x_val, y_val)
        record_node!(tp, Special, $(f), (x, y), out, cache)
        return out
    end

    @eval function Base.$(f!){V,X,S}(out::$(A){Tracked{V,S}},
                                     x::$(A){Tracked{X,S}},
                                     y::$(A))
        tp = tape(x)
        x_val = value(x)
        out_val = $(f)(x_val, y)
        track!(out, out_val, tp)
        cache = (similar(out, S), out_val, similar(x, S), x_val, y)
        record_node!(tp, Special, $(f), (x, y), out, cache)
        return out
    end

    @eval function Base.$(f!){V,Y,S}(out::$(A){Tracked{V,S}},
                                     x::$(A),
                                     y::$(A){Tracked{Y,S}})
        tp = tape(y)
        y_val = value(y)
        out_val = $(f)(x, y_val)
        track!(out, out_val, tp)
        cache = (similar(out, S), out_val, similar(y, S), x, y_val)
        record_node!(tp, Special, $(f), (x, y), out, cache)
        return out
    end
end

# forward pass #
#--------------#

for (f!, f) in A_MUL_B_FUNCS
    @eval function special_forward_step!{A,B}(::typeof($f), inputs::Tuple{A,B}, output, cache)
        a, b = inputs
        if eltype(A) <: Tracked && eltype(B) <: Tracked
            output_adjoint, output_value, _, _, a_val, b_val = cache
            value!(a_val, a)
            value!(b_val, b)
            $(f!)(output_value, a_val, b_val)
            setvalue!(output, output_value)
        elseif eltype(A) <: Tracked
            output_adjoint, output_value, _, a_val, _ = cache
            value!(a_val, a)
            $(f!)(output_value, a_val, b)
            setvalue!(output, output_value)
        else
            output_adjoint, output_value, _, _, b_val = cache
            value!(b_val, b)
            $(f!)(output_value, a, b_val)
            setvalue!(output, output_value)
        end
        return nothing
    end
end

# reverse pass #
#--------------#

### *

function special_reverse_step!{A,B}(::typeof(*), inputs::Tuple{A,B}, output, cache)
    a, b = inputs
    if eltype(A) <: Tracked && eltype(B) <: Tracked
        output_adjoint, _, a_deriv, b_deriv, a_val, b_val = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(a, A_mul_Bc!(a_deriv, output_adjoint, b_val))
        increment_adjoint!(b, Ac_mul_B!(b_deriv, a_val, output_adjoint))
    elseif eltype(A) <: Tracked
        output_adjoint, _, a_deriv, _, b_val = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(a, A_mul_Bc!(a_deriv, output_adjoint, b_val))
    else
        output_adjoint, _, b_deriv, a_val, _ = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(b, Ac_mul_B!(b_deriv, a_val, output_adjoint))
    end
    return nothing
end

### A_mul_Bt

function special_reverse_step!{A,B}(::typeof(A_mul_Bt), inputs::Tuple{A,B}, output, cache)
    a, b = inputs
    if eltype(A) <: Tracked && eltype(B) <: Tracked
        output_adjoint, _, a_deriv, b_deriv, a_val, b_val = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(a, A_mul_B!(a_deriv, output_adjoint, b_val))
        increment_adjoint!(b, At_mul_B!(b_deriv, output_adjoint, a_val))
    elseif eltype(A) <: Tracked
        output_adjoint, _, a_deriv, _, b_val = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(a, A_mul_B!(a_deriv, output_adjoint, b_val))
    else
        output_adjoint, _, b_deriv, a_val, _ = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(b, At_mul_B!(b_deriv, output_adjoint, a_val))
    end
    return nothing
end

### At_mul_B

function special_reverse_step!{A,B}(::typeof(At_mul_B), inputs::Tuple{A,B}, output, cache)
    a, b = inputs
    if eltype(A) <: Tracked && eltype(B) <: Tracked
        output_adjoint, _, a_deriv, b_deriv, a_val, b_val = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(a, A_mul_Bt!(a_deriv, b_val, output_adjoint))
        increment_adjoint!(b, A_mul_B!(b_deriv, a_val, output_adjoint))
    elseif eltype(A) <: Tracked
        output_adjoint, _, a_deriv, _, b_val = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(a, A_mul_Bt!(a_deriv, b_val, output_adjoint))
    else
        output_adjoint, _, b_deriv, a_val, _ = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(b, A_mul_B!(b_deriv, a_val, output_adjoint))
    end
    return nothing
end

### At_mul_Bt

function special_reverse_step!{A,B}(::typeof(At_mul_Bt), inputs::Tuple{A,B}, output, cache)
    a, b = inputs
    if eltype(A) <: Tracked && eltype(B) <: Tracked
        output_adjoint, _, a_deriv, b_deriv, a_val, b_val = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(a, At_mul_Bt!(a_deriv, b_val, output_adjoint))
        increment_adjoint!(b, At_mul_Bt!(b_deriv, output_adjoint, a_val))
    elseif eltype(A) <: Tracked
        output_adjoint, _, a_deriv, _, b_val = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(a, At_mul_Bt!(a_deriv, b_val, output_adjoint))
    else
        output_adjoint, _, b_deriv, a_val, _ = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(b, At_mul_Bt!(b_deriv, output_adjoint, a_val))
    end
    return nothing
end

### A_mul_Bc

function special_reverse_step!{A,B}(::typeof(A_mul_Bc), inputs::Tuple{A,B}, output, cache)
    a, b = inputs
    if eltype(A) <: Tracked && eltype(B) <: Tracked
        output_adjoint, _, a_deriv, b_deriv, a_val, b_val = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(a, A_mul_B!(a_deriv, output_adjoint, b_val))
        increment_adjoint!(b, Ac_mul_B!(b_deriv, output_adjoint, a_val))
    elseif eltype(A) <: Tracked
        output_adjoint, _, a_deriv, _, b_val = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(a, A_mul_B!(a_deriv, output_adjoint, b_val))
    else
        output_adjoint, _, b_deriv, a_val, _ = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(b, Ac_mul_B!(b_deriv, output_adjoint, a_val))
    end
    return nothing
end

### Ac_mul_B

function special_reverse_step!{A,B}(::typeof(Ac_mul_B), inputs::Tuple{A,B}, output, cache)
    a, b = inputs
    if eltype(A) <: Tracked && eltype(B) <: Tracked
        output_adjoint, _, a_deriv, b_deriv, a_val, b_val = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(a, A_mul_Bc!(a_deriv, b_val, output_adjoint))
        increment_adjoint!(b, A_mul_B!(b_deriv, a_val, output_adjoint))
    elseif eltype(A) <: Tracked
        output_adjoint, _, a_deriv, _, b_val = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(a, A_mul_Bc!(a_deriv, b_val, output_adjoint))
    else
        output_adjoint, _, b_deriv, a_val, _ = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(b, A_mul_B!(b_deriv, a_val, output_adjoint))
    end
    return nothing
end

### Ac_mul_Bc

function special_reverse_step!{A,B}(::typeof(Ac_mul_Bc), inputs::Tuple{A,B}, output, cache)
    a, b = inputs
    if eltype(A) <: Tracked && eltype(B) <: Tracked
        output_adjoint, _, a_deriv, b_deriv, a_val, b_val = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(a, Ac_mul_Bc!(a_deriv, b_val, output_adjoint))
        increment_adjoint!(b, Ac_mul_Bc!(b_deriv, output_adjoint, a_val))
    elseif eltype(A) <: Tracked
        output_adjoint, _, a_deriv, _, b_val = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(a, Ac_mul_Bc!(a_deriv, b_val, output_adjoint))
    else
        output_adjoint, _, b_deriv, a_val, _ = cache
        adjoint!(output_adjoint, output)
        increment_adjoint!(b, Ac_mul_Bc!(b_deriv, output_adjoint, a_val))
    end
    return nothing
end

#######################
# misc linear algebra #
#######################

# det #
#-----#

for A in ARRAY_TYPES
    @eval function Base.det{V,S}(x::$(A){Tracked{V,S}})
        tp = tape(x)
        x_val = value(x)
        det_x_val = det(x_val)
        out = track(det_x_val, S, tp)
        if det_x_val == 0
            record_node!(tp, Special, det, x, out, nothing)
        else
            record_node!(tp, Special, det, x, out, (inv(x_val), similar(x_val)))
        end
        return out
    end
end

function special_reverse_step!(::typeof(det), input, output, cache)
    if output != 0
        inv_input_value, inv_input_value_transpose = cache
        k = adjoint(output) * value(output)
        # this transpose must occur in the backwards pass for
        # nested differentiation to work properly
        transpose!(inv_input_value_transpose, inv_input_value)
        increment_adjoint!(input, scale!(k, inv_input_value_transpose))
    end
    return nothing
end

function special_forward_step!(::typeof(det), input, output, cache)
    inv_input_value, _ = cache
    value!(inv_input_value, input)
    setvalue!(output, det(inv_input_value))
    output != 0 && copy!(inv_input_value, inv(inv_input_value))
    return nothing
end

# inv #
#-----#

for A in ARRAY_TYPES
    @eval function Base.inv{V,S}(x::$(A){Tracked{V,S}})
        tp = tape(x)
        out_val = inv(value(x))
        out = track(out_val, S, tp)
        cache = (similar(out, S, (size(out, 2), size(out, 2))),
                 similar(out, S, (size(out, 2), size(out, 1))),
                 similar(out, S), out_val)
        record_node!(tp, Special, inv, x, out, cache)
        return out
    end
end

function special_reverse_step!(::typeof(inv), input, output, cache)
    deriv_part1, deriv_part2, output_adjoint, output_value = cache
    adjoint!(output_adjoint, output)
    Ac_mul_B!(deriv_part1, output_value, output_adjoint)
    map!(-, deriv_part1, deriv_part1)
    A_mul_Bc!(deriv_part2, deriv_part1, output_value)
    increment_adjoint!(input, deriv_part2)
    return nothing
end

function special_forward_step!(::typeof(inv), input, output, cache)
    _, _, _, output_value = cache
    value!(output_value, input)
    setvalue!(output, inv(output_value))
    value!(output_value, output)
end
