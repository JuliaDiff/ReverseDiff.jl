###########
# forward #
###########

const A_MUL_B_FUNCS = (:*, :A_mul_Bt, :At_mul_B, :At_mul_Bt, :A_mul_Bc, :Ac_mul_B, :Ac_mul_Bc)

const A_MUL_B!_FUNCS = ((:A_mul_B!, :*),
                        (:A_mul_Bt!, :A_mul_Bt), (:At_mul_B!, :At_mul_B), (:At_mul_Bt!, :At_mul_Bt),
                        (:A_mul_Bc!, :A_mul_Bc), (:Ac_mul_B!, :Ac_mul_B), (:Ac_mul_Bc!, :Ac_mul_Bc))

for A in ARRAY_TYPES

    # addition/subtraction #
    #----------------------#

    @eval function Base.sum{V,S}(x::$(A){Tracked{V,S}})
        result = zero(V)
        for t in x
            result += value(t)
        end
        tp = tape(x)
        out = Tracked(result, S, tp)
        record!(tp, sum, x, out)
        return out
    end

    for f in (:-, :+)
        @eval function Base.$(f){X,Y,S}(x::$(A){Tracked{X,S}}, y::$(A){Tracked{Y,S}})
            tp = tape(x, y)
            out = track($(f)(value(x), value(y)), S, tp)
            record!(tp, $(f), (x, y), out)
            return out
        end

        if f != :-
            @eval function Base.$(f){X,S}(x::$(A){Tracked{X,S}}, y::$(A))
                tp = tape(x)
                out = track($(f)(value(x), y), S, tp)
                record!(tp, $(f), x, out)
                return out
            end
        end

        @eval function Base.$(f){Y,S}(x::$(A), y::$(A){Tracked{Y,S}})
            tp = tape(y)
            out = track($(f)(x, value(y)), S, tp)
            record!(tp, $(f), y, out)
            return out
        end
    end

    @eval function Base.:-{X,S}(x::$(A){Tracked{X,S}}, y::$(A))
        tp = tape(x)
        out = track(value(x) - y, S, tp)
        record!(tp, +, x, out)
        return out
    end

    @eval function Base.:-{V,S}(x::$(A){Tracked{V,S}})
        tp = tape(x)
        out = track(-(value(x)), S, tp)
        record!(tp, -, x, out)
        return out
    end

    # A_mul_B family #
    #----------------#

    for f in A_MUL_B_FUNCS
        @eval function Base.$(f){X,Y,S}(x::$(A){Tracked{X,S}}, y::$(A){Tracked{Y,S}})
            tp = tape(x, y)
            x_val, y_val = value(x), value(y)
            out_val = $(f)(x_val, y_val)
            out = track(out_val, S, tp)
            cache = (similar(out_val, S), similar(x_val, S), similar(y_val, S), x_val, y_val)
            record!(tp, $(f), (x, y), out, cache)
            return out
        end

        @eval function Base.$(f){X,S}(x::$(A){Tracked{X,S}}, y::$(A))
            tp = tape(x)
            x_val = value(x)
            out_val = $(f)(x_val, y)
            out = track(out_val, S, tp)
            cache = (similar(out_val, S), similar(x_val, S), y)
            record!(tp, $(f), (x, nothing), out, cache)
            return out
        end

        @eval function Base.$(f){Y,S}(x::$(A), y::$(A){Tracked{Y,S}})
            tp = tape(y)
            y_val = value(y)
            out_val = $(f)(x, y_val)
            out = track(out_val, S, tp)
            cache = (similar(out_val, S), similar(y_val, S), x)
            record!(tp, $(f), (nothing, y), out, cache)
            return out
        end
    end

    # in-place versions
    for (f!, f) in A_MUL_B!_FUNCS
        @eval function Base.$(f!){V,X,Y,S}(out::$(A){Tracked{V,S}},
                                           x::$(A){Tracked{X,S}},
                                           y::$(A){Tracked{Y,S}})
            tp = tape(x, y)
            x_val, y_val = value(x), value(y)
            out_val = $(f)(x_val, y_val)
            track!(out, out_val, tp)
            cache = (similar(out_val, S), similar(x_val, S), similar(y_val, S), x_val, y_val)
            record!(tp, $(f), (x, y), out, cache)
            return out
        end

        @eval function Base.$(f!){V,X,S}(out::$(A){Tracked{V,S}},
                                         x::$(A){Tracked{X,S}},
                                         y::$(A))
            tp = tape(x)
            x_val = value(x)
            out_val = $(f)(x_val, y)
            track!(out, out_val, tp)
            cache = (similar(out_val, S), similar(x_val, S), y)
            record!(tp, $(f), (x, nothing), out, cache)
            return out
        end

        @eval function Base.$(f!){V,Y,S}(out::$(A){Tracked{V,S}},
                                         x::$(A),
                                         y::$(A){Tracked{Y,S}})
            tp = tape(y)
            y_val = value(y)
            out_val = $(f)(x, y_val)
            track!(out, out_val, tp)
            cache = (similar(out_val, S), similar(y_val, S), x)
            record!(tp, $(f), (nothing, y), out, cache)
            return out
        end
    end

    # linear algebra #
    #----------------#

    @eval function Base.inv{V,S}(x::$(A){Tracked{V,S}})
        tp = tape(x)
        out_val = inv(value(x))
        out = track(out_val, S, tp)
        record!(tp, inv, x, out, out_val)
        return out
    end

    @eval function Base.det{V,S}(x::$(A){Tracked{V,S}})
        tp = tape(x)
        x_val = value(x)
        det_x_val = det(x_val)
        out = track(det_x_val, S, tp)
        if det_x_val == 0
            record!(tp, det, x, out, nothing)
        else
            record!(tp, det, x, out, inv(x_val))
        end
        return out
    end
end

###########
# reverse #
###########

# addition/subtraction #
#----------------------#

function special_reverse_step!(::typeof(sum), input, output, __)
    increment_adjoint!(input, adjoint(output))
    return nothing
end

function special_reverse_step!{A,B}(::typeof(+), inputs::Tuple{A,B}, output::AbstractArray, _)
    extract_and_increment_adjoint!(inputs[1], output)
    extract_and_increment_adjoint!(inputs[2], output)
    return nothing
end

function special_reverse_step!(::typeof(+), input, output, _)
    extract_and_increment_adjoint!(input, output)
    return nothing
end

function special_reverse_step!{A,B}(::typeof(-), inputs::Tuple{A,B}, output::AbstractArray, _)
    extract_and_increment_adjoint!(inputs[1], output)
    extract_and_decrement_adjoint!(inputs[2], output)
    return nothing
end

function special_reverse_step!(::typeof(-), input, output, _)
    extract_and_decrement_adjoint!(input, output)
    return nothing
end

# A_mul_B family #
#----------------#

# *

function special_reverse_step!{A,B}(::typeof(*), inputs::Tuple{A,B}, output, cache)
    a, b = inputs
    output_adjoint, a_deriv, b_deriv, a_val, b_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(a, A_mul_Bc!(a_deriv, output_adjoint, b_val))
    increment_adjoint!(b, Ac_mul_B!(b_deriv, a_val, output_adjoint))
    return nothing
end

function special_reverse_step!{T}(::typeof(*), inputs::Tuple{T,Void}, output, cache)
    a, _ = inputs
    output_adjoint, a_deriv, b_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(a, A_mul_Bc!(a_deriv, output_adjoint, b_val))
    return nothing
end

function special_reverse_step!{T}(::typeof(*), inputs::Tuple{Void,T}, output, cache)
    _, b = inputs
    output_adjoint, b_deriv, a_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(b, Ac_mul_B!(b_deriv, a_val, output_adjoint))
    return nothing
end

# A_mul_Bt

function special_reverse_step!{A,B}(::typeof(A_mul_Bt), inputs::Tuple{A,B}, output, cache)
    a, b = inputs
    output_adjoint, a_deriv, b_deriv, a_val, b_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(a, A_mul_B!(a_deriv, output_adjoint, b_val))
    increment_adjoint!(b, At_mul_B!(b_deriv, output_adjoint, a_val))
    return nothing
end

function special_reverse_step!{T}(::typeof(A_mul_Bt), inputs::Tuple{T,Void}, output, cache)
    a, _ = inputs
    output_adjoint, a_deriv, b_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(a, A_mul_B!(a_deriv, output_adjoint, b_val))
    return nothing
end

function special_reverse_step!{T}(::typeof(A_mul_Bt), inputs::Tuple{Void,T}, output, cache)
    _, b = inputs
    output_adjoint, b_deriv, a_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(b, At_mul_B!(b_deriv, output_adjoint, a_val))
    return nothing
end

# At_mul_B

function special_reverse_step!{A,B}(::typeof(At_mul_B), inputs::Tuple{A,B}, output, cache)
    a, b = inputs
    output_adjoint, a_deriv, b_deriv, a_val, b_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(a, A_mul_Bt!(a_deriv, b_val, output_adjoint))
    increment_adjoint!(b, A_mul_B!(b_deriv, a_val, output_adjoint))
    return nothing
end

function special_reverse_step!{T}(::typeof(At_mul_B), inputs::Tuple{T,Void}, output, cache)
    a, _ = inputs
    output_adjoint, a_deriv, b_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(a, A_mul_Bt!(a_deriv, b_val, output_adjoint))
    return nothing
end

function special_reverse_step!{T}(::typeof(At_mul_B), inputs::Tuple{Void,T}, output, cache)
    _, b = inputs
    output_adjoint, b_deriv, a_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(b, A_mul_B!(b_deriv, a_val, output_adjoint))
    return nothing
end

# At_mul_Bt

function special_reverse_step!{A,B}(::typeof(At_mul_Bt), inputs::Tuple{A,B}, output, cache)
    a, b = inputs
    output_adjoint, a_deriv, b_deriv, a_val, b_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(a, At_mul_Bt!(a_deriv, b_val, output_adjoint))
    increment_adjoint!(b, At_mul_Bt!(b_deriv, output_adjoint, a_val))
    return nothing
end

function special_reverse_step!{T}(::typeof(At_mul_Bt), inputs::Tuple{T,Void}, output, cache)
    a, _ = inputs
    output_adjoint, a_deriv, b_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(a, At_mul_Bt!(a_deriv, b_val, output_adjoint))
    return nothing
end

function special_reverse_step!{T}(::typeof(At_mul_Bt), inputs::Tuple{Void,T}, output, cache)
    _, b = inputs
    output_adjoint, b_deriv, a_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(b, At_mul_Bt!(b_deriv, output_adjoint, a_val))
    return nothing
end

# A_mul_Bc

function special_reverse_step!{A,B}(::typeof(A_mul_Bc), inputs::Tuple{A,B}, output, cache)
    a, b = inputs
    output_adjoint, a_deriv, b_deriv, a_val, b_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(a, A_mul_B!(a_deriv, output_adjoint, b_val))
    increment_adjoint!(b, Ac_mul_B!(b_deriv, output_adjoint, a_val))
    return nothing
end

function special_reverse_step!{T}(::typeof(A_mul_Bc), inputs::Tuple{T,Void}, output, cache)
    a, _ = inputs
    output_adjoint, a_deriv, b_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(a, A_mul_B!(a_deriv, output_adjoint, b_val))
    return nothing
end

function special_reverse_step!{T}(::typeof(A_mul_Bc), inputs::Tuple{Void,T}, output, cache)
    _, b = inputs
    output_adjoint, b_deriv, a_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(b, Ac_mul_B!(b_deriv, output_adjoint, a_val))
    return nothing
end

# Ac_mul_B

function special_reverse_step!{A,B}(::typeof(Ac_mul_B), inputs::Tuple{A,B}, output, cache)
    a, b = inputs
    output_adjoint, a_deriv, b_deriv, a_val, b_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(a, A_mul_Bc!(a_deriv, b_val, output_adjoint))
    increment_adjoint!(b, A_mul_B!(b_deriv, a_val, output_adjoint))
    return nothing
end

function special_reverse_step!{T}(::typeof(Ac_mul_B), inputs::Tuple{T,Void}, output, cache)
    a, _ = inputs
    output_adjoint, a_deriv, b_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(a, A_mul_Bc!(a_deriv, b_val, output_adjoint))
    return nothing
end

function special_reverse_step!{T}(::typeof(Ac_mul_B), inputs::Tuple{Void,T}, output, cache)
    _, b = inputs
    output_adjoint, b_deriv, a_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(b, A_mul_B!(b_deriv, a_val, output_adjoint))
    return nothing
end

# Ac_mul_Bc

function special_reverse_step!{A,B}(::typeof(Ac_mul_Bc), inputs::Tuple{A,B}, output, cache)
    a, b = inputs
    output_adjoint, a_deriv, b_deriv, a_val, b_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(a, Ac_mul_Bc!(a_deriv, b_val, output_adjoint))
    increment_adjoint!(b, Ac_mul_Bc!(b_deriv, output_adjoint, a_val))
    return nothing
end

function special_reverse_step!{T}(::typeof(Ac_mul_Bc), inputs::Tuple{T,Void}, output, cache)
    a, _ = inputs
    output_adjoint, a_deriv, b_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(a, Ac_mul_Bc!(a_deriv, b_val, output_adjoint))
    return nothing
end

function special_reverse_step!{T}(::typeof(Ac_mul_Bc), inputs::Tuple{Void,T}, output, cache)
    _, b = inputs
    output_adjoint, b_deriv, a_val = cache
    adjoint!(output_adjoint, output)
    increment_adjoint!(b, Ac_mul_Bc!(b_deriv, output_adjoint, a_val))
    return nothing
end

# special functions #
#-------------------#

function special_reverse_step!(::typeof(inv), input, output, output_value)
    increment_adjoint!(input, -(output_value' * adjoint(output)) * output_value')
    return nothing
end

function special_reverse_step!(::typeof(det), input, output, inv_input_value)
    if output != 0
        increment_adjoint!(input, scale!((adjoint(output) * value(output)), inv_input_value'))
    end
    return nothing
end
