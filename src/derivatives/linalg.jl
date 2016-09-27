###########
# forward #
###########

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
    end

    @eval function Base.:-{V,S}(x::$(A){Tracked{V,S}})
        tp = tape(x)
        out = track(-(value(x)), S, tp)
        record!(tp, -, x, out)
        return out
    end

    # A_mul_B family #
    #----------------#

    for f in (:*,
              :A_mul_Bt, :At_mul_B, :At_mul_Bt,
              :A_mul_Bc, :Ac_mul_B, :Ac_mul_Bc)
        @eval function Base.$(f){X,Y,S}(x::$(A){Tracked{X,S}}, y::$(A){Tracked{Y,S}})
            tp = tape(x, y)
            xval, yval = value(x), value(y)
            out = track($(f)(xval, yval), S, tp)
            record!(tp, $(f), (x, y), out, (xval, yval))
            return out
        end
    end

    # in-place versions
    for (f!, f) in ((:A_mul_B!, :*),
                    (:A_mul_Bt!, :A_mul_Bt), (:At_mul_B!, :At_mul_B), (:At_mul_Bt!, :At_mul_Bt),
                    (:A_mul_Bc!, :A_mul_Bc), (:Ac_mul_B!, :Ac_mul_B), (:Ac_mul_Bc!, :Ac_mul_Bc))
        @eval function Base.$(f!){V,X,Y,S}(out::$(A){Tracked{V,S}},
                                           x::$(A){Tracked{X,S}},
                                           y::$(A){Tracked{Y,S}})
            tp = tape(x, y)
            xval, yval = value(x), value(y)
            track!(out, $(f)(xval, yval), tp)
            record!(tp, $(f), (x, y), out, (xval, yval))
            return out
        end
    end

    # linear algebra #
    #----------------#

    @eval function Base.inv{V,S}(x::$(A){Tracked{V,S}})
        tp = tape(x)
        outval = inv(value(x))
        out = track(outval, S, tp)
        record!(tp, inv, x, out, outval)
        return out
    end

    @eval function Base.det{V,S}(x::$(A){Tracked{V,S}})
        tp = tape(x)
        xval = value(x)
        det_xval = det(xval)
        out = track(det_xval, S, tp)
        if det_xval == 0
            record!(tp, det, x, out, nothing)
        else
            record!(tp, det, x, out, inv(xval))
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

function special_reverse_step!(::typeof(-), input, output, _)
    extract_and_decrement_adjoint!(input, output)
    return nothing
end

function special_reverse_step!{A,B}(::typeof(-), inputs::Tuple{A,B}, output::AbstractArray, _)
    extract_and_increment_adjoint!(inputs[1], output)
    extract_and_decrement_adjoint!(inputs[2], output)
    return nothing
end

# A_mul_B family #
#----------------#

function special_reverse_step!{A,B}(::typeof(*), inputs::Tuple{A,B}, output, vals)
    a, b = inputs
    aval, bval = vals
    output_adjoint = adjoint(output)
    increment_adjoint!(a, output_adjoint * bval')
    increment_adjoint!(b, aval' * output_adjoint)
    return nothing
end

function special_reverse_step!{A,B}(::typeof(A_mul_Bt), inputs::Tuple{A,B}, output, vals)
    a, b = inputs
    aval, bval = vals
    output_adjoint = adjoint(output)
    increment_adjoint!(a, output_adjoint   * bval)
    increment_adjoint!(b, output_adjoint.' * aval)
    return nothing
end

function special_reverse_step!{A,B}(::typeof(At_mul_B), inputs::Tuple{A,B}, output, vals)
    a, b = inputs
    aval, bval = vals
    output_adjoint = adjoint(output)
    increment_adjoint!(a, bval * output_adjoint.')
    increment_adjoint!(b, aval * output_adjoint)
    return nothing
end

function special_reverse_step!{A,B}(::typeof(At_mul_Bt), inputs::Tuple{A,B}, output, vals)
    a, b = inputs
    aval, bval = vals
    output_adjoint = adjoint(output)
    increment_adjoint!(a, (output_adjoint * bval).')
    increment_adjoint!(b, (aval * output_adjoint).')
    return nothing
end

function special_reverse_step!{A,B}(::typeof(A_mul_Bc), inputs::Tuple{A,B}, output, vals)
    a, b = inputs
    aval, bval = vals
    output_adjoint = adjoint(output)
    increment_adjoint!(a, output_adjoint  * bval)
    increment_adjoint!(b, output_adjoint' * aval)
    return nothing
end

function special_reverse_step!{A,B}(::typeof(Ac_mul_B), inputs::Tuple{A,B}, output, vals)
    a, b = inputs
    aval, bval = vals
    output_adjoint = adjoint(output)
    increment_adjoint!(a, bval * output_adjoint')
    increment_adjoint!(b, aval * output_adjoint)
    return nothing
end

function special_reverse_step!{A,B}(::typeof(Ac_mul_Bc), inputs::Tuple{A,B}, output, vals)
    a, b = inputs
    aval, bval = vals
    output_adjoint = adjoint(output)
    increment_adjoint!(a, (output_adjoint * bval)')
    increment_adjoint!(b, (aval * output_adjoint)')
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
