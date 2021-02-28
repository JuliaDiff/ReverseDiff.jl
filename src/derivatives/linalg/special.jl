########
# copy #
########

for (T, f) in [(:Adjoint, :adjoint), (:Transpose, :transpose)]
    _f = Symbol(:_copy, f)
    @eval begin
        Base.copy(A::$T{<:TrackedReal, <:TrackedVecOrMat}) = $_f(parent(A))
        $_f(A) = copy($f(A))
        $_f(A::TrackedVecOrMat) = track($_f, A)
        @grad function $_f(A::AbstractVecOrMat)
            return copy($f(value(A))), ∇ -> (copy($f(∇)),)
        end
    end
end

#######
# det #
#######

function LinearAlgebra.det(x::TrackedArray{V,D}) where {V,D}
    tp = tape(x)
    x_value = value(x)
    det_x_value = det(x_value)
    out = track(det_x_value, D, tp)
    record!(tp, SpecialInstruction, det, x, out)
    return out
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(det)})
    output = instruction.output
    if output != 0
        input = instruction.input
        input_deriv = deriv(input)
        inv_input_value = inv(value(input))
        k = deriv(output) * value(output)
        for i in 1:size(input_deriv, 1), j in 1:size(input_deriv, 2)
            input_deriv[i, j] += k * inv_input_value[j, i]
        end
    end
    unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(det)})
    output, input = instruction.output, instruction.input
    value!(output, det(value(input)))
    return nothing
end

#######
# inv #
#######

function LinearAlgebra.inv(x::TrackedArray{V,D}) where {V,D}
    tp = tape(x)
    out_value = inv(value(x))
    out = track(out_value, D, tp)
    cache = (similar(out_value, D), similar(out_value, D))
    record!(tp, SpecialInstruction, inv, x, out, cache)
    return out
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(inv)})
    output = instruction.output
    output_value, output_deriv = value(output), deriv(output)
    output_tmp1, output_tmp2 = instruction.cache
    mul!(output_tmp1, output_deriv, adjoint(output_value))
    mul!(output_tmp2, adjoint(output_value), output_tmp1)
    decrement_deriv!(instruction.input, output_tmp2)
    unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(inv)})
    input = instruction.input
    value!(instruction.output, inv(value(input)))
    return nothing
end
