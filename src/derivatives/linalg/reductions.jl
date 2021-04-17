#######
# sum #
#######

# basic sum #
#-----------#

function Base.sum(x::TrackedArray{V,D}; dims=:) where {V,D}
    tp = tape(x)
    out = track(sum(value(x), dims = dims), D, tp)
    record!(tp, SpecialInstruction, sum, (x, dims), out)
    return out
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(sum)})
    input, dims = instruction.input
    output = instruction.output
    if istracked(input)
        if dims === Colon()
            increment_deriv!(input, deriv(output))
        else
            increment_deriv!(input, zero(value(input)) .+ deriv(output))
        end
    end
    unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(sum)})
    input, dims = instruction.input
    value!(instruction.output, sum(value(input); dims = dims))
    return nothing
end

# sum over dimensions #
#---------------------#

function record_sum!(y::TrackedArray, x)
    sum!(value(y), value(x))
    record!(tape(x, y), SpecialInstruction, sum!, x, y, index_bound(y, x))
    return y
end

Base.sum!(y::TrackedArray, x::TrackedArray) = record_sum!(y, x)
Base.sum!(y::TrackedArray, x::AbstractArray) = record_sum!(y, x)

function Base.sum(x::TrackedArray{V,D}, dims) where {V,D}
    tp = tape(x)
    out = track(sum(value(x), dims), D, tp)
    record!(tp, SpecialInstruction, sum!, x, out, index_bound(out, x))
    return out
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(sum!)})
    input = instruction.input
    output = instruction.output
    bound = instruction.cache
    istracked(input) && reduction_increment_deriv!(input, deriv(output), bound)
    unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(sum!)})
    sum!(value(instruction.output), value(instruction.input))
    return nothing
end

########
# mean #
########

function Statistics.mean(x::TrackedArray{V,D}) where {V,D}
    tp = tape(x)
    out = track(mean(value(x)), D, tp)
    record!(tp, SpecialInstruction, mean, x, out)
    return out
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(mean)})
    input = instruction.input
    output = instruction.output
    istracked(input) && increment_deriv!(input, inv(length(input)) * deriv(output))
    unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(mean)})
    input = instruction.input
    value!(instruction.output, mean(value(input)))
    return nothing
end

#######
# dot #
#######

function record_dot(x, y, ::Type{D}) where D
    tp = tape(x, y)
    out = track(dot(value(x), value(y)), D, tp)
    cache = (similar(x, D), similar(y, D))
    record!(tp, SpecialInstruction, dot, (x, y), out, cache)
    return out
end

LinearAlgebra.dot(x::TrackedArray{X,D}, y::TrackedArray{Y,D}) where {X,Y,D} = record_dot(x, y, D)

for A in ARRAY_TYPES
    @eval LinearAlgebra.dot(x::TrackedArray{X,D}, y::$A) where {X,D} = record_dot(x, y, D)
    @eval LinearAlgebra.dot(x::$A, y::TrackedArray{Y,D}) where {Y,D} = record_dot(x, y, D)
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(dot)})
    a, b = instruction.input
    pull_value!(a)
    pull_value!(b)
    value!(instruction.output, dot(value(a), value(b)))
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(dot)})
    a, b = instruction.input
    a_tmp, b_tmp = instruction.cache
    output = instruction.output
    output_deriv = deriv(output)
    if istracked(a)
        copyto!(b_tmp, value(b))
        lmul!(output_deriv, b_tmp)
        increment_deriv!(a, b_tmp)
    end
    if istracked(b)
        copyto!(a_tmp, value(a))
        lmul!(output_deriv, a_tmp)
        increment_deriv!(b, a_tmp)
    end
    unseed!(output)
    return nothing
end
