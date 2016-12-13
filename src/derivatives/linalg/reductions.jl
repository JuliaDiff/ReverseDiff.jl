#######
# sum #
#######

# basic sum #
#-----------#

function Base.sum{V,D}(x::TrackedArray{V,D})
    tp = tape(x)
    out = track(sum(value(x)), D, tp)
    record!(tp, SpecialInstruction, sum, x, out)
    return out
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(sum)})
    input = instruction.input
    output = instruction.output
    istracked(input) && increment_deriv!(input, deriv(output))
    unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(sum)})
    input = instruction.input
    value!(instruction.output, sum(value(input)))
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

function Base.sum{V,D}(x::TrackedArray{V,D}, dims)
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

function Base.mean{V,D}(x::TrackedArray{V,D})
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

function record_dot{D}(x, y, ::Type{D})
    tp = tape(x, y)
    out = track(dot(value(x), value(y)), D, tp)
    cache = (similar(x, D), similar(y, D))
    record!(tp, SpecialInstruction, dot, (x, y), out, cache)
    return out
end

Base.dot{X,Y,D}(x::TrackedArray{X,D}, y::TrackedArray{Y,D}) = record_dot(x, y, D)

for A in ARRAY_TYPES
    @eval Base.dot{X,D}(x::TrackedArray{X,D}, y::$A) = record_dot(x, y, D)
    @eval Base.dot{Y,D}(x::$A, y::TrackedArray{Y,D}) = record_dot(x, y, D)
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
        copy!(b_tmp, value(b))
        scale!(output_deriv, b_tmp)
        increment_deriv!(a, b_tmp)
    end
    if istracked(b)
        copy!(a_tmp, value(a))
        scale!(output_deriv, a_tmp)
        increment_deriv!(b, a_tmp)
    end
    unseed!(output)
    return nothing
end
