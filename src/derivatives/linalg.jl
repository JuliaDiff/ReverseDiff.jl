########################
# addition/subtraction #
########################

# sum #
#-----#

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

# + #
#---#

# in-place version (necessary for nested differentiation to work)

function plus!(out, x, y)
    for i in eachindex(out)
        out[i] = x[i] + y[i]
    end
    return out
end

@inline plus!(out::TrackedArray, x::TrackedArray, y::TrackedArray) = record_plus!(out, x, y)

for A in ARRAY_TYPES
    @eval @inline plus!(out::TrackedArray, x::TrackedArray, y::$(A)) = record_plus!(out, x, y)
    @eval @inline plus!(out::TrackedArray, x::$(A), y::TrackedArray) = record_plus!(out, x, y)
end

function record_plus!(out::TrackedArray, x, y)
    copy!(value(out), value(x) + value(y))
    record!(tape(x, y), SpecialInstruction, +, (x, y), out)
    return out
end

# Base allocating version

@inline Base.:+{X,Y,D}(x::TrackedArray{X,D}, y::TrackedArray{Y,D}) = record_plus(x, y, D)

for A in ARRAY_TYPES
    @eval @inline Base.:+{V,D}(x::TrackedArray{V,D}, y::$(A)) = record_plus(x, y, D)
    @eval @inline Base.:+{V,D}(x::$(A), y::TrackedArray{V,D}) = record_plus(x, y, D)
end

function record_plus{D}(x, y, ::Type{D})
    tp = tape(x, y)
    out = track(value(x) + value(y), D, tp)
    record!(tp, SpecialInstruction, +, (x, y), out)
    return out
end

# reverse/forward passes

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(+)})
    a, b = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    istracked(a) && increment_deriv!(a, output_deriv)
    istracked(b) && increment_deriv!(b, output_deriv)
    unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(+)})
    a, b = instruction.input
    pull_value!(a)
    pull_value!(b)
    plus!(value(instruction.output), value(a), value(b))
    return nothing
end

# - #
#---#

# in-place version (necessary for nested differentiation to work)

function minus!(out, x)
    for i in eachindex(out)
        out[i] = -(x[i])
    end
end

function minus!(out, x, y)
    for i in eachindex(out)
        out[i] = x[i] - y[i]
    end
end

@inline minus!(out::TrackedArray, x::TrackedArray, y::TrackedArray) = record_minus!(out, x, y)
@inline minus!(out::TrackedArray, x::TrackedArray) = record_minus!(out, x, y)

for A in ARRAY_TYPES
    @eval @inline minus!(out::TrackedArray, x::TrackedArray, y::$(A)) = record_minus!(out, x, y)
    @eval @inline minus!(out::TrackedArray, x::$(A), y::TrackedArray) = record_minus!(out, x, y)
    @eval @inline minus!(out::TrackedArray, x::$(A)) = record_minus!(out, x)
end

function record_minus!(out::TrackedArray, x)
    copy!(value(out), -(value(x)))
    record!(tape(x), SpecialInstruction, -, x, out)
    return out
end

function record_minus!(out::TrackedArray, x, y)
    copy!(value(out), value(x) - value(y))
    record!(tape(x, y), SpecialInstruction, -, (x, y), out)
    return out
end

# Base allocating version

Base.:-{X,Y,D}(x::TrackedArray{X,D}, y::TrackedArray{Y,D}) = record_minus(x, y, D)

for A in ARRAY_TYPES
    @eval Base.:-{V,D}(x::TrackedArray{V,D}, y::$(A)) = record_minus(x, y, D)
    @eval Base.:-{V,D}(x::$(A), y::TrackedArray{V,D}) = record_minus(x, y, D)
end

function Base.:-{V,D}(x::TrackedArray{V,D})
    tp = tape(x)
    out = track(-(value(x)), D, tp)
    record!(tp, SpecialInstruction, -, x, out)
    return out
end

function record_minus{D}(x, y, ::Type{D})
    tp = tape(x, y)
    out = track(value(x) - value(y), D, tp)
    record!(tp, SpecialInstruction, -, (x, y), out)
    return out
end

# reverse/forward passes

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(-)})
    input = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    if istracked(input)
        decrement_deriv!(input, output_deriv)
    else
        a, b = input
        istracked(a) && increment_deriv!(a, output_deriv)
        istracked(b) && decrement_deriv!(b, output_deriv)
    end
    unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(-)})
    input = instruction.input
    output = instruction.output
    if istracked(input)
        minus!(value(output), value(input))
    else
        a, b = input
        pull_value!(a)
        pull_value!(b)
        minus!(value(output), value(a), value(b))
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

for (f!, f) in A_MUL_B_FUNCS
    record_f = Symbol(string("record_", f))
    record_f! = Symbol(string("record_", f!))

    @eval begin
        @inline function $(record_f){D}(x, y, ::Type{D})
            tp = tape(x, y)
            out = track($(f)(value(x), value(y)), D, tp)
            cache = (similar(x, D), similar(y, D))
            record!(tp, SpecialInstruction, $(f), (x, y), out, cache)
            return out
        end

        @inline function $(record_f!){V,D}(out::TrackedArray{V,D}, x, y)
            copy!(value(out), $(f)(value(x), value(y)))
            cache = (similar(x, D), similar(y, D))
            record!(tape(x, y), SpecialInstruction, $(f), (x, y), out, cache)
            return out
        end
    end

    @eval Base.$(f){X,Y,D}(x::TrackedArray{X,D}, y::TrackedArray{Y,D}) = $(record_f)(x, y, D)
    @eval Base.$(f!){V,X,Y,D}(out::TrackedArray{V,D}, x::TrackedArray{X,D}, y::TrackedArray{Y,D}) = $(record_f!)(out, x, y)

    for T in ARRAY_TYPES
        @eval Base.$(f){V,D}(x::TrackedArray{V,D}, y::$(T)) = $(record_f)(x, y, D)
        @eval Base.$(f){V,D}(x::$(T), y::TrackedArray{V,D}) = $(record_f)(x, y, D)
        @eval Base.$(f!)(out::TrackedArray, x::TrackedArray, y::$(T)) = $(record_f!)(out, x, y)
        @eval Base.$(f!)(out::TrackedArray, x::$(T), y::TrackedArray) = $(record_f!)(out, x, y)
    end
end

# forward pass #
#--------------#

for (f!, f) in A_MUL_B_FUNCS
    @eval begin
        @noinline function special_forward_exec!(instruction::SpecialInstruction{typeof($f)})
            a, b = instruction.input
            pull_value!(a)
            pull_value!(b)
            $(f!)(value(instruction.output), value(a), value(b))
            return nothing
        end
    end
end

# reverse pass #
#--------------#

### *

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(*)})
    a, b = instruction.input
    a_tmp, b_tmp = instruction.cache
    output = instruction.output
    output_deriv = deriv(output)
    istracked(a) && increment_deriv!(a, A_mul_Bc!(a_tmp, output_deriv, value(b)))
    istracked(b) && increment_deriv!(b, Ac_mul_B!(b_tmp, value(a), output_deriv))
    unseed!(output)
    return nothing
end

### A_mul_Bt

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(A_mul_Bt)})
    a, b = instruction.input
    a_tmp, b_tmp = instruction.cache
    output = instruction.output
    output_deriv = deriv(output)
    istracked(a) && increment_deriv!(a, A_mul_B!(a_tmp, output_deriv, value(b)))
    istracked(b) && increment_deriv!(b, At_mul_B!(b_tmp, output_deriv, value(a)))
    unseed!(output)
    return nothing
end

### At_mul_B

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(At_mul_B)})
    a, b = instruction.input
    a_tmp, b_tmp = instruction.cache
    output = instruction.output
    output_deriv = deriv(output)
    istracked(a) && increment_deriv!(a, A_mul_Bt!(a_tmp, value(b), output_deriv))
    istracked(b) && increment_deriv!(b, A_mul_B!(b_tmp, value(a), output_deriv))
    unseed!(output)
    return nothing
end

### At_mul_Bt

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(At_mul_Bt)})
    a, b = instruction.input
    a_tmp, b_tmp = instruction.cache
    output = instruction.output
    output_deriv = deriv(output)
    istracked(a) && increment_deriv!(a, At_mul_Bt!(a_tmp, value(b), output_deriv))
    istracked(b) && increment_deriv!(b, At_mul_Bt!(b_tmp, output_deriv, value(a)))
    unseed!(output)
    return nothing
end

### A_mul_Bc

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(A_mul_Bc)})
    a, b = instruction.input
    a_tmp, b_tmp = instruction.cache
    output = instruction.output
    output_deriv = deriv(output)
    istracked(a) && increment_deriv!(a, A_mul_B!(a_tmp, output_deriv, value(b)))
    istracked(b) && increment_deriv!(b, Ac_mul_B!(b_tmp, output_deriv, value(a)))
    unseed!(output)
    return nothing
end

### Ac_mul_B

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(Ac_mul_B)})
    a, b = instruction.input
    a_tmp, b_tmp = instruction.cache
    output = instruction.output
    output_deriv = deriv(output)
    istracked(a) && increment_deriv!(a, A_mul_Bc!(a_tmp, value(b), output_deriv))
    istracked(b) && increment_deriv!(b, A_mul_B!(b_tmp, value(a), output_deriv))
    unseed!(output)
    return nothing
end

### Ac_mul_Bc

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(Ac_mul_Bc)})
    a, b = instruction.input
    a_tmp, b_tmp = instruction.cache
    output = instruction.output
    output_deriv = deriv(output)
    istracked(a) && increment_deriv!(a, Ac_mul_Bc!(a_tmp, value(b), output_deriv))
    istracked(b) && increment_deriv!(b, Ac_mul_Bc!(b_tmp, output_deriv, value(a)))
    unseed!(output)
    return nothing
end

#######################
# misc linear algebra #
#######################

# dot #
#-----#

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

# det #
#-----#

function Base.det{V,D}(x::TrackedArray{V,D})
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

# inv #
#-----#

function Base.inv{V,D}(x::TrackedArray{V,D})
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
    A_mul_Bc!(output_tmp1, output_deriv, output_value)
    Ac_mul_B!(output_tmp2, output_value, output_tmp1)
    decrement_deriv!(instruction.input, output_tmp2)
    unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(inv)})
    input = instruction.input
    value!(instruction.output, inv(value(input)))
    return nothing
end
