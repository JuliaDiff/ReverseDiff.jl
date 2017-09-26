###################
# ForwardOptimize #
###################

for (M, f, arity) in DiffRules.diffrules()
    if arity == 1
        @eval @inline $M.$(f)(t::TrackedReal) = ForwardOptimize($f)(t)
    elseif arity == 2
        @eval @inline $M.$(f)(a::TrackedReal, b::TrackedReal) = ForwardOptimize($f)(a, b)
        for R in REAL_TYPES
            @eval begin
                @inline $M.$(f)(a::TrackedReal, b::$R) = ForwardOptimize($f)(a, b)
                @inline $M.$(f)(a::$R, b::TrackedReal) = ForwardOptimize($f)(a, b)
            end
        end
    end
end

################
# SkipOptimize #
################

# unary #
#-------#

for f in SKIPPED_UNARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(t::TrackedReal) = SkipOptimize($(f))(t)
end

# binary #
#--------#

for f in SKIPPED_BINARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(a::TrackedReal, b::TrackedReal) = SkipOptimize($(f))(a, b)
    for R in REAL_TYPES
        @eval begin
            @inline Base.$(f)(a::$R, b::TrackedReal) = SkipOptimize($(f))(a, b)
            @inline Base.$(f)(a::TrackedReal, b::$R) = SkipOptimize($(f))(a, b)
        end
    end
end

###########
# reverse #
###########

@noinline function scalar_reverse_exec!{F,I,O,C}(instruction::ScalarInstruction{F,I,O,C})
    f = instruction.func
    input = instruction.input
    output = instruction.output
    partials = instruction.cache[]
    if istracked(input)
        increment_deriv!(input, deriv(output) * partials)
    else
        a, b = input
        output_deriv = deriv(output)
        a_partial, b_partial = partials
        if istracked(a) && istracked(b)
            increment_deriv!(a, output_deriv * a_partial)
            increment_deriv!(b, output_deriv * b_partial)
        elseif istracked(a)
            increment_deriv!(a, output_deriv * a_partial)
        else
            increment_deriv!(b, output_deriv * b_partial)
        end
    end
    unseed!(output)
    return nothing
end

###########
# forward #
###########

@noinline function scalar_forward_exec!{F,I,O,C}(instruction::ScalarInstruction{F,I,O,C})
    f = instruction.func
    input = instruction.input
    output = instruction.output
    cache = instruction.cache
    # these annotations are needed to help inference along
    VO = valtype(output)
    local result1::DiffResult{1,VO,Tuple{VO}}
    local result2::DiffResult{1,VO,Tuple{SVector{2,VO}}}
    if istracked(input)
        pull_value!(input)
        result1 = DiffResult(zero(VO), zero(VO))
        result1 = ForwardDiff.derivative!(result1, f, value(input))
        value!(output, DiffResults.value(result1))
        cache[] = DiffResults.derivative(result1)
    else
        a, b = input
        pull_value!(a)
        pull_value!(b)
        if istracked(a) && istracked(b)
            result2 = DiffResults.GradientResult(SVector(zero(VO), zero(VO)))
            result2 = ForwardDiff.gradient!(result2, x -> f(x[1], x[2]), SVector(value(a), value(b)))
            value!(output, DiffResults.value(result2))
            cache[] = DiffResults.gradient(result2)
        else
            result1 = DiffResult(zero(VO), zero(VO))
            if istracked(a)
                result1 = ForwardDiff.derivative!(result1, va -> f(va, b), value(a))
            else
                result1 = ForwardDiff.derivative!(result1, vb -> f(a, vb), value(b))
            end
            value!(output, DiffResults.value(result1))
            partial = DiffResults.derivative(result1)
            cache[] = SVector(partial, partial)
        end
    end
    return nothing
end
