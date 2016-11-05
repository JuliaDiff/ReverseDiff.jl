###################
# ForwardOptimize #
###################

# unary #
#-------#

for f in FORWARD_UNARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(t::TrackedReal) = ForwardOptimize($f)(t)
end

# binary #
#--------#

for f in FORWARD_BINARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(a::TrackedReal, b::TrackedReal) = ForwardOptimize($f)(a, b)
    for R in REAL_TYPES
        @eval begin
            @inline Base.$(f)(a::TrackedReal, b::$R) = ForwardOptimize($f)(a, b)
            @inline Base.$(f)(a::$R, b::TrackedReal) = ForwardOptimize($f)(a, b)
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
    local dual1::Dual{1,valtype(output)}
    local dual2::Dual{2,valtype(output)}
    if istracked(input)
        pull_value!(input)
        dual1 = f(Dual(value(input), one(valtype(input))))
        value!(output, ForwardDiff.value(dual1))
        cache[] = ForwardDiff.partials(dual1, 1)
    else
        a, b = input
        pull_value!(a)
        pull_value!(b)
        if istracked(a) && istracked(b)
            VA, VB = valtype(a), valtype(b)
            dual2 = f(Dual(value(a), one(VA), zero(VA)), Dual(value(b), zero(VB), one(VB)))
            value!(output, ForwardDiff.value(dual2))
            cache[] = ForwardDiff.partials(dual2)
        else
            if istracked(a)
                dual1 = f(Dual(value(a), one(valtype(a))), b)
            else
                dual1 = f(a, Dual(value(b), one(valtype(b))))
            end
            value!(output, ForwardDiff.value(dual1))
            partial = ForwardDiff.partials(dual1, 1)
            cache[] = Partials((partial, partial))
        end
    end
    return nothing
end
