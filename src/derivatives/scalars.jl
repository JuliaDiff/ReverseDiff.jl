###################
# ForwardOptimize #
###################

# unary #
#-------#

for f in FORWARD_UNARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(t::Tracked) = ForwardOptimize($f)(t)
end

# binary #
#--------#

for f in FORWARD_BINARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(a::Tracked, b::Tracked) = ForwardOptimize($f)(a, b)
    for R in REAL_TYPES
        @eval begin
            @inline Base.$(f)(a::Tracked, b::$R) = ForwardOptimize($f)(a, b)
            @inline Base.$(f)(a::$R, b::Tracked) = ForwardOptimize($f)(a, b)
        end
    end
end

################
# SkipOptimize #
################

# unary #
#-------#

for f in SKIPPED_UNARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(t::Tracked) = SkipOptimize($(f))(t)
end

# binary #
#--------#

for f in SKIPPED_BINARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(a::Tracked, b::Tracked) = SkipOptimize($(f))(a, b)
    for R in REAL_TYPES
        @eval begin
            @inline Base.$(f)(a::$R, b::Tracked) = SkipOptimize($(f))(a, b)
            @inline Base.$(f)(a::Tracked, b::$R) = SkipOptimize($(f))(a, b)
        end
    end
end

###########
# reverse #
###########

# f(::Number)::Number
function scalar_reverse_step!(input::Tracked, output::Tracked, deriv::RefValue)
    increment_adjoint!(input, adjoint(output) * deriv[])
    return nothing
end

# f(::Number, ::Number)::Number
function scalar_reverse_step!{A,B}(inputs::Tuple{A,B}, output::Tracked, grad::RefValue)
    a, b = inputs
    output_adjoint = adjoint(output)
    if A <: Tracked && B <: Tracked
        a_partial, b_partial = grad[]
        increment_adjoint!(a, output_adjoint * a_partial)
        increment_adjoint!(b, output_adjoint * b_partial)
    elseif A <: Tracked
        increment_adjoint!(a, output_adjoint * grad[])
    else
        increment_adjoint!(b, output_adjoint * grad[])
    end
    return nothing
end

###########
# forward #
###########

# f(::Number)::Number
function scalar_forward_step!(f, input::Tracked, output::Tracked, deriv::RefValue)
    dual = f(Dual(value(input), one(valtype(input))))
    setvalue!(output, value(dual))
    deriv[] = partials(dual, 1)
    return nothing
end

# f(::Number, ::Number)::Number
function scalar_forward_step!{A,B}(f, inputs::Tuple{A,B}, output::Tracked, grad::RefValue)
    a, b = inputs
    if A <: Tracked && B <: Tracked
        VA, VB = valtype(A), valtype(B)
        dual_a = Dual(value(a), one(VA), zero(VA))
        dual_b = Dual(value(b), zero(VB), one(VB))
        dual_c = f(dual_a, dual_b)
        setvalue!(output, value(dual_c))
        grad[] = partials(dual_c)
    elseif A <: Tracked
        dual = f(Dual(value(a), one(valtype(a))), b)
        setvalue!(output, value(dual))
        grad[] = partials(dual, 1)
    else
        dual = f(a, Dual(value(b), one(valtype(b))))
        setvalue!(output, value(dual))
        grad[] = partials(dual, 1)
    end
    return nothing
end
