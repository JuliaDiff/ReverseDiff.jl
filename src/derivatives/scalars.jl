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
function scalar_reverse_step!(input::Tracked, output::Tracked, deriv::Partials{1})
    input.adjoint += adjoint(output) * deriv[1]
    return nothing
end

# f(::Number...)::Number
function scalar_reverse_step!{N}(inputs::Tuple, output::Tracked, grad::Partials{N})
    for i in 1:N
        inputs[i].adjoint += adjoint(output) * grad[i]
    end
    return nothing
end
