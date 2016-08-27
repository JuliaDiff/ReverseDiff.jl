###################
# ForwardOptimize #
###################

# unary #
#-------#

for f in FORWARD_UNARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(t::TraceReal) = ForwardOptimize($f)(t)
end

# binary #
#--------#

for f in FORWARD_BINARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(a::TraceReal, b::TraceReal) = ForwardOptimize($f)(a, b)
    for R in REAL_TYPES
        @eval begin
            @inline Base.$(f)(a::TraceReal, b::$R) = ForwardOptimize($f)(a, b)
            @inline Base.$(f)(a::$R, b::TraceReal) = ForwardOptimize($f)(a, b)
        end
    end
end

################
# SkipOptimize #
################

# binary #
#--------#

for f in SKIP_BINARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(a::TraceReal, b::TraceReal) = SkipOptimize($(f))(a, b)
    for R in REAL_TYPES
        @eval begin
            @inline Base.$(f)(a::$R, b::TraceReal) = SkipOptimize($(f))(a, b)
            @inline Base.$(f)(a::TraceReal, b::$R) = SkipOptimize($(f))(a, b)
        end
    end
end
