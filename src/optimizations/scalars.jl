###################
# ForwardOptimize #
###################

# unary #
#-------#

for f in FORWARD_UNARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(t::Tracer) = ForwardOptimize($f)(t)
end

# binary #
#--------#

for f in FORWARD_BINARY_SCALAR_FUNCS
    @eval @inline Base.$(f)(a::Tracer, b::Tracer) = ForwardOptimize($f)(a, b)
    for R in REAL_TYPES
        @eval begin
            @inline Base.$(f)(a::Tracer, b::$R) = ForwardOptimize($f)(a, b)
            @inline Base.$(f)(a::$R, b::Tracer) = ForwardOptimize($f)(a, b)
        end
    end
end

################
# SkipOptimize #
################

# binary #
#--------#

for f in SKIPPED_SCALAR_COMPARATORS
    @eval @inline Base.$(f)(a::Tracer, b::Tracer) = SkipOptimize($(f))(a, b)
    for R in REAL_TYPES
        @eval begin
            @inline Base.$(f)(a::$R, b::Tracer) = SkipOptimize($(f))(a, b)
            @inline Base.$(f)(a::Tracer, b::$R) = SkipOptimize($(f))(a, b)
        end
    end
end
