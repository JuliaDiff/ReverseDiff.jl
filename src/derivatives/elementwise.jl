#####################
# SkipOptimized map #
#####################

# dispatch #
#----------#

for f in SKIPPED_UNARY_SCALAR_FUNCS
    @eval @inline Base.map(f::typeof($f), t::TrackedArray) = map(SkipOptimize(f), t)
end

for f in SKIPPED_BINARY_SCALAR_FUNCS
    @eval begin
        @inline Base.map(f::typeof($f), x::TrackedArray, y::TrackedArray) = map(SkipOptimize(f), x, y)
        @inline Base.map(f::typeof($f), x::TrackedArray, y::TrackedReal) = map(SkipOptimize(f), x, y)
        @inline Base.map(f::typeof($f), x::TrackedReal, y::TrackedArray) = map(SkipOptimize(f), x, y)
    end
    for A in ARRAY_TYPES
        @eval begin
            @inline Base.map(f::typeof($f), x::$A, y::TrackedArray) = map(SkipOptimize(f), x, y)
            @inline Base.map(f::typeof($f), x::TrackedArray, y::$A) = map(SkipOptimize(f), x, y)
            @inline Base.map(f::typeof($f), x::$A, y::TrackedReal) = map(SkipOptimize(f), x, y)
            @inline Base.map(f::typeof($f), x::TrackedReal, y::$A) = map(SkipOptimize(f), x, y)
        end
    end
    for R in REAL_TYPES
        @eval begin
            @inline Base.map(f::typeof($f), x::$R, y::TrackedArray) = map(SkipOptimize(f), x, y)
            @inline Base.map(f::typeof($f), x::TrackedArray, y::$R) = map(SkipOptimize(f), x, y)
        end
    end
end

# record #
#--------#

@inline Base.map(f::SkipOptimize{F}, t::TrackedArray) where {F} = map(f.f, value(t))

@inline Base.map(f::SkipOptimize{F}, x::TrackedArray, y::TrackedArray) where {F} = map(f.f, value(x), value(y))
@inline Base.map(f::SkipOptimize{F}, x::TrackedArray, y::TrackedReal) where {F} = map(f.f, value(x), value(y))
@inline Base.map(f::SkipOptimize{F}, x::TrackedReal, y::TrackedArray) where {F} = map(f.f, value(x), value(y))
for A in ARRAY_TYPES
    @eval begin
        @inline Base.map(f::SkipOptimize{F}, x::$A, y::TrackedArray) where {F} = map(f.f, value(x), value(y))
        @inline Base.map(f::SkipOptimize{F}, x::TrackedArray, y::$A) where {F} = map(f.f, value(x), value(y))
        @inline Base.map(f::SkipOptimize{F}, x::$A, y::TrackedReal) where {F} = map(f.f, value(x), value(y))
        @inline Base.map(f::SkipOptimize{F}, x::TrackedReal, y::$A) where {F} = map(f.f, value(x), value(y))
    end
end
for R in REAL_TYPES
    @eval begin
        @inline Base.map(f::SkipOptimize{F}, x::$R, y::TrackedArray) where {F} = map(f.f, value(x), value(y))
        @inline Base.map(f::SkipOptimize{F}, x::TrackedArray, y::$R) where {F} = map(f.f, value(x), value(y))
    end
end

########################
# ForwardOptimized map #
########################

# dispatch #
#----------#

for (M, f, arity) in DiffRules.diffrules(; filter_modules=nothing)
    if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), f))
        @warn "$M.$f is not available and hence rule for it can not be defined"
        continue  # Skip rules for methods not defined in the current scope
    end
    if arity == 1
        @eval @inline Base.map(f::typeof($M.$f), t::TrackedArray) = map(ForwardOptimize(f), t)
    elseif arity == 2
        (M, f) in SKIPPED_DIFFRULES && continue
        @eval @inline Base.map(f::typeof($M.$f), x::TrackedArray, y::TrackedArray) = map(ForwardOptimize(f), x, y)
        for A in ARRAY_TYPES
            @eval begin
                @inline Base.map(f::typeof($M.$f), x::$A, y::TrackedArray) = map(ForwardOptimize(f), x, y)
                @inline Base.map(f::typeof($M.$f), x::TrackedArray, y::$A) = map(ForwardOptimize(f), x, y)
            end
        end
        for R in REAL_TYPES
            @eval begin
                @inline Base.map(f::typeof($M.$f), x::$R, y::TrackedArray) = map(ForwardOptimize(f), x, y)
                @inline Base.map(f::typeof($M.$f), x::TrackedArray, y::$R) = map(ForwardOptimize(f), x, y)
            end
        end
    end
end

# record #
#--------#

function Base.map(f::ForwardOptimize{F}, x::TrackedArray{X,D}) where {F,X,D}
    T = promote_type(X, D)
    result = DiffResults.DiffResult(zero(T), zero(T))
    df = v -> ForwardDiff.derivative!(result, f.f, v)
    results = map(df, value(x))
    tp = tape(x)
    out = track(map(DiffResults.value, results), D, tp)
    cache = (results, df, index_bound(x, out), nothing)
    record!(tp, SpecialInstruction, map, x, out, cache)
    return out
end

for A in ARRAY_TYPES
    @eval function Base.map(f::ForwardOptimize{F}, x::TrackedArray{X,D}, y::$A) where {F,X,D}
        result = DiffResults.GradientResult(SVector(zero(X)))
        df = (vx, vy) -> let vy=vy
            ForwardDiff.gradient!(result, s -> f.f(s[1], vy), SVector(vx))
        end
        results = map(df, value(x), value(y))
        tp = tape(x)
        out = track(map(DiffResults.value, results), D, tp)
        cache = (results, df, index_bound(x, out), index_bound(y, out))
        record!(tp, SpecialInstruction, map, (x, y), out, cache)
        return out
    end
    @eval function Base.map(f::ForwardOptimize{F}, x::$A, y::TrackedArray{Y,D}) where {F,Y,D}
        result = DiffResults.GradientResult(SVector(zero(Y)))
        df = (vx, vy) -> let vx=vx
            ForwardDiff.gradient!(result, s -> f.f(vx, s[1]), SVector(vy))
        end
        results = map(df, value(x), value(y))
        tp = tape(y)
        out = track(map(DiffResults.value, results), D, tp)
        cache = (results, df, index_bound(x, out), index_bound(y, out))
        record!(tp, SpecialInstruction, map, (x, y), out, cache)
        return out
    end
end

function Base.map(f::ForwardOptimize{F}, x::TrackedArray{X,D}, y::TrackedArray{Y,D}) where {F,X,Y,D}
    result = DiffResults.GradientResult(SVector(zero(D), zero(D)))
    df = (vx, vy) -> ForwardDiff.gradient!(result, s -> f.f(s[1], s[2]), SVector(vx, vy))
    results = map(df, value(x), value(y))
    tp = tape(x, y)
    out = track(map(DiffResults.value, results), D, tp)
    cache = (results, df, index_bound(x, out), index_bound(y, out))
    record!(tp, SpecialInstruction, map, (x, y), out, cache)
    return out
end

################
# forward pass #
################

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(map)})
    input, output = instruction.input, instruction.output
    results, df, _, _ = instruction.cache
    if istracked(input)
        map!(df, results, value(input))
    else
        a, b = input
        pull_value!(a)
        pull_value!(b)
        map!(df, results, value(a), value(b))
    end
    output_value = value(output)
    for i in eachindex(output_value)
        output_value[i] = DiffResults.value(results[i])
    end
    return nothing
end

################
# reverse pass #
################

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(map)})
    input = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    results = first(instruction.cache)
    if istracked(input)
        diffresult_increment_deriv!(Nothing, input, output_deriv, results, 1)
    else
        a, b = input
        p = 0
        if istracked(a)
            p += 1
            diffresult_increment_deriv!(Nothing, a, output_deriv, results, p)
        end
        if istracked(b)
            p += 1
            diffresult_increment_deriv!(Nothing, b, output_deriv, results, p)
        end
    end
    unseed!(output)
    return nothing
end
