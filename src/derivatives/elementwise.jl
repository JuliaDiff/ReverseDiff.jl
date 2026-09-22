#####################
# SkipOptimized map #
#####################

# dispatch #
#----------#

for g in (:map,), f in SKIPPED_UNARY_SCALAR_FUNCS
    @eval @inline Base.$(g)(f::typeof($f), t::TrackedArray) = $(g)(SkipOptimize(f), t)
end

for g in (:map,), f in SKIPPED_BINARY_SCALAR_FUNCS
    @eval begin
        @inline Base.$(g)(f::typeof($f), x::TrackedArray, y::TrackedArray) = $(g)(SkipOptimize(f), x, y)
        @inline Base.$(g)(f::typeof($f), x::TrackedArray, y::TrackedReal) = $(g)(SkipOptimize(f), x, y)
        @inline Base.$(g)(f::typeof($f), x::TrackedReal, y::TrackedArray) = $(g)(SkipOptimize(f), x, y)
    end
    for A in ARRAY_TYPES
        @eval begin
            @inline Base.$(g)(f::typeof($f), x::$A, y::TrackedArray) = $(g)(SkipOptimize(f), x, y)
            @inline Base.$(g)(f::typeof($f), x::TrackedArray, y::$A) = $(g)(SkipOptimize(f), x, y)
            @inline Base.$(g)(f::typeof($f), x::$A, y::TrackedReal) = $(g)(SkipOptimize(f), x, y)
            @inline Base.$(g)(f::typeof($f), x::TrackedReal, y::$A) = $(g)(SkipOptimize(f), x, y)
        end
    end
    for R in REAL_TYPES
        @eval begin
            @inline Base.$(g)(f::typeof($f), x::$R, y::TrackedArray) = $(g)(SkipOptimize(f), x, y)
            @inline Base.$(g)(f::typeof($f), x::TrackedArray, y::$R) = $(g)(SkipOptimize(f), x, y)
        end
    end
end

# record #
#--------#

for g in (:map,)
    @eval @inline Base.$(g)(f::SkipOptimize{F}, t::TrackedArray) where {F} = $(g)(f.f, value(t))
end

for g in (:map,)
    @eval begin
        @inline Base.$(g)(f::SkipOptimize{F}, x::TrackedArray, y::TrackedArray) where {F} = $(g)(f.f, value(x), value(y))
        @inline Base.$(g)(f::SkipOptimize{F}, x::TrackedArray, y::TrackedReal) where {F} = $(g)(f.f, value(x), value(y))
        @inline Base.$(g)(f::SkipOptimize{F}, x::TrackedReal, y::TrackedArray) where {F} = $(g)(f.f, value(x), value(y))
    end
    for A in ARRAY_TYPES
        @eval begin
            @inline Base.$(g)(f::SkipOptimize{F}, x::$A, y::TrackedArray) where {F} = $(g)(f.f, value(x), value(y))
            @inline Base.$(g)(f::SkipOptimize{F}, x::TrackedArray, y::$A) where {F} = $(g)(f.f, value(x), value(y))
            @inline Base.$(g)(f::SkipOptimize{F}, x::$A, y::TrackedReal) where {F} = $(g)(f.f, value(x), value(y))
            @inline Base.$(g)(f::SkipOptimize{F}, x::TrackedReal, y::$A) where {F} = $(g)(f.f, value(x), value(y))
        end
    end
    for R in REAL_TYPES
        @eval begin
            @inline Base.$(g)(f::SkipOptimize{F}, x::$R, y::TrackedArray) where {F} = $(g)(f.f, value(x), value(y))
            @inline Base.$(g)(f::SkipOptimize{F}, x::TrackedArray, y::$R) where {F} = $(g)(f.f, value(x), value(y))
        end
    end
end

########################
# ForwardOptimized map #
########################

# dispatch #
#----------#

for g in (:map,), (M, f, arity) in DiffRules.diffrules(; filter_modules=nothing)
    if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), f))
        @warn "$M.$f is not available and hence rule for it can not be defined"
        continue  # Skip rules for methods not defined in the current scope
    end
    if arity == 1
        @eval @inline Base.$(g)(f::typeof($M.$f), t::TrackedArray) = $(g)(ForwardOptimize(f), t)
    elseif arity == 2
        (M, f) in SKIPPED_DIFFRULES && continue
        @eval begin
            @inline Base.$(g)(f::typeof($M.$f), x::TrackedArray, y::TrackedArray) = $(g)(ForwardOptimize(f), x, y)
            @inline Base.$(g)(f::typeof($M.$f), x::TrackedArray, y::TrackedReal) = $(g)(ForwardOptimize(f), x, y)
            @inline Base.$(g)(f::typeof($M.$f), x::TrackedReal, y::TrackedArray) = $(g)(ForwardOptimize(f), x, y)
        end
        for A in ARRAY_TYPES
            @eval begin
                @inline Base.$(g)(f::typeof($M.$f), x::$A, y::TrackedArray) = $(g)(ForwardOptimize(f), x, y)
                @inline Base.$(g)(f::typeof($M.$f), x::TrackedArray, y::$A) = $(g)(ForwardOptimize(f), x, y)
                @inline Base.$(g)(f::typeof($M.$f), x::$A, y::TrackedReal) = $(g)(ForwardOptimize(f), x, y)
                @inline Base.$(g)(f::typeof($M.$f), x::TrackedReal, y::$A) = $(g)(ForwardOptimize(f), x, y)
            end
        end
        for R in REAL_TYPES
            @eval begin
                @inline Base.$(g)(f::typeof($M.$f), x::$R, y::TrackedArray) = $(g)(ForwardOptimize(f), x, y)
                @inline Base.$(g)(f::typeof($M.$f), x::TrackedArray, y::$R) = $(g)(ForwardOptimize(f), x, y)
            end
        end
    end
end

# record #
#--------#

for g in (:map,)
    @eval function Base.$(g)(f::ForwardOptimize{F}, x::TrackedArray{X,D}) where {F,X,D}
        T = promote_type(X, D)
        result = DiffResults.DiffResult(zero(T), zero(T))
        df = v -> ForwardDiff.derivative!(result, f.f, v)
        results = $(g)(df, value(x))
        tp = tape(x)
        out = track(map(DiffResults.value, results), D, tp)
        cache = (results, df, index_bound(x, out), nothing)
        record!(tp, SpecialInstruction, $(g), x, out, cache)
        return out
    end
    for A in ARRAY_TYPES
        @eval function Base.$(g)(f::ForwardOptimize{F}, x::TrackedReal{X,D}, y::$A) where {F,X,D}
            result = DiffResults.DiffResult(zero(X), zero(D))
            df = let result=result
                (vx, vy) -> let vy=vy
                    ForwardDiff.derivative!(result, s -> f.f(s, vy), vx)
                end
            end
            results = $(g)(df, value(x), value(y))
            tp = tape(x)
            out = track(map(DiffResults.value, results), D, tp)
            cache = (results, df, index_bound(x, out), index_bound(y, out))
            record!(tp, SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end
        @eval function Base.$(g)(f::ForwardOptimize{F}, x::$A, y::TrackedReal{Y,D}) where {F,Y,D}
            result = DiffResults.DiffResult(zero(Y), zero(D))
            df = let result=result
                (vx, vy) -> let vx=vx
                    ForwardDiff.derivative!(result, s -> f.f(vx, s), vy)
                end
            end
            results = $(g)(df, value(x), value(y))
            tp = tape(y)
            out = track(map(DiffResults.value, results), D, tp)
            cache = (results, df, index_bound(x, out), index_bound(y, out))
            record!(tp, SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end
        @eval function Base.$(g)(f::ForwardOptimize{F}, x::TrackedArray{X,D}, y::$A) where {F,X,D}
            result = DiffResults.GradientResult(SVector(zero(X)))
            df = (vx, vy) -> let vy=vy
                ForwardDiff.gradient!(result, s -> f.f(s[1], vy), SVector(vx))
            end
            results = $(g)(df, value(x), value(y))
            tp = tape(x)
            out = track(map(DiffResults.value, results), D, tp)
            cache = (results, df, index_bound(x, out), index_bound(y, out))
            record!(tp, SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end
        @eval function Base.$(g)(f::ForwardOptimize{F}, x::$A, y::TrackedArray{Y,D}) where {F,Y,D}
            result = DiffResults.GradientResult(SVector(zero(Y)))
            df = (vx, vy) -> let vx=vx
                ForwardDiff.gradient!(result, s -> f.f(vx, s[1]), SVector(vy))
            end
            results = $(g)(df, value(x), value(y))
            tp = tape(y)
            out = track(map(DiffResults.value, results), D, tp)
            cache = (results, df, index_bound(x, out), index_bound(y, out))
            record!(tp, SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end
    end

    for TX in (:TrackedArray, :TrackedReal), TY in (:TrackedArray, :TrackedReal)
        TX == :TrackedReal && TY == :TrackedReal && continue
        @eval function Base.$(g)(f::ForwardOptimize{F}, x::$(TX){X,D}, y::$(TY){Y,D}) where {F,X,Y,D}
            result = DiffResults.GradientResult(SVector(zero(D), zero(D)))
            df = (vx, vy) -> ForwardDiff.gradient!(result, s -> f.f(s[1], s[2]), SVector(vx, vy))
            results = $(g)(df, value(x), value(y))
            tp = tape(x, y)
            out = track(map(DiffResults.value, results), D, tp)
            cache = (results, df, index_bound(x, out), index_bound(y, out))
            record!(tp, SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end
    end
end

################
# forward pass #
################

for (g!, g) in ((:map!, :map),)
    @eval begin
        @noinline function special_forward_exec!(instruction::SpecialInstruction{typeof($g)})
            input, output = instruction.input, instruction.output
            results, df, _, _ = instruction.cache
            if istracked(input)
                ($g!)(df, results, value(input))
            else
                a, b = input
                pull_value!(a)
                pull_value!(b)
                ($g!)(df, results, value(a), value(b))
            end
            output_value = value(output)
            for i in eachindex(output_value)
                output_value[i] = DiffResults.value(results[i])
            end
            return nothing
        end
    end
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
