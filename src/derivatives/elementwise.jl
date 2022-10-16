###############################
# SkipOptimized map/broadcast #
###############################

# dispatch #
#----------#

for g in (:map, :broadcast), f in SKIPPED_UNARY_SCALAR_FUNCS
    @eval @inline Base.$(g)(f::typeof($f), t::TrackedArray) = $(g)(SkipOptimize(f), t)
end

for g in (:map, :broadcast), f in SKIPPED_BINARY_SCALAR_FUNCS
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

for g in (:map, :broadcast)
    @eval @inline Base.$(g)(f::SkipOptimize{F}, t::TrackedArray) where {F} = $(g)(f.f, value(t))
end

for g in (:map, :broadcast)
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

####################################
# ForwardOptimized map!/broadcast! #
####################################

# dispatch #
#----------#

for g! in (:map!, :broadcast!), (M, f, arity) in DiffRules.diffrules(; filter_modules=nothing)
    if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), f))
        @warn "$M.$f is not available and hence rule for it can not be defined"
        continue  # Skip rules for methods not defined in the current scope
    end
    (M, f) in SKIPPED_DIFFRULES && continue
    if arity == 1
        @eval @inline Base.$(g!)(f::typeof($M.$f), out::TrackedArray, t::TrackedArray) = $(g!)(ForwardOptimize(f), out, t)
    elseif arity == 2
        @eval begin
            @inline Base.$(g!)(f::typeof($M.$f), out::TrackedArray, x::TrackedArray, y::TrackedArray) = $(g!)(ForwardOptimize(f), out, x, y)
            @inline Base.$(g!)(f::typeof($M.$f), out::TrackedArray, x::TrackedArray, y::TrackedReal) = $(g!)(ForwardOptimize(f), out, x, y)
            @inline Base.$(g!)(f::typeof($M.$f), out::TrackedArray, x::TrackedReal, y::TrackedArray) = $(g!)(ForwardOptimize(f), out, x, y)
        end
        for A in ARRAY_TYPES
            @eval begin
                @inline Base.$(g!)(f::typeof($M.$f), out::TrackedArray, x::$A, y::TrackedArray) = $(g!)(ForwardOptimize(f), out, x, y)
                @inline Base.$(g!)(f::typeof($M.$f), out::TrackedArray, x::TrackedArray, y::$A) = $(g!)(ForwardOptimize(f), out, x, y)
                @inline Base.$(g!)(f::typeof($M.$f), out::TrackedArray, x::$A, y::TrackedReal) = $(g!)(ForwardOptimize(f), out, x, y)
                @inline Base.$(g!)(f::typeof($M.$f), out::TrackedArray, x::TrackedReal, y::$A) = $(g!)(ForwardOptimize(f), out, x, y)
            end
        end
        for R in REAL_TYPES
            @eval begin
                @inline Base.$(g!)(f::typeof($M.$f), out::TrackedArray, x::$R, y::TrackedArray) = $(g!)(ForwardOptimize(f), out, x, y)
                @inline Base.$(g!)(f::typeof($M.$f), out::TrackedArray, x::TrackedArray, y::$R) = $(g!)(ForwardOptimize(f), out, x, y)
            end
        end
    end
end

# record #
#--------#

for (g!, g) in ((:map!, :map), (:broadcast!, :broadcast))
    @eval function Base.$(g!)(f::ForwardOptimize{F}, out::TrackedArray{S}, x::TrackedArray{X}) where {F,S,X}
        result = DiffResults.DiffResult(zero(S), zero(S))
        df = v -> ForwardDiff.derivative!(result, f.f, v)
        results = $(g)(df, value(x))
        map!(DiffResult.value, value(out), results)
        cache = (results, df, index_bound(x, out), nothing)
        record!(tape(x), SpecialInstruction, $(g), x, out, cache)
        return out
    end
    for TX in (:TrackedArray, :TrackedReal), TY in (:TrackedArray, :TrackedReal)
        (TX == TY == :TrackedReal) && continue
        @eval function Base.$(g!)(f::ForwardOptimize{F}, out::TrackedArray{S}, x::$(TX){X}, y::$(TY){Y}) where {F,S,X,Y}
            result = DiffResults.GradientResult(SVector(zero(S), zero(S)))
            df = (vx, vy) -> ForwardDiff.gradient!(result, s -> f.f(s[1], s[2]), SVector(vx, vy))
            results = $(g)(df, value(x), value(y))
            map!(DiffResult.value, value(out), results)
            cache = (results, df, index_bound(x, out), index_bound(y, out))
            record!(tape(x, y), SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end
    end
    for A in ARRAY_TYPES
        @eval function Base.$(g!)(f::ForwardOptimize{F}, out::TrackedArray, x::TrackedReal{X,D}, y::$A) where {F,X,D}
            result = DiffResults.DiffResult(zero(X), zero(D))
            df = let result=result
                (vx, vy) -> let vy=vy
                    ForwardDiff.derivative!(result, s -> f.f(s, vy), vx)
                end
            end
            results = $(g)(df, value(x), value(y))
            map!(DiffResult.value, value(out), results)
            cache = (results, df, index_bound(x, out), index_bound(y, out))
            record!(tape(x), SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end
        @eval function Base.$(g!)(f::ForwardOptimize{F}, out::TrackedArray, x::$A, y::TrackedReal{Y,D}) where {F,Y,D}
            result = DiffResults.DiffResult(zero(Y), zero(D))
            df = let result=result
                (vx, vy) -> let vx=vx
                    ForwardDiff.derivative!(result, s -> f.f(vx, s), vy)
                end
            end
            results = $(g)(df, value(x), value(y))
            map!(DiffResult.value, value(out), results)
            cache = (results, df, index_bound(x, out), index_bound(y, out))
            record!(tape(y), SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end
        @eval function Base.$(g!)(f::ForwardOptimize{F}, out::TrackedArray, x::TrackedArray{X}, y::$A) where {F,X}
            result = DiffResults.GradientResult(SVector(zero(X)))
            df = (vx, vy) -> let vy=vy
                ForwardDiff.gradient!(result, s -> f.f(s[1], vy), SVector(vx))
            end
            results = $(g)(df, value(x), value(y))
            map!(DiffResult.value, value(out), results)
            cache = (results, df, index_bound(x, out), index_bound(y, out))
            record!(tape(x), SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end
        @eval function Base.$(g!)(f::ForwardOptimize{F}, out::TrackedArray, x::$A, y::TrackedArray{Y}) where {F,Y}
            result = DiffResults.GradientResult(SVector(zero(Y)))
            df = let vx=vx
                (vx, vy) -> ForwardDiff.gradient!(result, s -> f.f(vx, s[1]), SVector(vy))
            end
            results = $(g)(df, value(x), value(y))
            map!(DiffResult.value, value(out), results)
            cache = (results, df, index_bound(x, out), index_bound(y, out))
            record!(tape(y), SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end
    end
end

for R in REAL_TYPES
    @eval begin
        @inline Base.broadcast!(f::ForwardOptimize{F}, out::TrackedArray, x::TrackedArray, y::$R) where {F} = broadcast!(ForwardOptimize(t -> f.f(t, y)), out, x)
        @inline Base.broadcast!(f::ForwardOptimize{F}, out::TrackedArray, x::$R, y::TrackedArray) where {F} = broadcast!(ForwardOptimize(t -> f.f(x, t)), out, y)
    end
end

##################################
# ForwardOptimized map/broadcast #
##################################

# dispatch #
#----------#

for g in (:map, :broadcast), (M, f, arity) in DiffRules.diffrules(; filter_modules=nothing)
    if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), f))
        @warn "$M.$f is not available and hence rule for it can not be defined"
        continue  # Skip rules for methods not defined in the current scope
    end
    if arity == 1
        @eval @inline Base.$(g)(f::typeof($M.$f), t::TrackedArray) = $(g)(ForwardOptimize(f), t)
    elseif arity == 2
        (M, f) in SKIPPED_DIFFRULES && continue
        # skip these definitions if `f` is one of the functions
        # that will get a manually defined broadcast definition
        # later (see "built-in infix operations" below)
        g == :broadcast && in(f, (:+, :-, :*, :/, :\, :^)) && continue
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

for g in (:map, :broadcast)
    @eval function Base.$(g)(f::ForwardOptimize{F}, x::TrackedArray{X,D}) where {F,X,D}
        T = promote_type(X, D)
        result = DiffResults.DiffResult(zero(T), zero(T))
        df = v -> ForwardDiff.derivative!(result, f.f, v)
        results = $(g)(df, value(x))
        tp = tape(x)
        out = track(DiffResults.value.(results), D, tp)
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
            out = track(DiffResults.value.(results), D, tp)
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
            out = track(DiffResults.value.(results), D, tp)
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
            out = track(DiffResults.value.(results), D, tp)
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
            out = track(DiffResults.value.(results), D, tp)
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
            out = track(DiffResults.value.(results), D, tp)
            cache = (results, df, index_bound(x, out), index_bound(y, out))
            record!(tp, SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end
    end
end

for R in REAL_TYPES
    @eval begin
        @inline Base.broadcast(f::ForwardOptimize{F}, x::TrackedArray{X,D}, y::$R) where {F,X,D} = broadcast(ForwardOptimize(t -> f.f(t, y)), x)
        @inline Base.broadcast(f::ForwardOptimize{F}, x::$R, y::TrackedArray{Y,D}) where {F,Y,D} = broadcast(ForwardOptimize(t -> f.f(x, t)), y)
    end
end

################
# forward pass #
################

for (g!, g) in ((:map!, :map), (:broadcast!, :broadcast))
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
        diffresult_increment_deriv!(input, output_deriv, results, 1)
    else
        a, b = input
        p = 0
        if istracked(a)
            p += 1
            diffresult_increment_deriv!(a, output_deriv, results, p)
        end
        if istracked(b)
            p += 1
            diffresult_increment_deriv!(b, output_deriv, results, p)
        end
    end
    unseed!(output)
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(broadcast)})
    input = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    results, _, a_bound, b_bound = instruction.cache
    if istracked(input)
        if size(input) == size(output_deriv)
            diffresult_increment_deriv!(input, output_deriv, results, 1)
        else
            diffresult_increment_deriv!(input, output_deriv, results, 1, a_bound)
        end
    else
        a, b = input
        p = 0
        if size(a) == size(b)
            if istracked(a)
                p += 1
                diffresult_increment_deriv!(a, output_deriv, results, p)
            end
            if istracked(b)
                p += 1
                diffresult_increment_deriv!(b, output_deriv, results, p)
            end
        else
            if istracked(a)
                p += 1
                diffresult_increment_deriv!(a, output_deriv, results, p, a_bound)
            end
            if istracked(b)
                p += 1
                diffresult_increment_deriv!(b, output_deriv, results, p, b_bound)
            end
        end
    end
    unseed!(output)
    return nothing
end

#############################
# built-in infix operations #
#############################

const TrackedType = Union{TrackedArray,TrackedReal}

# dispatch #
#----------#

for (F, broadcast_f) in ((typeof(+), :broadcast_plus),
                         (typeof(-), :broadcast_minus),
                         (typeof(*), :broadcast_mul),
                         (typeof(/), :broadcast_rdiv),
                         (typeof(\), :broadcast_ldiv),
                         (typeof(^), :broadcast_pow))
    @eval begin
        @inline Base.broadcast(::$F, x::TrackedArray{X,D}, y::TrackedArray{Y,D}) where {X,Y,D} = $(broadcast_f)(x, y, D)
        @inline Base.broadcast(::$F, x::TrackedReal{X,D}, y::TrackedArray{Y,D}) where {X,Y,D} = $(broadcast_f)(x, y, D)
        @inline Base.broadcast(::$F, x::TrackedArray{X,D}, y::TrackedReal{Y,D}) where {X,Y,D} = $(broadcast_f)(x, y, D)
    end
    for A in ARRAY_TYPES
        @eval begin
            @inline Base.broadcast(::$F, x::TrackedArray{X,D}, y::$A{<:Real}) where {X,D} = $(broadcast_f)(x, y, D)
            @inline Base.broadcast(::$F, x::$A{<:Real}, y::TrackedArray{Y,D}) where {Y,D} = $(broadcast_f)(x, y, D)
            @inline Base.broadcast(::$F, x::TrackedReal{X,D}, y::$A{<:Real}) where {X,D} = $(broadcast_f)(x, y, D)
            @inline Base.broadcast(::$F, x::$A{<:Real}, y::TrackedReal{Y,D}) where {Y,D} = $(broadcast_f)(x, y, D)
        end
    end
    for R in REAL_TYPES
        @eval begin
            @inline Base.broadcast(::$F, x::TrackedArray{X,D}, y::$R) where {X,D} = $(broadcast_f)(x, y, D)
            @inline Base.broadcast(::$F, x::$R, y::TrackedArray{Y,D}) where {Y,D} = $(broadcast_f)(x, y, D)
        end
    end
end

# .+ #
#----#

function broadcast_plus(x, y, ::Type{D}) where D
    tp = tape(x, y)
    out = track(value(x) .+ value(y), D, tp)
    cache = (index_bound(x, out), index_bound(y, out))
    record!(tp, SpecialInstruction, (broadcast, +), (x, y), out, cache)
    return out
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{Tuple{typeof(broadcast),typeof(+)}})
    a, b = instruction.input
    output = instruction.output
    pull_value!(a)
    pull_value!(b)
    broadcast!(+, value(output), value(a), value(b))
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{Tuple{typeof(broadcast),typeof(+)}})
    a, b = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    a_bound, b_bound = instruction.cache
    istracked(a) && broadcast_increment_deriv!(a, output_deriv, a_bound)
    istracked(b) && broadcast_increment_deriv!(b, output_deriv, b_bound)
    unseed!(output)
    return nothing
end

# .- #
#----#

function broadcast_minus(x, y, ::Type{D}) where D
    tp = tape(x, y)
    out = track(value(x) .- value(y), D, tp)
    cache = (index_bound(x, out), index_bound(y, out))
    record!(tp, SpecialInstruction, (broadcast, -), (x, y), out, cache)
    return out
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{Tuple{typeof(broadcast),typeof(-)}})
    a, b = instruction.input
    output = instruction.output
    pull_value!(a)
    pull_value!(b)
    broadcast!(-, value(output), value(a), value(b))
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{Tuple{typeof(broadcast),typeof(-)}})
    a, b = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    a_bound, b_bound = instruction.cache
    istracked(a) && broadcast_increment_deriv!(a, output_deriv, a_bound)
    istracked(b) && broadcast_decrement_deriv!(b, output_deriv, b_bound)
    unseed!(output)
    return nothing
end

# .* #
#----#

function broadcast_mul(x, y, ::Type{D}) where D
    tp = tape(x, y)
    out = track(value(x) .* value(y), D, tp)
    cache = (index_bound(x, out), index_bound(y, out))
    record!(tp, SpecialInstruction, (broadcast, *), (x, y), out, cache)
    return out
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{Tuple{typeof(broadcast),typeof(*)}})
    a, b = instruction.input
    output = instruction.output
    pull_value!(a)
    pull_value!(b)
    broadcast!(*, value(output), value(a), value(b))
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{Tuple{typeof(broadcast),typeof(*)}})
    a, b = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    a_bound, b_bound = instruction.cache
    istracked(a) && broadcast_increment_deriv!(a, output_deriv, value(b), a_bound, b_bound)
    istracked(b) && broadcast_increment_deriv!(b, output_deriv, value(a), b_bound, a_bound)
    unseed!(output)
    return nothing
end

# ./ #
#----#

numer_partials(d::Real) = Ref(inv(d))
numer_partials(d::AbstractArray) = broadcast(inv, d)
numer_partials!(out::Ref, d) = (out[] = inv(d); nothing)
numer_partials!(out::AbstractArray, d) = (broadcast!(inv, out, d); nothing)

denom_partials_kernel(n::Real, d::Real) =  -(n / (d * d))
denom_partials(n::Real, d::Real) = Ref(denom_partials_kernel(n, d))
denom_partials(n, d) = broadcast(denom_partials_kernel, n, d)
denom_partials!(out::Ref, n, d) = (out[] = denom_partials_kernel(n, d); nothing)
denom_partials!(out::AbstractArray, n, d) = (broadcast!(denom_partials_kernel, out, n, d); nothing)

rdiv_cache(x, y) = (numer_partials(value(y)), denom_partials(value(x), value(y)))

function broadcast_rdiv(x, y, ::Type{D}) where D
    tp = tape(x, y)
    out = track(value(x) ./ value(y), D, tp)
    n_partials, d_partials = rdiv_cache(x, y)
    cache = (n_partials, d_partials,
             index_bound(x, out), index_bound(y, out),
             index_bound(n_partials, out), index_bound(d_partials, out))
    record!(tp, SpecialInstruction, (broadcast, /), (x, y), out, cache)
    return out
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{Tuple{typeof(broadcast),typeof(/)}})
    a, b = instruction.input
    a_value, b_value = value(a), value(b)
    n_partials, d_partials = instruction.cache
    output = instruction.output
    pull_value!(a)
    pull_value!(b)
    broadcast!(/, value(output), a_value, b_value)
    istracked(a) && numer_partials!(n_partials, b_value)
    istracked(b) && denom_partials!(d_partials, a_value, b_value)
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{Tuple{typeof(broadcast),typeof(/)}})
    a, b = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    n_partials, d_partials, a_bound, b_bound,
    n_partials_bound, d_partials_bound = instruction.cache
    istracked(a) && broadcast_increment_deriv!(a, output_deriv, n_partials, a_bound, n_partials_bound)
    istracked(b) && broadcast_increment_deriv!(b, output_deriv, d_partials, b_bound, d_partials_bound)
    unseed!(output)
    return nothing
end

# .\ #
#----#

function broadcast_ldiv(x, y, ::Type{D}) where D
    tp = tape(x, y)
    out = track(value(x) .\ value(y), D, tp)
    n_partials, d_partials = rdiv_cache(y, x)
    cache = (n_partials, d_partials,
             index_bound(x, out), index_bound(y, out),
             index_bound(n_partials, out), index_bound(d_partials, out))
    record!(tp, SpecialInstruction, (broadcast, \), (x, y), out, cache)
    return out
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{Tuple{typeof(broadcast),typeof(\)}})
    a, b = instruction.input
    a_value, b_value = value(a), value(b)
    n_partials, d_partials = instruction.cache
    output = instruction.output
    pull_value!(a)
    pull_value!(b)
    broadcast!(\, value(output), a_value, b_value)
    istracked(b) && numer_partials!(n_partials, a_value)
    istracked(a) && denom_partials!(d_partials, b_value, a_value)
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{Tuple{typeof(broadcast),typeof(\)}})
    a, b = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    n_partials, d_partials, a_bound, b_bound,
    n_partials_bound, d_partials_bound = instruction.cache
    istracked(a) && broadcast_increment_deriv!(a, output_deriv, d_partials, a_bound, d_partials_bound)
    istracked(b) && broadcast_increment_deriv!(b, output_deriv, n_partials, b_bound, n_partials_bound)
    unseed!(output)
    return nothing
end

# .^ #
#----#

base_partials_kernel(b::Real, e::Real) = e * b^(e - 1)
base_partials(b::Real, e::Real) = Ref(base_partials_kernel(b, e))
base_partials(b, e) = broadcast(base_partials_kernel, b, e)
base_partials!(out::Ref, b, e) = (out[] = base_partials_kernel(b, e); nothing)
base_partials!(out::AbstractArray, b, e) = (broadcast!(base_partials_kernel, out, b, e); nothing)

exp_partials_kernel(b::Real, e::Real) = log(b) * b^e
exp_partials(b::Real, e::Real) = Ref(exp_partials_kernel(b, e))
exp_partials(b, e) = broadcast(exp_partials_kernel, b, e)
exp_partials!(out::Ref, b, e) = (out[] = exp_partials_kernel(b, e); nothing)
exp_partials!(out::AbstractArray, b, e) = (broadcast!(exp_partials_kernel, out, b, e); nothing)

function pow_cache(x, y)
    pow_x = istracked(x) ? base_partials(value(x), value(y)) : value(x)
    pow_y = istracked(y) ? exp_partials(value(x), value(y)) : value(y)
    return (pow_x, pow_y)
end

function broadcast_pow(x, y, ::Type{D}) where D
    tp = tape(x, y)
    out = track(value(x) .^ value(y), D, tp)
    bs_partials, ex_partials = pow_cache(x, y)
    cache = (bs_partials, ex_partials,
             index_bound(x, out), index_bound(y, out),
             index_bound(bs_partials, out), index_bound(ex_partials, out))
    record!(tp, SpecialInstruction, (broadcast, ^), (x, y), out, cache)
    return out
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{Tuple{typeof(broadcast),typeof(^)}})
    a, b = instruction.input
    a_value, b_value = value(a), value(b)
    bs_partials, ex_partials = instruction.cache
    output = instruction.output
    pull_value!(a)
    pull_value!(b)
    broadcast!(^, value(output), a_value, b_value)
    istracked(a) && base_partials!(bs_partials, a_value, b_value)
    istracked(b) && exp_partials!(ex_partials, a_value, b_value)
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{Tuple{typeof(broadcast),typeof(^)}})
    a, b = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    bs_partials, ex_partials, a_bound, b_bound,
    bs_partials_bound, ex_partials_bound = instruction.cache
    istracked(a) && broadcast_increment_deriv!(a, output_deriv, bs_partials, a_bound, bs_partials_bound)
    istracked(b) && broadcast_increment_deriv!(b, output_deriv, ex_partials, b_bound, ex_partials_bound)
    unseed!(output)
    return nothing
end
