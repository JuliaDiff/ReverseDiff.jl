function dual_values(duals)
    values = similar(duals, ForwardDiff.valtype(eltype(duals)))
    return map!(ForwardDiff.value, values, duals)
end

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
    @eval @inline Base.$(g){F}(f::SkipOptimize{F}, t::TrackedArray) = $(g)(f.f, value(t))
end

for g in (:map, :broadcast)
    @eval begin
        @inline Base.$(g){F}(f::SkipOptimize{F}, x::TrackedArray, y::TrackedArray) = $(g)(f.f, value(x), value(y))
        @inline Base.$(g){F}(f::SkipOptimize{F}, x::TrackedArray, y::TrackedReal) = $(g)(f.f, value(x), value(y))
        @inline Base.$(g){F}(f::SkipOptimize{F}, x::TrackedReal, y::TrackedArray) = $(g)(f.f, value(x), value(y))
    end
    for A in ARRAY_TYPES
        @eval begin
            @inline Base.$(g){F}(f::SkipOptimize{F}, x::$A, y::TrackedArray) = $(g)(f.f, value(x), value(y))
            @inline Base.$(g){F}(f::SkipOptimize{F}, x::TrackedArray, y::$A) = $(g)(f.f, value(x), value(y))
            @inline Base.$(g){F}(f::SkipOptimize{F}, x::$A, y::TrackedReal) = $(g)(f.f, value(x), value(y))
            @inline Base.$(g){F}(f::SkipOptimize{F}, x::TrackedReal, y::$A) = $(g)(f.f, value(x), value(y))
        end
    end
    for R in REAL_TYPES
        @eval begin
            @inline Base.$(g){F}(f::SkipOptimize{F}, x::$R, y::TrackedArray) = $(g)(f.f, value(x), value(y))
            @inline Base.$(g){F}(f::SkipOptimize{F}, x::TrackedArray, y::$R) = $(g)(f.f, value(x), value(y))
        end
    end
end

####################################
# ForwardOptimized map!/broadcast! #
####################################

# dispatch #
#----------#

for g! in (:map!, :broadcast!), f in FORWARD_UNARY_SCALAR_FUNCS
    @eval @inline Base.$(g!)(f::typeof($f), out::TrackedArray, t::TrackedArray) = $(g!)(ForwardOptimize(f), out, t)
end

for g! in (:map!, :broadcast!), f in FORWARD_BINARY_SCALAR_FUNCS
    @eval begin
        @inline Base.$(g!)(f::typeof($f), out::TrackedArray, x::TrackedArray, y::TrackedArray) = $(g!)(ForwardOptimize(f), out, x, y)
        @inline Base.$(g!)(f::typeof($f), out::TrackedArray, x::TrackedArray, y::TrackedReal) = $(g!)(ForwardOptimize(f), out, x, y)
        @inline Base.$(g!)(f::typeof($f), out::TrackedArray, x::TrackedReal, y::TrackedArray) = $(g!)(ForwardOptimize(f), out, x, y)
    end
    for A in ARRAY_TYPES
        @eval begin
            @inline Base.$(g!)(f::typeof($f), out::TrackedArray, x::$A, y::TrackedArray) = $(g!)(ForwardOptimize(f), out, x, y)
            @inline Base.$(g!)(f::typeof($f), out::TrackedArray, x::TrackedArray, y::$A) = $(g!)(ForwardOptimize(f), out, x, y)
            @inline Base.$(g!)(f::typeof($f), out::TrackedArray, x::$A, y::TrackedReal) = $(g!)(ForwardOptimize(f), out, x, y)
            @inline Base.$(g!)(f::typeof($f), out::TrackedArray, x::TrackedReal, y::$A) = $(g!)(ForwardOptimize(f), out, x, y)
        end
    end
    for R in REAL_TYPES
        @eval begin
            @inline Base.$(g!)(f::typeof($f), out::TrackedArray, x::$R, y::TrackedArray) = $(g!)(ForwardOptimize(f), out, x, y)
            @inline Base.$(g!)(f::typeof($f), out::TrackedArray, x::TrackedArray, y::$R) = $(g!)(ForwardOptimize(f), out, x, y)
        end
    end
end

# record #
#--------#

for (g!, g) in ((:map!, :map), (:broadcast!, :broadcast))
    @eval function Base.$(g!){F,X}(f::ForwardOptimize{F}, out::TrackedArray, x::TrackedArray{X})
        fdual = v -> f.f(Dual(v, one(X)))
        duals = $(g)(fdual, value(x))
        copy!(value(out), dual_values(duals))
        cache = (duals, fdual, index_bound(x, out), nothing)
        record!(tape(x), SpecialInstruction, $(g), x, out, cache)
        return out
    end

    for TX in (:TrackedArray, :TrackedReal), TY in (:TrackedArray, :TrackedReal)
        (TX == TY == :TrackedReal) && continue
        @eval function Base.$(g!){F,X,Y}(f::ForwardOptimize{F}, out::TrackedArray, x::$(TX){X}, y::$(TY){Y})
            fdual = (vx, vy) -> f.f(Dual(vx, one(X), zero(X)), Dual(vy, zero(Y), one(Y)))
            duals = $(g)(fdual, value(x), value(y))
            copy!(value(out), dual_values(duals))
            cache = (duals, fdual, index_bound(x, out), index_bound(y, out))
            record!(tape(x, y), SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end
    end

    for A in ARRAY_TYPES, T in (:TrackedArray, :TrackedReal)
        @eval function Base.$(g!){F,X}(f::ForwardOptimize{F}, out::TrackedArray, x::$(T){X}, y::$A)
            fdual = (vx, vy) -> f.f(Dual(vx, one(X), zero(X)), Dual(vy, zero(vy), one(vy)))
            duals = $(g)(fdual, value(x), value(y))
            copy!(value(out), dual_values(duals))
            cache = (duals, fdual, index_bound(x, out), index_bound(y, out))
            record!(tape(x), SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end

        @eval function Base.$(g!){F,Y}(f::ForwardOptimize{F}, out::TrackedArray, x::$A, y::$(T){Y})
            fdual = (vx, vy) -> f.f(Dual(vx, one(vx), zero(vx)), Dual(vy, zero(Y), one(Y)))
            duals = $(g)(fdual, value(x), value(y))
            copy!(value(out), dual_values(duals))
            cache = (duals, fdual, index_bound(x, out), index_bound(y, out))
            record!(tape(x), SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end
    end
end

for R in REAL_TYPES
    @eval begin
        @inline Base.broadcast!{F}(f::ForwardOptimize{F}, out::TrackedArray, x::TrackedArray, y::$R) = broadcast!(ForwardOptimize(t -> f.f(t, y)), out, x)
        @inline Base.broadcast!{F}(f::ForwardOptimize{F}, out::TrackedArray, x::$R, y::TrackedArray) = broadcast!(ForwardOptimize(t -> f.f(x, t)), out, y)
    end
end

##################################
# ForwardOptimized map/broadcast #
##################################

# dispatch #
#----------#

for g in (:map, :broadcast), f in FORWARD_UNARY_SCALAR_FUNCS
    @eval @inline Base.$(g)(f::typeof($f), t::TrackedArray) = $(g)(ForwardOptimize(f), t)
end

for g in (:map, :broadcast), f in FORWARD_BINARY_SCALAR_FUNCS
    @eval begin
        @inline Base.$(g)(f::typeof($f), x::TrackedArray, y::TrackedArray) = $(g)(ForwardOptimize(f), x, y)
        @inline Base.$(g)(f::typeof($f), x::TrackedArray, y::TrackedReal) = $(g)(ForwardOptimize(f), x, y)
        @inline Base.$(g)(f::typeof($f), x::TrackedReal, y::TrackedArray) = $(g)(ForwardOptimize(f), x, y)
    end
    for A in ARRAY_TYPES
        @eval begin
            @inline Base.$(g)(f::typeof($f), x::$A, y::TrackedArray) = $(g)(ForwardOptimize(f), x, y)
            @inline Base.$(g)(f::typeof($f), x::TrackedArray, y::$A) = $(g)(ForwardOptimize(f), x, y)
            @inline Base.$(g)(f::typeof($f), x::$A, y::TrackedReal) = $(g)(ForwardOptimize(f), x, y)
            @inline Base.$(g)(f::typeof($f), x::TrackedReal, y::$A) = $(g)(ForwardOptimize(f), x, y)
        end
    end
    for R in REAL_TYPES
        @eval begin
            @inline Base.$(g)(f::typeof($f), x::$R, y::TrackedArray) = $(g)(ForwardOptimize(f), x, y)
            @inline Base.$(g)(f::typeof($f), x::TrackedArray, y::$R) = $(g)(ForwardOptimize(f), x, y)
        end
    end
end


# record #
#--------#

for g in (:map, :broadcast)
    @eval function Base.$(g){F,X,D}(f::ForwardOptimize{F}, x::TrackedArray{X,D})
        fdual = v -> f.f(Dual(v, one(X)))
        duals = $(g)(fdual, value(x))
        tp = tape(x)
        out = track(dual_values(duals), D, tp)
        cache = (duals, fdual, index_bound(x, out), nothing)
        record!(tp, SpecialInstruction, $(g), x, out, cache)
        return out
    end

    for A in ARRAY_TYPES, T in (:TrackedArray, :TrackedReal)
        @eval function Base.$(g){F,X,D}(f::ForwardOptimize{F}, x::$(T){X,D}, y::$A)
            fdual = (vx, vy) -> f.f(Dual(vx, one(X), zero(X)), Dual(vy, zero(vy), one(vy)))
            duals = $(g)(fdual, value(x), value(y))
            tp = tape(x)
            out = track(dual_values(duals), D, tp)
            cache = (duals, fdual, index_bound(x, out), index_bound(y, out))
            record!(tp, SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end

        @eval function Base.$(g){F,Y,D}(f::ForwardOptimize{F}, x::$A, y::$(T){Y,D})
            fdual = (vx, vy) -> f.f(Dual(vx, one(vx), zero(vx)), Dual(vy, zero(Y), one(Y)))
            duals = $(g)(fdual, value(x), value(y))
            tp = tape(y)
            out = track(dual_values(duals), D, tp)
            cache = (duals, fdual, index_bound(x, out), index_bound(y, out))
            record!(tp, SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end
    end

    for TX in (:TrackedArray, :TrackedReal), TY in (:TrackedArray, :TrackedReal)
        TX == :TrackedReal && TY == :TrackedReal && continue
        @eval function Base.$(g){F,X,Y,D}(f::ForwardOptimize{F}, x::$(TX){X,D}, y::$(TY){Y,D})
            fdual = (vx, vy) -> f.f(Dual(vx, one(X), zero(X)), Dual(vy, zero(Y), one(Y)))
            duals = $(g)(fdual, value(x), value(y))
            tp = tape(x, y)
            out = track(dual_values(duals), D, tp)
            cache = (duals, fdual, index_bound(x, out), index_bound(y, out))
            record!(tp, SpecialInstruction, $(g), (x, y), out, cache)
            return out
        end
    end
end

for R in REAL_TYPES
    @eval begin
        @inline Base.broadcast{F,X,D}(f::ForwardOptimize{F}, x::TrackedArray{X,D}, y::$R) = broadcast(ForwardOptimize(t -> f.f(t, y)), x)
        @inline Base.broadcast{F,Y,D}(f::ForwardOptimize{F}, x::$R, y::TrackedArray{Y,D}) = broadcast(ForwardOptimize(t -> f.f(x, t)), y)
    end
end

################
# forward pass #
################

for (g!, g) in ((:map!, :map), (:broadcast!, :broadcast))
    @eval begin
        @noinline function special_forward_exec!(instruction::SpecialInstruction{typeof($g)})
            input, output = instruction.input, instruction.output
            duals, fdual, _, _ = instruction.cache
            if istracked(input)
                ($g!)(fdual, duals, value(input))
            else
                a, b = input
                pull_value!(a)
                pull_value!(b)
                ($g!)(fdual, duals, value(a), value(b))
            end
            output_value = value(output)
            for i in eachindex(output_value)
                output_value[i] = ForwardDiff.value(duals[i])
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
    duals = first(instruction.cache)
    if istracked(input)
        duals_increment_deriv!(input, output_deriv, duals, 1)
    else
        a, b = input
        istracked(a) && duals_increment_deriv!(a, output_deriv, duals, 1)
        istracked(b) && duals_increment_deriv!(b, output_deriv, duals, 2)
    end
    unseed!(output)
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(broadcast)})
    input = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    duals, _, a_bound, b_bound = instruction.cache
    if istracked(input)
        if size(input) == size(output_deriv)
            duals_increment_deriv!(input, output_deriv, duals, 1)
        else
            duals_increment_deriv!(input, output_deriv, duals, 1, a_bound)
        end
    else
        a, b = input
        if size(a) == size(b)
            istracked(a) && duals_increment_deriv!(a, output_deriv, duals, 1)
            istracked(b) && duals_increment_deriv!(b, output_deriv, duals, 2)
        else
            istracked(a) && duals_increment_deriv!(a, output_deriv, duals, 1, a_bound)
            istracked(b) && duals_increment_deriv!(b, output_deriv, duals, 2, b_bound)
        end
    end
    unseed!(output)
    return nothing
end

#############################
# built-in infix operations #
#############################

typealias TrackedType Union{TrackedArray,TrackedReal}

# dispatch #
#----------#

if VERSION >= v"0.6.0-dev.1614"
    for (F, broadcast_f) in ((typeof(+), :broadcast_plus),
                             (typeof(-), :broadcast_minus),
                             (typeof(*), :broadcast_mul),
                             (typeof(/), :broadcast_rdiv),
                             (typeof(\), :broadcast_ldiv),
                             (typeof(^), :broadcast_pow))
        @eval begin
            @inline Base.broadcast{X,Y,D}(::$F, x::TrackedArray{X,D}, y::TrackedArray{Y,D}) = $(broadcast_f)(x, y, D)
            @inline Base.broadcast{X,Y,D}(::$F, x::TrackedReal{X,D}, y::TrackedArray{Y,D}) = $(broadcast_f)(x, y, D)
            @inline Base.broadcast{X,Y,D}(::$F, x::TrackedArray{X,D}, y::TrackedReal{Y,D}) = $(broadcast_f)(x, y, D)
        end
        for A in ARRAY_TYPES
            @eval begin
                @inline Base.broadcast{X,D}(::$F, x::TrackedArray{X,D}, y::$A) = $(broadcast_f)(x, y, D)
                @inline Base.broadcast{Y,D}(::$F, x::$A, y::TrackedArray{Y,D}) = $(broadcast_f)(x, y, D)
                @inline Base.broadcast{X,D}(::$F, x::TrackedReal{X,D}, y::$A) = $(broadcast_f)(x, y, D)
                @inline Base.broadcast{Y,D}(::$F, x::$A, y::TrackedReal{Y,D}) = $(broadcast_f)(x, y, D)
            end
        end
        for R in REAL_TYPES
            @eval begin
                @inline Base.broadcast{X,D}(::$F, x::TrackedArray{X,D}, y::$R) = $(broadcast_f)(x, y, D)
                @inline Base.broadcast{Y,D}(::$F, x::$R, y::TrackedArray{Y,D}) = $(broadcast_f)(x, y, D)
            end
        end
    end
else
    for (f, broadcast_f) in ((:.+, :broadcast_plus),
                             (:.-, :broadcast_minus),
                             (:.*, :broadcast_mul),
                             (:./, :broadcast_rdiv),
                             (:.\, :broadcast_ldiv),
                             (:.^, :broadcast_pow))
        @eval begin
            @inline Base.$(f){X,Y,D}(x::TrackedArray{X,D}, y::TrackedArray{Y,D}) = $(broadcast_f)(x, y, D)
            @inline Base.$(f){X,Y,D}(x::TrackedReal{X,D}, y::TrackedArray{Y,D}) = $(broadcast_f)(x, y, D)
            @inline Base.$(f){X,Y,D}(x::TrackedArray{X,D}, y::TrackedReal{Y,D}) = $(broadcast_f)(x, y, D)
        end
        for A in ARRAY_TYPES
            @eval begin
                @inline Base.$(f){X,D}(x::TrackedArray{X,D}, y::$A) = $(broadcast_f)(x, y, D)
                @inline Base.$(f){Y,D}(x::$A, y::TrackedArray{Y,D}) = $(broadcast_f)(x, y, D)
                @inline Base.$(f){X,D}(x::TrackedReal{X,D}, y::$A) = $(broadcast_f)(x, y, D)
                @inline Base.$(f){Y,D}(x::$A, y::TrackedReal{Y,D}) = $(broadcast_f)(x, y, D)
            end
        end
        for R in REAL_TYPES
            @eval begin
                @inline Base.$(f){X,D}(x::TrackedArray{X,D}, y::$R) = $(broadcast_f)(x, y, D)
                @inline Base.$(f){Y,D}(x::$R, y::TrackedArray{Y,D}) = $(broadcast_f)(x, y, D)
            end
        end
    end
end

# .+ #
#----#

function broadcast_plus{D}(x, y, ::Type{D})
    tp = tape(x, y)
    out = track(value(x) .+ value(y), D, tp)
    cache = (index_bound(x, out), index_bound(y, out))
    record!(tp, SpecialInstruction, Base.:(.+), (x, y), out, cache)
    return out
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(.+)})
    a, b = instruction.input
    output = instruction.output
    pull_value!(a)
    pull_value!(b)
    broadcast!(+, value(output), value(a), value(b))
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(.+)})
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

function broadcast_minus{D}(x, y, ::Type{D})
    tp = tape(x, y)
    out = track(value(x) .- value(y), D, tp)
    cache = (index_bound(x, out), index_bound(y, out))
    record!(tp, SpecialInstruction, Base.:(.-), (x, y), out, cache)
    return out
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(.-)})
    a, b = instruction.input
    output = instruction.output
    pull_value!(a)
    pull_value!(b)
    broadcast!(-, value(output), value(a), value(b))
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(.-)})
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

function broadcast_mul{D}(x, y, ::Type{D})
    tp = tape(x, y)
    out = track(value(x) .* value(y), D, tp)
    cache = (index_bound(x, out), index_bound(y, out))
    record!(tp, SpecialInstruction, Base.:(.*), (x, y), out, cache)
    return out
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(.*)})
    a, b = instruction.input
    output = instruction.output
    pull_value!(a)
    pull_value!(b)
    broadcast!(*, value(output), value(a), value(b))
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(.*)})
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

function broadcast_rdiv{D}(x, y, ::Type{D})
    tp = tape(x, y)
    out = track(value(x) ./ value(y), D, tp)
    n_partials, d_partials = rdiv_cache(x, y)
    cache = (n_partials, d_partials,
             index_bound(x, out), index_bound(y, out),
             index_bound(n_partials, out), index_bound(d_partials, out))
    record!(tp, SpecialInstruction, Base.:(./), (x, y), out, cache)
    return out
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(./)})
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

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(./)})
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

function broadcast_ldiv{D}(x, y, ::Type{D})
    tp = tape(x, y)
    out = track(value(x) .\ value(y), D, tp)
    n_partials, d_partials = rdiv_cache(y, x)
    cache = (n_partials, d_partials,
             index_bound(x, out), index_bound(y, out),
             index_bound(n_partials, out), index_bound(d_partials, out))
    record!(tp, SpecialInstruction, Base.:(.\), (x, y), out, cache)
    return out
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(.\)})
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

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(.\)})
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

pow_cache(x, y) = (base_partials(value(x), value(y)), exp_partials(value(x), value(y)))

function broadcast_pow{D}(x, y, ::Type{D})
    tp = tape(x, y)
    out = track(value(x) .^ value(y), D, tp)
    bs_partials, ex_partials = pow_cache(x, y)
    cache = (bs_partials, ex_partials,
             index_bound(x, out), index_bound(y, out),
             index_bound(bs_partials, out), index_bound(ex_partials, out))
    record!(tp, SpecialInstruction, Base.:(.^), (x, y), out, cache)
    return out
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(.^)})
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

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(.^)})
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
