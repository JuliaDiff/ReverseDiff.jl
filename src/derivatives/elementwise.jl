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
        record!(tp, SpecialInstruction, $(g), x, out, (fdual, duals))
        return out
    end

    for A in ARRAY_TYPES, T in (:TrackedArray, :TrackedReal)
        @eval function Base.$(g){F,X,D}(f::ForwardOptimize{F}, x::$(T){X,D}, y::$A)
            fdual = (vx, vy) -> f.f(Dual(vx, one(X), zero(X)), Dual(vy, zero(vy), one(vy)))
            duals = $(g)(fdual, value(x), value(y))
            tp = tape(x)
            out = track(dual_values(duals), D, tp)
            record!(tp, SpecialInstruction, $(g), (x, y), out, (fdual, duals))
            return out
        end

        @eval function Base.$(g){F,Y,D}(f::ForwardOptimize{F}, x::$A, y::$(T){Y,D})
            fdual = (vx, vy) -> f.f(Dual(vx, one(vx), zero(vx)), Dual(vy, zero(Y), one(Y)))
            duals = $(g)(fdual, value(x), value(y))
            tp = tape(y)
            out = track(dual_values(duals), D, tp)
            record!(tp, SpecialInstruction, $(g), (x, y), out, (fdual, duals))
            return out
        end
    end

    for TX in (:TrackedArray, :TrackedReal), TY in (:TrackedArray, :TrackedReal)
        (TX == TY == :TrackedReal) && continue
        @eval function Base.$(g){F,X,Y,D}(f::ForwardOptimize{F}, x::$(TX){X,D}, y::$(TY){Y,D})
            fdual = (vx, vy) -> f.f(Dual(vx, one(X), zero(X)), Dual(vy, zero(Y), one(Y)))
            duals = $(g)(fdual, value(x), value(y))
            tp = tape(x, y)
            out = track(dual_values(duals), D, tp)
            record!(tp, SpecialInstruction, $(g), (x, y), out, (fdual, duals))
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

# forward pass #
#--------------#

for (g!, g) in ((:map!, :map), (:broadcast!, :broadcast))
    @eval begin
        @noinline function special_forward_exec!(instruction::SpecialInstruction{typeof($g)})
            input, output = instruction.input, instruction.output
            fdual, duals = instruction.cache
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

# reverse pass (map) #
#--------------------#

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(map)})
    input = instruction.input
    output = instruction.output
    _, duals = instruction.cache
    if istracked(input)
        map_duals_increment!(input, output, duals, 1)
    else
        a, b = input
        istracked(a) && map_duals_increment!(a, output, duals, 1)
        istracked(b) && map_duals_increment!(b, output, duals, 2)
    end
    unseed!(output)
    return nothing
end

# reverse pass (broadcast) #
#--------------------------#

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(broadcast)})
    input = instruction.input
    output = instruction.output
    cache = instruction.cache
    _, duals = instruction.cache
    if istracked(input)
        if size(input) == size(output)
            map_duals_increment!(input, output, duals, 1)
        else
            broadcast_duals_increment!(input, output, duals, 1)
        end
    else
        a, b = input
        if size(a) == size(b)
            istracked(a) && map_duals_increment!(a, output, duals, 1)
            istracked(b) && map_duals_increment!(b, output, duals, 2)
        else
            istracked(a) && broadcast_duals_increment!(a, output, duals, 1)
            istracked(b) && broadcast_duals_increment!(b, output, duals, 2)
        end
    end
    unseed!(output)
    return nothing
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
        record!(tape(x), SpecialInstruction, $(g), x, out, (fdual, duals))
        return out
    end

    for TX in (:TrackedArray, :TrackedReal), TY in (:TrackedArray, :TrackedReal)
        (TX == TY == :TrackedReal) && continue
        @eval function Base.$(g!){F,X,Y}(f::ForwardOptimize{F}, out::TrackedArray, x::$(TX){X}, y::$(TY){Y})
            fdual = (vx, vy) -> f.f(Dual(vx, one(X), zero(X)), Dual(vy, zero(Y), one(Y)))
            duals = $(g)(fdual, value(x), value(y))
            copy!(value(out), dual_values(duals))
            record!(tape(x, y), SpecialInstruction, $(g), (x, y), out, (fdual, duals))
            return out
        end
    end

    for A in ARRAY_TYPES, T in (:TrackedArray, :TrackedReal)
        @eval function Base.$(g!){F,X}(f::ForwardOptimize{F}, out::TrackedArray, x::$(T){X}, y::$A)
            fdual = (vx, vy) -> f.f(Dual(vx, one(X), zero(X)), Dual(vy, zero(vy), one(vy)))
            duals = $(g)(fdual, value(x), value(y))
            copy!(value(out), dual_values(duals))
            record!(tape(x), SpecialInstruction, $(g), (x, y), out, (fdual, duals))
            return out
        end

        @eval function Base.$(g!){F,Y}(f::ForwardOptimize{F}, out::TrackedArray, x::$A, y::$(T){Y})
            fdual = (vx, vy) -> f.f(Dual(vx, one(vx), zero(vx)), Dual(vy, zero(Y), one(Y)))
            duals = $(g)(fdual, value(x), value(y))
            copy!(value(out), dual_values(duals))
            record!(tape(x), SpecialInstruction, $(g), (x, y), out, (fdual, duals))
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

#############################
# built-in infix operations #
#############################

# dispatch #
#----------#

for (f, broadcast_f) in ((:.+, :broadcast_plus),
                         (:.-, :broadcast_minus),
                         (:.*, :broadcast_mul),
                         (:./, :broadcast_rdiv),
                         (:.\, :broadcast_ldiv),
                         (:.^, :broadcast_exp))
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

# .+ #
#----#

function broadcast_plus{D}(x, y, ::Type{D})
    tp = tape(x, y)
    out = track(value(x) .+ value(y), D, tp)
    record!(tp, SpecialInstruction, Base.:(.+), (x, y), out)
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
    istracked(a) && broadcast_deriv_increment!(a, output)
    istracked(b) && broadcast_deriv_increment!(b, output)
    unseed!(output)
    return nothing
end

# .- #
#----#

function broadcast_minus{D}(x, y, ::Type{D})
    tp = tape(x, y)
    out = track(value(x) .- value(y), D, tp)
    record!(tp, SpecialInstruction, Base.:(.-), (x, y), out)
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
    istracked(a) && broadcast_deriv_increment!(a, output)
    istracked(b) && broadcast_deriv_decrement!(b, output)
    unseed!(output)
    return nothing
end

# .* #
#----#

function broadcast_mul{D}(x, y, ::Type{D})
    tp = tape(x, y)
    out = track(value(x) .* value(y), D, tp)
    record!(tp, SpecialInstruction, Base.:(.*), (x, y), out)
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
    istracked(a) && broadcast_deriv_increment!(a, output, value(b))
    istracked(b) && broadcast_deriv_increment!(b, output, value(a))
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

function broadcast_rdiv{D}(x, y, ::Type{D})
    tp = tape(x, y)
    out = track(value(x) ./ value(y), D, tp)
    n_partials = numer_partials(value(y))
    d_partials = denom_partials(value(x), value(y))
    cache = (n_partials, d_partials)
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
    numer_partials!(n_partials, b_value)
    denom_partials!(d_partials, a_value, b_value)
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(./)})
    a, b = instruction.input
    output = instruction.output
    n_partials, d_partials = instruction.cache
    istracked(a) && broadcast_deriv_increment!(a, output, n_partials)
    istracked(b) && broadcast_deriv_increment!(b, output, d_partials)
    unseed!(output)
    return nothing
end

# .\ #
#----#

function broadcast_ldiv{D}(x, y, ::Type{D})
    tp = tape(x, y)
    out = track(value(x) .\ value(y), D, tp)
    n_partials = numer_partials(value(x))
    d_partials = denom_partials(value(y), value(x))
    cache = (n_partials, d_partials)
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
    numer_partials!(n_partials, a_value)
    denom_partials!(d_partials, b_value, a_value)
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(.\)})
    a, b = instruction.input
    output = instruction.output
    n_partials, d_partials = instruction.cache
    istracked(a) && broadcast_deriv_increment!(a, output, d_partials)
    istracked(b) && broadcast_deriv_increment!(b, output, n_partials)
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

function broadcast_exp{D}(x, y, ::Type{D})
    tp = tape(x, y)
    out = track(value(x) .^ value(y), D, tp)
    bs_partials = base_partials(value(x), value(y))
    ex_partials = exp_partials(value(x), value(y))
    cache = (bs_partials, ex_partials)
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
    base_partials!(bs_partials, a_value, b_value)
    exp_partials!(ex_partials, a_value, b_value)
    return nothing
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(.^)})
    a, b = instruction.input
    output = instruction.output
    bs_partials, ex_partials = instruction.cache
    istracked(a) && broadcast_deriv_increment!(a, output, bs_partials)
    istracked(b) && broadcast_deriv_increment!(b, output, ex_partials)
    unseed!(output)
    return nothing
end

#############
# utilities #
#############

#=
The strategies below should be decently fast, but some might be prone to numerical error
if the accumulated derivative becomes too large compared to the individual terms being
added to it. This can be overcome by using the divide-and-conquer strategy from
Base.mapreducedim, but that strategy is less cache efficient and more complicated to
implement.

There's also a lot of boilerplate code here, but I'm not sure there's a cleaner way to
do this that doesn't sacrifice efficiency.
=#

max_leftover_index{T,N}(x, ::AbstractArray{T,N}) = CartesianIndex{N}(ntuple(i -> size(x, i), Val{N}))

broadcast_deriv_increment!(input, output, partials::Ref) = broadcast_deriv_increment!(input, output, partials[])

dual_values(duals) = map!(ForwardDiff.value, similar(duals, ForwardDiff.valtype(eltype(duals))), duals)

# map_duals_increment!/broadcast_duals_increment! #
#-------------------------------------------------#

function map_duals_increment!(input::TrackedArray, output, duals, p::Int)
    input_deriv, output_deriv = deriv(input), deriv(output)
    for i in eachindex(output_deriv)
        input_deriv[i] += output_deriv[i] * ForwardDiff.partials(duals[i], p)
    end
    return nothing
end

function map_duals_increment!(input::AbstractArray, output, duals, p::Int)
    output_deriv = deriv(output)
    for i in eachindex(output_deriv)
        increment_deriv!(input[i], output_deriv[i] * ForwardDiff.partials(duals[i], p))
    end
    return nothing
end

function broadcast_duals_increment!(input::TrackedArray, output, duals, p::Int)
    max_input_index = max_leftover_index(input, output)
    input_deriv, output_deriv = deriv(input), deriv(output)
    for i in CartesianRange(size(output))
        input_deriv[min(max_input_index, i)] += output_deriv[i] * ForwardDiff.partials(duals[i], p)
    end
    return nothing
end

function broadcast_duals_increment!(input::AbstractArray, output, duals, p::Int)
    max_input_index = max_leftover_index(input, output)
    output_deriv = deriv(output)
    for i in CartesianRange(size(output))
        increment_deriv!(input[min(max_input_index, i)], output_deriv[i] * ForwardDiff.partials(duals[i], p))
    end
    return nothing
end

function broadcast_duals_increment!(input::TrackedReal, output, duals, p::Int)
    for i in eachindex(duals)
        increment_deriv!(input, deriv(output[i]) * ForwardDiff.partials(duals[i], p))
    end
    return nothing
end

# broadcast_deriv_increment! #
#----------------------------#

function broadcast_deriv_increment!(input::TrackedArray, output)
    max_input_index = max_leftover_index(input, output)
    input_deriv, output_deriv = deriv(input), deriv(output)
    for i in CartesianRange(size(output))
        input_deriv[min(max_input_index, i)] += output_deriv[i]
    end
    return nothing
end

function broadcast_deriv_decrement!(input::TrackedArray, output)
    max_input_index = max_leftover_index(input, output)
    input_deriv, output_deriv = deriv(input), deriv(output)
    for i in CartesianRange(size(output))
        input_deriv[min(max_input_index, i)] -= output_deriv[i]
    end
    return nothing
end

function broadcast_deriv_increment!(input::TrackedArray, output, partials::AbstractArray)
    max_input_index = max_leftover_index(input, output)
    max_partials_index = max_leftover_index(partials, output)
    input_deriv, output_deriv = deriv(input), deriv(output)
    for i in CartesianRange(size(output))
        input_deriv[min(max_input_index, i)] += output_deriv[i] * partials[min(max_partials_index, i)]
    end
    return nothing
end

function broadcast_deriv_increment!(input::AbstractArray, output, partials::Real)
    max_input_index =  max_leftover_index(input, output)
    output_deriv = deriv(output)
    for i in CartesianRange(size(output))
        increment_deriv!(input[min(max_input_index, i)], output_deriv[i] * partials)
    end
    return nothing
end

function broadcast_deriv_increment!(input::TrackedArray, output, partials::Real)
    max_input_index = max_leftover_index(input, output)
    input_deriv, output_deriv = deriv(input), deriv(output)
    for i in CartesianRange(size(output))
        input_deriv[min(max_input_index, i)] += output_deriv[i] * partials
    end
    return nothing
end

function broadcast_deriv_increment!(input::AbstractArray, output, partials::AbstractArray)
    max_input_index = max_leftover_index(input, output)
    max_partials_index = max_leftover_index(partials, output)
    output_deriv = deriv(output)
    for i in CartesianRange(size(output))
        increment_deriv!(input[min(max_input_index, i)], output_deriv[i] * partials[min(max_partials_index, i)])
    end
    return nothing
end

function broadcast_deriv_increment!(input::TrackedReal, output::TrackedArray, partials::AbstractArray)
    output_deriv = deriv(output)
    for i in eachindex(output_deriv)
        increment_deriv!(input, output_deriv[i] * partials[i])
    end
    return nothing
end

function broadcast_deriv_increment!(input::TrackedReal, output::TrackedArray, partials::Real)
    output_deriv = deriv(output)
    for i in eachindex(output_deriv)
        increment_deriv!(input, output_deriv[i] * partials)
    end
    return nothing
end

function broadcast_deriv_increment!(input::TrackedReal, output::TrackedArray)
    output_deriv = deriv(output)
    for i in eachindex(output_deriv)
        increment_deriv!(input, output_deriv[i])
    end
    return nothing
end

function broadcast_deriv_decrement!(input::TrackedReal, output::TrackedArray)
    output_deriv = deriv(output)
    for i in eachindex(output_deriv)
        decrement_deriv!(input, output_deriv[i])
    end
    return nothing
end
