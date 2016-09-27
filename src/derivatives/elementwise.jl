###########
# forward #
###########

function dualwrap{N,T,A}(duals::AbstractArray{Dual{N,T}}, ::Type{A}, tp::Nullable{Tape})
    ts = similar(duals, Tracked{T,A})
    for i in eachindex(duals)
        ts[i] = Tracked(value(duals[i]), A, tp)
    end
    return ts
end

for A in ARRAY_TYPES
    # map/broadcast on arrays #
    #-------------------------#
    for g in (:map, :broadcast)
        @eval begin
            function Base.$(g){F,V,S}(fopt::ForwardOptimize{F}, x::$(A){Tracked{V,S}})
                fdual = t -> fopt.f(Dual(value(t), one(V)))
                duals = $(g)(fdual, x)
                tp = tape(x)
                out = dualwrap(duals, S, tp)
                record!(tp, $(g), x, out, duals)
                return out
            end

            function Base.$(g){F,V1,V2,S}(fopt::ForwardOptimize{F},
                                          x1::$(A){Tracked{V1,S}},
                                          x2::$(A){Tracked{V2,S}})
                fdual = (t1, t2) -> fopt.f(Dual(value(t1), one(V1), zero(V1)),
                                           Dual(value(t2), zero(V2), one(V2)))
                duals = $(g)(fdual, x1, x2)
                tp = tape(x1, x2)
                out = dualwrap(duals, S, tp)
                record!(tp, $(g), (x1, x2), out, duals)
                return out
            end
        end
    end

    # broadcast scalars vs. arrays #
    #------------------------------#
    for R in REAL_TYPES
        @eval begin
            @inline function Base.broadcast{F,V,S}(fopt::ForwardOptimize{F}, n::$R, x::$(A){Tracked{V,S}})
                return broadcast(ForwardOptimize(t -> fopt.f(n, t)), x)
            end

            @inline function Base.broadcast{F,V,S}(fopt::ForwardOptimize{F}, x::$(A){Tracked{V,S}}, n::$R)
                return broadcast(ForwardOptimize(t -> fopt.f(t, n)), x)
            end
        end
    end

    @eval begin
        function Base.broadcast{F,V,X,S}(fopt::ForwardOptimize{F}, n::Tracked{V,S}, x::$(A){Tracked{X,S}})
            ndual = Dual(value(n), one(V), zero(V))
            fdual = t -> fopt.f(ndual, Dual(value(t), zero(X), one(X)))
            duals = broadcast(fdual, x)
            tp = tape(n, x)
            out = dualwrap(duals, S, tp)
            record!(tp, broadcast, (n, x), out, duals)
            return out
        end

        function Base.broadcast{F,X,V,S}(fopt::ForwardOptimize{F}, x::$(A){Tracked{X,S}}, n::Tracked{V,S})
            ndual = Dual(value(n), zero(V), one(V))
            fdual = t -> fopt.f(Dual(value(t), one(X), zero(X)), ndual)
            duals = broadcast(fdual, x)
            tp = tape(n, x)
            out = dualwrap(duals, S, tp)
            record!(tp, broadcast, (x, n), out, duals)
            return out
        end
    end

    # standard elementwise operations (.+, .-, .*, etc.) #
    #----------------------------------------------------#
    for f in (:.+, :.-, :.*, :./, :.\, :.^)
        @eval begin
            @inline function Base.$(f){X<:Tracked,Y<:Tracked}(x::$(A){X}, y::$(A){Y})
                return broadcast(ForwardOptimize($(f)), x, y)
            end

            @inline function Base.$(f){T<:Tracked}(n::Tracked, x::$(A){T})
                return broadcast(ForwardOptimize($(f)), n, x)
            end

            @inline function Base.$(f){T<:Tracked}(x::$(A){T}, n::Tracked)
                return broadcast(ForwardOptimize($(f)), x, n)
            end
        end
        for R in REAL_TYPES
            @eval begin
                @inline function Base.$(f){V,S}(n::$R, x::$(A){Tracked{V,S}})
                    return broadcast(ForwardOptimize($(f)), n, x)
                end

                @inline function Base.$(f){V,S}(x::$(A){Tracked{V,S}}, n::$R)
                    return broadcast(ForwardOptimize($(f)), x, n)
                end
            end
        end
    end
end

###########
# reverse #
###########

# map #
#-----#

function special_reverse_step!(::typeof(map), input, output, duals)
    for i in eachindex(output)
        scalar_reverse_step!(input[i], output[i], partials(duals[i]))
    end
    return nothing
end

function special_reverse_step!{A,B}(::typeof(map), inputs::Tuple{A,B}, output, duals)
    a, b = inputs
    for i in eachindex(output)
        scalar_reverse_step!((a[i], b[i]), output[i], partials(duals[i]))
    end
    return nothing
end

# broadcast #
#-----------#

function special_reverse_step!(::typeof(broadcast), input::AbstractArray, output, duals)
    return special_reverse_step!(map, input, output, duals)
end

function special_reverse_step!{A,B}(::typeof(broadcast), inputs::Tuple{A,B}, output, duals)
    a, b = inputs
    if size(a) == size(b)
        special_reverse_step!(map, inputs, output, duals)
    else
        for i in eachindex(duals)
            duals[i] *= adjoint(output[i])
        end
        s = sumover(1, a, duals)
        increment_adjoint!(a, s)
        increment_adjoint!(b, sumover(2, b, duals))
    end
    return nothing
end

# Inference here is pretty wonky (see JuliaLang/julia#10533),
# so it's important that we allocate the array for the sum
# result ourselves. Otherwise, `reducedim_init` tries to
# allocate an array of the wrong type in some cases, which
# leads to conversion errors.
function sumover{N,M,T}(p, x::AbstractArray, duals::AbstractArray{Dual{N,T},M})
    dims = (size(x, i) != size(duals, i) ? 1 : size(duals, i) for i in 1:ndims(duals))
    result = similar(duals, T, (dims...)::NTuple{M,Int})
    sum!(d -> partials(d, p), result, duals)
    return result
end

sumover(p, x::Real, duals) = sum(d -> partials(d, p), duals)
