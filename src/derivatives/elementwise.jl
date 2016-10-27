###########
# forward #
###########

function retrack_duals{N,T,A}(duals::AbstractArray{Dual{N,T}}, ::Type{A}, tp::Nullable{Tape})
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
                out = retrack_duals(duals, S, tp)
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
                out = retrack_duals(duals, S, tp)
                record!(tp, $(g), (x1, x2), out, duals)
                return out
            end

            function Base.$(g){F,V,S}(fopt::ForwardOptimize{F},
                                      x1::$(A){Tracked{V,S}},
                                      x2::$(A))
                fdual = (t1, t2) -> fopt.f(Dual(value(t1), one(V)), t2)
                duals = $(g)(fdual, x1, x2)
                tp = tape(x1)
                out = retrack_duals(duals, S, tp)
                record!(tp, $(g), x1, out, duals)
                return out
            end

            function Base.$(g){F,V,S}(fopt::ForwardOptimize{F},
                                      x1::$(A),
                                      x2::$(A){Tracked{V,S}})
                fdual = (t1, t2) -> fopt.f(t1, Dual(value(t2), one(V)))
                duals = $(g)(fdual, x1, x2)
                tp = tape(x2)
                out = retrack_duals(duals, S, tp)
                record!(tp, $(g), x2, out, duals)
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
            out = retrack_duals(duals, S, tp)
            record!(tp, broadcast, (n, x), out, duals)
            return out
        end

        function Base.broadcast{F,X,V,S}(fopt::ForwardOptimize{F}, x::$(A){Tracked{X,S}}, n::Tracked{V,S})
            ndual = Dual(value(n), zero(V), one(V))
            fdual = t -> fopt.f(Dual(value(t), one(X), zero(X)), ndual)
            duals = broadcast(fdual, x)
            tp = tape(n, x)
            out = retrack_duals(duals, S, tp)
            record!(tp, broadcast, (x, n), out, duals)
            return out
        end

        function Base.broadcast{F,V,S}(fopt::ForwardOptimize{F}, n::Tracked{V,S}, x::$(A))
            ndual = Dual(value(n), one(V))
            fdual = t -> fopt.f(ndual, t)
            duals = broadcast(fdual, x)
            tp = tape(n)
            out = retrack_duals(duals, S, tp)
            record!(tp, broadcast, n, out, duals)
            return out
        end

        function Base.broadcast{F,V,S}(fopt::ForwardOptimize{F}, x::$(A), n::Tracked{V,S})
            ndual = Dual(value(n), one(V))
            fdual = t -> fopt.f(t, ndual)
            duals = broadcast(fdual, x)
            tp = tape(n)
            out = retrack_duals(duals, S, tp)
            record!(tp, broadcast, n, out, duals)
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

            @inline function Base.$(f){X<:Tracked}(x::$(A){X}, y::$(A))
                return broadcast(ForwardOptimize($(f)), x, y)
            end

            @inline function Base.$(f){Y<:Tracked}(x::$(A), y::$(A){Y})
                return broadcast(ForwardOptimize($(f)), x, y)
            end

            @inline function Base.$(f){T<:Tracked}(n::Tracked, x::$(A){T})
                return broadcast(ForwardOptimize($(f)), n, x)
            end

            @inline function Base.$(f){T<:Tracked}(x::$(A){T}, n::Tracked)
                return broadcast(ForwardOptimize($(f)), x, n)
            end

            @inline function Base.$(f)(n::Tracked, x::$(A))
                return broadcast(ForwardOptimize($(f)), n, x)
            end

            @inline function Base.$(f)(x::$(A), n::Tracked)
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

function special_reverse_step!{A,B}(::typeof(broadcast), inputs::Tuple{A,B}, output, duals)
    a, b = inputs
    if size(a) == size(b)
        special_reverse_step!(map, inputs, output, duals)
    else
        broadcast_adjoint_reduce!(a, output, duals, 1)
        broadcast_adjoint_reduce!(b, output, duals, 2)
    end
    return nothing
end

function special_reverse_step!(::typeof(broadcast), input, output, duals)
    if size(input) == size(output)
        special_reverse_step!(map, input, output, duals)
    else
        broadcast_adjoint_reduce!(input, output, duals, 1)
    end
    return nothing
end

# This strategy should be pretty fast, but it might be prone to numerical error if the
# accumulated adjoint becomes too large compared to the individual terms being added to
# it. This can be overcome by using the divide-and-conquer strategy used by
# Base.mapreducedim, but that strategy is less cache efficient and more complicated to
# implement.
function broadcast_adjoint_reduce!{T,N}(input::AbstractArray, output::AbstractArray{T,N}, duals, p)
    max_input_index = CartesianIndex(ntuple(i -> size(input, i), N)::NTuple{N,Int})
    output_index_range = CartesianRange(size(output))
    for i in output_index_range
        increment_adjoint!(input[min(max_input_index, i)], adjoint(output[i]) * partials(duals[i], p))
    end
    return nothing
end

function broadcast_adjoint_reduce!{T,N}(input::Number, output::AbstractArray{T,N}, duals, p)
    for i in eachindex(duals)
        increment_adjoint!(input, adjoint(output[i]) * partials(duals[i], p))
    end
    return nothing
end
