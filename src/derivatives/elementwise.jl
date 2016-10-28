###############
# record pass #
###############

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
                record_node!(tp, Special, $(g), x, out, (fdual, duals))
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
                record_node!(tp, Special, $(g), (x1, x2), out, (fdual, duals))
                return out
            end

            function Base.$(g){F,V,S}(fopt::ForwardOptimize{F},
                                      x1::$(A){Tracked{V,S}},
                                      x2::$(A))
                fdual = (t1, t2) -> fopt.f(Dual(value(t1), one(V)), t2)
                duals = $(g)(fdual, x1, x2)
                tp = tape(x1)
                out = retrack_duals(duals, S, tp)
                record_node!(tp, Special, $(g), (x1, x2), out, (fdual, duals))
                return out
            end

            function Base.$(g){F,V,S}(fopt::ForwardOptimize{F},
                                      x1::$(A),
                                      x2::$(A){Tracked{V,S}})
                fdual = (t1, t2) -> fopt.f(t1, Dual(value(t2), one(V)))
                duals = $(g)(fdual, x1, x2)
                tp = tape(x2)
                out = retrack_duals(duals, S, tp)
                record_node!(tp, Special, $(g), (x1, x2), out, (fdual, duals))
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
            fdual = (n, t) -> fopt.f(Dual(value(n), one(V), zero(V)),
                                     Dual(value(t), zero(X), one(X)))
            duals = broadcast(fdual, n, x)
            tp = tape(n, x)
            out = retrack_duals(duals, S, tp)
            record_node!(tp, Special, broadcast, (n, x), out, (fdual, duals))
            return out
        end

        function Base.broadcast{F,X,V,S}(fopt::ForwardOptimize{F}, x::$(A){Tracked{X,S}}, n::Tracked{V,S})
            fdual = (t, n) -> fopt.f(Dual(value(t), one(X), zero(X)),
                                     Dual(value(n), zero(V), one(V)))
            duals = broadcast(fdual, x, n)
            tp = tape(n, x)
            out = retrack_duals(duals, S, tp)
            record_node!(tp, Special, broadcast, (x, n), out, (fdual, duals))
            return out
        end

        function Base.broadcast{F,V,S}(fopt::ForwardOptimize{F}, n::Tracked{V,S}, x::$(A))
            fdual = (n, t) -> fopt.f(Dual(value(n), one(V)), t)
            duals = broadcast(fdual, n, x)
            tp = tape(n)
            out = retrack_duals(duals, S, tp)
            record_node!(tp, Special, broadcast, (n, x), out, (fdual, duals))
            return out
        end

        function Base.broadcast{F,V,S}(fopt::ForwardOptimize{F}, x::$(A), n::Tracked{V,S})
            fdual = (t, n) -> fopt.f(t, Dual(value(n), one(V)))
            duals = broadcast(fdual, x, n)
            tp = tape(n)
            out = retrack_duals(duals, S, tp)
            record_node!(tp, Special, broadcast, (x, n), out, (fdual, duals))
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

################
# forward pass #
################

for (g!, g) in ((:map!, :map), (:broadcast!, :broadcast))
    @eval function special_forward_step!(::typeof($g), input, output, cache)
        fdual, duals = cache
        ($g!)(fdual, duals, input)
        setvalue!(value, output, duals)
        return nothing
    end

    @eval function special_forward_step!{A,B}(::typeof($g), inputs::Tuple{A,B}, output, cache)
        a, b = inputs
        fdual, duals = cache
        ($g!)(fdual, duals, a, b)
        setvalue!(value, output, duals)
        return nothing
    end
end

################
# reverse pass #
################

# map #
#-----#

function special_reverse_step!(::typeof(map), input::AbstractArray, output::AbstractArray, cache)
    _, duals = cache
    for i in eachindex(output)
        increment_adjoint!(input[i], adjoint(output[i]) * partials(duals[i], 1))
    end
    return nothing
end

function special_reverse_step!{A,B}(::typeof(map), inputs::Tuple{A,B}, output::AbstractArray, cache)
    a, b = inputs
    _, duals = cache
    if eltype(A) <: Tracked && eltype(B) <: Tracked
        map_adjoint_reduce!(a, b, output, duals)
    elseif eltype(A) <: Tracked
        map_adjoint_reduce!(a, output, duals)
    else
        map_adjoint_reduce!(b, output, duals)
    end
    return nothing
end

function map_adjoint_reduce!(input, output, duals)
    for i in eachindex(output)
        increment_adjoint!(input[i], adjoint(output[i]) * partials(duals[i], 1))
    end
    return nothing
end

function map_adjoint_reduce!(a, b, output, duals)
    for i in eachindex(output)
        output_adjoint = adjoint(output[i])
        a_partial, b_partial = partials(duals[i])
        increment_adjoint!(a[i], output_adjoint * a_partial)
        increment_adjoint!(b[i], output_adjoint * b_partial)
    end
    return nothing
end

# broadcast #
#-----------#

function special_reverse_step!{A,B}(::typeof(broadcast), inputs::Tuple{A,B}, output, cache)
    a, b = inputs
    if size(a) == size(b)
        special_reverse_step!(map, inputs, output, cache)
    else
        _, duals = cache
        if eltype(A) <: Tracked && eltype(B) <: Tracked
            broadcast_adjoint_reduce!(a, output, duals, 1)
            broadcast_adjoint_reduce!(b, output, duals, 2)
        elseif eltype(A) <: Tracked
            broadcast_adjoint_reduce!(a, output, duals)
        else
            broadcast_adjoint_reduce!(b, output, duals)
        end
    end
    return nothing
end

function special_reverse_step!(::typeof(broadcast), input, output, cache)
    if size(input) == size(output)
        special_reverse_step!(map, input, output, cache)
    else
        _, duals = cache
        broadcast_adjoint_reduce!(input, output, duals)
    end
    return nothing
end

# This strategy should be pretty fast, but it might be prone to numerical error if the
# accumulated adjoint becomes too large compared to the individual terms being added to
# it. This can be overcome by using the divide-and-conquer strategy used by
# Base.mapreducedim, but that strategy is less cache efficient and more complicated to
# implement.
function broadcast_adjoint_reduce!{T,N}(input::AbstractArray, output::AbstractArray{T,N},
                                        duals::AbstractArray, p = 1)
    max_input_index = CartesianIndex(ntuple(i -> size(input, i), N)::NTuple{N,Int})
    output_index_range = CartesianRange(size(output))
    for i in output_index_range
        increment_adjoint!(input[min(max_input_index, i)], adjoint(output[i]) * partials(duals[i], p))
    end
    return nothing
end

function broadcast_adjoint_reduce!{T,N}(input::Number, output::AbstractArray{T,N},
                                        duals::AbstractArray, p = 1)
    for i in eachindex(duals)
        increment_adjoint!(input, adjoint(output[i]) * partials(duals[i], p))
    end
    return nothing
end
