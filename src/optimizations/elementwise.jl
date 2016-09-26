function dualwrap{S,N,T}(::Type{S}, duals::AbstractArray{Dual{N,T}}, tp::Nullable{Tape})
    ts = similar(duals, Tracer{S,T})
    for i in eachindex(duals)
        ts[i] = Tracer{S}(value(duals[i]), tp)
    end
    return ts
end

for A in ARRAY_TYPES
    # map/broadcast on arrays #
    #-------------------------#
    for g in (:map, :broadcast)
        @eval begin
            function Base.$(g){F,S,T}(fopt::ForwardOptimize{F}, x::$(A){Tracer{S,T}})
                fdual = t -> fopt.f(Dual(value(t), one(T)))
                duals = $(g)(fdual, x)
                tp = tape(x)
                out = dualwrap(S, duals, tp)
                record!(tp, $(g), x, out, duals)
                return out
            end

            function Base.$(g){F,S,T1,T2}(fopt::ForwardOptimize{F},
                                          x1::$(A){Tracer{S,T1}},
                                          x2::$(A){Tracer{S,T2}})
                fdual = (t1, t2) -> fopt.f(Dual(value(t1), one(T1), zero(T1)),
                                           Dual(value(t2), zero(T2), one(T2)))
                duals = $(g)(fdual, x1, x2)
                tp = tape(x1, x2)
                out = dualwrap(S, duals, tp)
                record!(tp, $(g), (x1, x2), out, duals)
                return out
            end
        end
    end

    # broadcast scalars vs. arrays #
    #------------------------------#
    for R in REAL_TYPES
        @eval begin
            @inline function Base.broadcast{F,S,T}(fopt::ForwardOptimize{F}, n::$R, x::$(A){Tracer{S,T}})
                return broadcast(ForwardOptimize(t -> fopt.f(n, t)), x)
            end

            @inline function Base.broadcast{F,S,T}(fopt::ForwardOptimize{F}, x::$(A){Tracer{S,T}}, n::$R)
                return broadcast(ForwardOptimize(t -> fopt.f(t, n)), x)
            end
        end
    end

    @eval begin
        function Base.broadcast{F,S,T,X}(fopt::ForwardOptimize{F}, n::Tracer{S,T}, x::$(A){Tracer{S,X}})
            ndual = Dual(value(n), one(T), zero(T))
            fdual = t -> fopt.f(ndual, Dual(value(t), zero(X), one(X)))
            duals = broadcast(fdual, x)
            tp = tape(n, x)
            out = dualwrap(S, duals, tp)
            record!(tp, broadcast, (n, x), out, duals)
            return out
        end

        function Base.broadcast{F,S,T,X}(fopt::ForwardOptimize{F}, x::$(A){Tracer{S,X}}, n::Tracer{S,T})
            ndual = Dual(value(n), zero(T), one(T))
            fdual = t -> fopt.f(Dual(value(t), one(X), zero(X)), ndual)
            duals = broadcast(fdual, x)
            tp = tape(n, x)
            out = dualwrap(S, duals, tp)
            record!(tp, broadcast, (x, n), out, duals)
            return out
        end
    end

    # standard elementwise operations (.+, .-, .*, etc.) #
    #----------------------------------------------------#
    for f in (:.+, :.-, :.*, :./, :.\, :.^)
        @eval begin
            @inline function Base.$(f){S,X,Y}(x::$(A){Tracer{S,X}}, y::$(A){Tracer{S,Y}})
                return broadcast(ForwardOptimize($(f)), x, y)
            end

            @inline function Base.$(f){S,T}(n::Tracer, x::$(A){Tracer{S,T}})
                return broadcast(ForwardOptimize($(f)), n, x)
            end

            @inline function Base.$(f){S,T}(x::$(A){Tracer{S,T}}, n::Tracer)
                return broadcast(ForwardOptimize($(f)), x, n)
            end
        end
        for R in REAL_TYPES
            @eval begin
                @inline function Base.$(f){S,T}(n::$R, x::$(A){Tracer{S,T}})
                    return broadcast(ForwardOptimize($(f)), n, x)
                end

                @inline function Base.$(f){S,T}(x::$(A){Tracer{S,T}}, n::$R)
                    return broadcast(ForwardOptimize($(f)), x, n)
                end
            end
        end
    end
end
