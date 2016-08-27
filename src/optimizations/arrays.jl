for A in ARRAY_TYPES
    @eval function Base.sum{S,T}(x::$(A){TraceReal{S,T}})
        result = zero(T)
        for t in x
            result += value(t)
        end
        tr = trace(x)
        out = TraceReal{S}(result, tr)
        record!(tr, sum, x, out)
        return out
    end

    # unary #
    #-------#

    for f in (:-, :inv, :det)
        @eval function Base.$(f){S,T}(x::$(A){TraceReal{S,T}})
            tr = trace(x)
            out = wrap(S, $(f)(value(x)), tr)
            record!(tr, $(f), x, out)
            return out
        end
    end

    # binary #
    #--------#

    for f in (:-, :+, :*,
              :A_mul_Bt, :At_mul_B, :At_mul_Bt,
              :A_mul_Bc, :Ac_mul_B, :Ac_mul_Bc)
        @eval function Base.$(f){S,X,Y}(x::$(A){TraceReal{S,X}}, y::$(A){TraceReal{S,Y}})
            tr = trace(x, y)
            out = wrap(S, $(f)(value(x), value(y)), tr)
            record!(tr, $(f), (x, y), out)
            return out
        end
    end


    # in-place A_mul_B family #
    #-------------------------#

    for (f!, f) in ((:A_mul_B!, :*),
                    (:A_mul_Bt!, :A_mul_Bt), (:At_mul_B!, :At_mul_B), (:At_mul_Bt!, :At_mul_Bt),
                    (:A_mul_Bc!, :A_mul_Bc), (:Ac_mul_B!, :Ac_mul_B), (:Ac_mul_Bc!, :Ac_mul_Bc))
        @eval function Base.$(f!){S,T,X,Y}(out::$(A){TraceReal{S,T}},
                                           x::$(A){TraceReal{S,X}},
                                           y::$(A){TraceReal{S,Y}})
            tr = trace(x, y)
            wrap!(out, $(f)(value(x), value(y)), tr)
            record!(tr, $(f), (x, y), out)
            return out
        end
    end
end
