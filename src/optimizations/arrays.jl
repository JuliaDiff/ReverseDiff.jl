for A in ARRAY_TYPES

    # addition/subtraction #
    #----------------------#

    @eval function Base.sum{S,T}(x::$(A){Tracer{S,T}})
        result = zero(T)
        for t in x
            result += value(t)
        end
        tp = tape(x)
        out = Tracer{S}(result, tp)
        record!(tp, sum, x, out)
        return out
    end

    for f in (:-, :+)
        @eval function Base.$(f){S,X,Y}(x::$(A){Tracer{S,X}}, y::$(A){Tracer{S,Y}})
            tp = tape(x, y)
            out = track(S, $(f)(value(x), value(y)), tp)
            record!(tp, $(f), (x, y), out)
            return out
        end
    end

    @eval function Base.:-{S,T}(x::$(A){Tracer{S,T}})
        tp = tape(x)
        out = track(S, -(value(x)), tp)
        record!(tp, -, x, out)
        return out
    end

    # A_mul_B family #
    #----------------#

    for f in (:*,
              :A_mul_Bt, :At_mul_B, :At_mul_Bt,
              :A_mul_Bc, :Ac_mul_B, :Ac_mul_Bc)
        @eval function Base.$(f){S,X,Y}(x::$(A){Tracer{S,X}}, y::$(A){Tracer{S,Y}})
            tp = tape(x, y)
            xval, yval = value(x), value(y)
            out = track(S, $(f)(xval, yval), tp)
            record!(tp, $(f), (x, y), out, (xval, yval))
            return out
        end
    end

    # in-place versions
    for (f!, f) in ((:A_mul_B!, :*),
                    (:A_mul_Bt!, :A_mul_Bt), (:At_mul_B!, :At_mul_B), (:At_mul_Bt!, :At_mul_Bt),
                    (:A_mul_Bc!, :A_mul_Bc), (:Ac_mul_B!, :Ac_mul_B), (:Ac_mul_Bc!, :Ac_mul_Bc))
        @eval function Base.$(f!){S,T,X,Y}(out::$(A){Tracer{S,T}},
                                           x::$(A){Tracer{S,X}},
                                           y::$(A){Tracer{S,Y}})
            tp = tape(x, y)
            xval, yval = value(x), value(y)
            track!(out, $(f)(xval, yval), tp)
            record!(tp, $(f), (x, y), out, (xval, yval))
            return out
        end
    end

    # linear algebra #
    #----------------#

    @eval function Base.inv{S,T}(x::$(A){Tracer{S,T}})
        tp = tape(x)
        outval = inv(value(x))
        out = track(S, outval, tp)
        record!(tp, inv, x, out, outval)
        return out
    end

    @eval function Base.det{S,T}(x::$(A){Tracer{S,T}})
        tp = tape(x)
        xval = value(x)
        out = track(S, det(xval), tp)
        record!(tp, det, x, out, inv(xval))
        return out
    end
end
