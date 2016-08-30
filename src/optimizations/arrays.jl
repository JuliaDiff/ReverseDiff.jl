for A in ARRAY_TYPES

    # addition/subtraction #
    #----------------------#

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

    for f in (:-, :+)
        @eval function Base.$(f){S,X,Y}(x::$(A){TraceReal{S,X}}, y::$(A){TraceReal{S,Y}})
            tr = trace(x, y)
            out = wrap(S, $(f)(value(x), value(y)), tr)
            record!(tr, $(f), (x, y), out)
            return out
        end
    end

    @eval function Base.:-{S,T}(x::$(A){TraceReal{S,T}})
        tr = trace(x)
        out = wrap(S, -(value(x)), tr)
        record!(tr, -, x, out)
        return out
    end

    # A_mul_B family #
    #----------------#

    for f in (:*,
              :A_mul_Bt, :At_mul_B, :At_mul_Bt,
              :A_mul_Bc, :Ac_mul_B, :Ac_mul_Bc)
        @eval function Base.$(f){S,X,Y}(x::$(A){TraceReal{S,X}}, y::$(A){TraceReal{S,Y}})
            tr = trace(x, y)
            xval, yval = value(x), value(y)
            out = wrap(S, $(f)(xval, yval), tr)
            record!(tr, $(f), (x, y), out, (xval, yval))
            return out
        end
    end

    # in-place versions
    for (f!, f) in ((:A_mul_B!, :*),
                    (:A_mul_Bt!, :A_mul_Bt), (:At_mul_B!, :At_mul_B), (:At_mul_Bt!, :At_mul_Bt),
                    (:A_mul_Bc!, :A_mul_Bc), (:Ac_mul_B!, :Ac_mul_B), (:Ac_mul_Bc!, :Ac_mul_Bc))
        @eval function Base.$(f!){S,T,X,Y}(out::$(A){TraceReal{S,T}},
                                           x::$(A){TraceReal{S,X}},
                                           y::$(A){TraceReal{S,Y}})
            tr = trace(x, y)
            xval, yval = value(x), value(y)
            wrap!(out, $(f)(xval, yval), tr)
            record!(tr, $(f), (x, y), out, (xval, yval))
            return out
        end
    end

    # linear algebra #
    #----------------#

    @eval function Base.inv{S,T}(x::$(A){TraceReal{S,T}})
        tr = trace(x)
        outval = inv(value(x))
        out = wrap(S, outval, tr)
        record!(tr, inv, x, out, outval)
        return out
    end

    @eval function Base.det{S,T}(x::$(A){TraceReal{S,T}})
        tr = trace(x)
        xval = value(x)
        out = wrap(S, det(xval), tr)
        record!(tr, det, x, out, inv(xval))
        return out
    end
end
