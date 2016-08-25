######################
# gradient/gradient! #
######################

function gradient(f, x, trx = wrap(x))
    result = f(trx)
    seed!(result)
    backprop!(get(trace(result)))
    return adjoint(trx)
end

function gradient!(out, f, x, trx = wrap(eltype(out), x))
    result = f(trx)
    seed!(result)
    backprop!(get(trace(result)))
    return adjoint!(out, trx)
end

######################
# jacobian/jacobian! #
######################

function load_jacobian!(out, trx, y, tr::Trace)
    for i in eachindex(y)
        n = y[i]
        seed!(n)
        backprop!(tr)
        for j in eachindex(trx)
            m = trx[j]
            out[i, j] = adjoint(m)
            unseed!(m)
        end
        unseed!(n)
    end
    return out
end

function jacobian(f, x, trx = wrap(x))
    tr = get(trace(first(trx)))
    y = f(trx)
    out = similar(y, eltype(x), length(y), length(x))
    return load_jacobian!(out, trx, y, tr)
end

function jacobian!(out, f, x, trx = wrap(eltype(out), x))
    tr = get(trace(first(trx)))
    y = f(trx)
    load_jacobian!(reshape(out, length(y), length(x)), trx, y, tr)
    return out
end
