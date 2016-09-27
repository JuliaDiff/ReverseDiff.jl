######################################################
# Jacobian of `f(::AbstractArray...)::AbstractArray` #
######################################################

# jacobian #
#----------#

function jacobian(f, x, opts::Options = Options(x))
    xt, tp = opts.state, opts.tape
    track!(xt, x, tp)
    out = jacobian_reverse_pass(f(xt), xt, tp)
    empty!(tp)
    return out
end

function jacobian(f, xs::Tuple, opts::Options = Options(xs))
    xts, tp = opts.state, opts.tape
    track!(xts, xs, tp)
    out = jacobian_reverse_pass(f(xts...), xts, tp)
    empty!(tp)
    return out
end

# jacobian! #
#-----------#

function jacobian!(out, f, x, opts::Options = Options(x, eltype(out)))
    xt, tp = opts.state, opts.tape
    track!(xt, x, tp)
    load_jacobian!(out, f(xt), xt, tp)
    empty!(tp)
    return out
end

function jacobian!(outs::Tuple, f, xs::Tuple, opts::Options = Options(xs, map(eltype, outs)))
    xts, tp = opts.state, opts.tape
    track!(xts, xs, tp)
    yt = f(xts...)
    for i in eachindex(outs)
        load_jacobian!(outs[i], yt, xts[i], tp)
    end
    empty!(tp)
    return outs
end

# utilities #
#-----------#

function jacobian_reverse_pass!(out, yt, xt, tp::Tape)
    out = reshape(out, length(yt), length(xt))
    for i in eachindex(yt)
        n = yt[i]
        noskip = hastape(n)
        noskip && (seed!(n); reverse_pass!(tp))
        for j in eachindex(xt)
            out[i, j] = adjoint(xt[j])
        end
        noskip && unseed!(tp)
    end
    return out
end

function jacobian_reverse_pass(yt, xt, tp)
    out = similar(yt, valtype(eltype(yt)), length(yt), length(xt))
    return jacobian_reverse_pass!(out, yt, xt, tp)
end

function jacobian_reverse_pass(yt, xts::Tuple, tp)
    outs = map(xt -> similar(yt, valtype(eltype(yt)), length(yt), length(xt)), xts)
    for i in eachindex(outs)
        jacobian_reverse_pass!(outs[i], yt, xts[i], tp)
    end
    return outs
end

function jacobian_reverse_pass(yts::Tuple, xts::Tuple, tp)
    error("Taking the jacobian of a function which returns multiple arrays is not yet supported.")
end

load_jacobian!(out, yt, xt, tp) = jacobian_reverse_pass!(out, yt, xt, tp)

function load_jacobian!(out::JacobianResult, yt, xt, tp)
    DiffBase.value!(value, out, yt)
    jacobian_reverse_pass!(DiffBase.jacobian(out), yt, xt, tp)
    return out
end
