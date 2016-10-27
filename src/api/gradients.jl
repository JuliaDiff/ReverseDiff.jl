#############################################
# Gradient of `f(::AbstractArray...)::Real` #
#############################################

# gradient #
#----------#

function gradient(f, x, opts::Options = Options(x))
    xt, tp = opts.state, opts.tape
    track!(xt, x, tp)
    yt = f(xt)
    if hastape(yt)
        seed!(yt)
        reverse_pass!(tp)
    end
    out = adjoint(xt)
    empty!(tp)
    return out
end

function gradient(f, xs::Tuple, opts::Options = Options(xs))
    xts, tp = opts.state, opts.tape
    track!(xts, xs, tp)
    yt = f(xts...)
    if hastape(yt)
        seed!(yt)
        reverse_pass!(tp)
    end
    outs = map(adjoint, xts)
    empty!(tp)
    return outs
end

# gradient! #
#-----------#

function gradient!(out, f, x, opts::Options = Options(x, eltype(out)))
    xt, tp = opts.state, opts.tape
    track!(xt, x, tp)
    yt = f(xt)
    if hastape(yt)
        seed!(yt)
        reverse_pass!(tp)
    end
    load_gradient!(out, yt, xt)
    empty!(tp)
    return out
end

function gradient!(outs::Tuple, f, xs::Tuple, opts::Options = Options(xs, eltype(first(outs))))
    xts, tp = opts.state, opts.tape
    track!(xts, xs, tp)
    yt = f(xts...)
    if hastape(yt)
        seed!(yt)
        reverse_pass!(tp)
    end
    for i in eachindex(outs)
        load_gradient!(outs[i], yt, xts[i])
    end
    empty!(tp)
    return outs
end

# utilities #
#-----------#

load_gradient!(out, yt, xt) = adjoint!(out, xt)

function load_gradient!(out::DiffResult, yt, xt)
    DiffBase.value!(value, out, yt)
    DiffBase.gradient!(adjoint, out, xt)
    return out
end
