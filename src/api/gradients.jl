#############################################
# Gradient of `f(::AbstractArray...)::Real` #
#############################################

# gradient #
#----------#

function gradient(f, x, opts::Options = Options(x))
    r = Record(f, x, opts)
    xt, yt, tp = r.inputs, r.outputs, r.tape
    run_gradient_passes!(yt, tp)
    out = adjoint(xt)
    empty!(tp)
    return out
end

function gradient(f, xs::Tuple, opts::Options = Options(xs))
    r = Record(f, xs, opts)
    xts, yt, tp = r.inputs, r.outputs, r.tape
    run_gradient_passes!(yt, tp)
    outs = map(adjoint, xts)
    empty!(tp)
    return outs
end

# gradient! #
#-----------#

function gradient!(out, f, x, opts::Options = Options(x, eltype(out)))
    r = Record(f, x, opts)
    xt, yt, tp = r.inputs, r.outputs, r.tape
    run_gradient_passes!(yt, tp)
    load_gradient!(out, yt, xt)
    empty!(tp)
    return out
end

function gradient!(outs::Tuple, f, xs::Tuple, opts::Options = Options(xs, eltype(first(outs))))
    r = Record(f, xs, opts)
    xts, yt, tp = r.inputs, r.outputs, r.tape
    run_gradient_passes!(yt, tp)
    for i in eachindex(outs)
        load_gradient!(outs[i], yt, xts[i])
    end
    empty!(tp)
    return outs
end

########################
# Gradient of `Record` #
########################

function gradient!(r::Record, x)
    xt, yt, tp = r.inputs, r.outputs, r.tape
    run_gradient_passes!(xt, yt, tp, x)
    return adjoint(xt)
end

function gradient!(r::Record, xs::Tuple)
    xts, yt, tp = r.inputs, r.outputs, r.tape
    run_gradient_passes!(xts, yt, tp, xs)
    return map(adjoint, xts)
end

function gradient!(out, r::Record, x)
    xt, yt, tp = r.inputs, r.outputs, r.tape
    run_gradient_passes!(xt, yt, tp, x)
    load_gradient!(out, yt, xt)
    return out
end

function gradient!(outs::Tuple, r::Record, xs::Tuple)
    xts, yt, tp = r.inputs, r.outputs, r.tape
    run_gradient_passes!(xts, yt, tp, xs)
    for i in eachindex(outs)
        load_gradient!(outs[i], yt, xts[i])
    end
    return outs
end

# utilities #
#-----------#

function run_gradient_passes!(yt, tp)
    if hastape(yt)
        seed!(yt)
        reverse_pass!(tp)
    end
end

function run_gradient_passes!(xt, yt, tp, x)
    if hastape(yt)
        unseed!(xt)
        setvalue!(xt, x)
        forward_pass!(tp)
        seed!(yt)
        reverse_pass!(tp)
    end
end

function run_gradient_passes!(xts::Tuple, yt, tp, xs::Tuple)
    if hastape(yt)
        for i in eachindex(xs)
            xt = xts[i]
            unseed!(xt)
            setvalue!(xt, xs[i])
        end
        forward_pass!(tp)
        seed!(yt)
        reverse_pass!(tp)
    end
end

load_gradient!(out, yt, xt) = adjoint!(out, xt)

function load_gradient!(out::DiffResult, yt, xt)
    DiffBase.value!(value, out, yt)
    DiffBase.gradient!(adjoint, out, xt)
    return out
end
