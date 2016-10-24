######################################################
# Jacobian of `f(::AbstractArray...)::AbstractArray` #
######################################################

# jacobian #
#----------#

function jacobian(f, x, opts::Options = Options(x))
    xt, tp = opts.state, opts.tape
    track!(xt, x, tp)
    yt = f(xt)
    out = construct_jacobian_output(yt, xt)
    jacobian_reverse_pass!(out, yt, xt, tp)
    empty!(tp)
    return out
end

function jacobian(f, xs::Tuple, opts::Options = Options(xs))
    xts, tp = opts.state, opts.tape
    track!(xts, xs, tp)
    yt = f(xts...)
    outs = construct_jacobian_output(yt, xts)
    jacobian_reverse_pass!(outs, yt, xts, tp)
    empty!(tp)
    return out
end

# jacobian! #
#-----------#

function jacobian!(out, f, x, opts::Options = Options(x, eltype(out)))
    xt, tp = opts.state, opts.tape
    track!(xt, x, tp)
    yt = f(xt)
    jacobian_reverse_pass!(out, yt, xt, tp)
    jacobian_extract_value!(out, yt)
    empty!(tp)
    return out
end

function jacobian!(outs::Tuple, f, xs::Tuple, opts::Options = Options(xs, map(eltype, outs)))
    xts, tp = opts.state, opts.tape
    track!(xts, xs, tp)
    yt = f(xts...)
    for i in eachindex(outs)
        out = outs[i]
        jacobian_reverse_pass!(out, yt, xts[i], tp)
        jacobian_extract_value!(out, yt)
    end
    empty!(tp)
    return outs
end

#########################################################
# Jacobian of `f!(::AbstractArray, ::AbstractArray...)` #
#########################################################

# jacobian #
#----------#

function jacobian(f!, y, x, opts::Options = Options(y, x))
    yt, xt = opts.state
    tp = opts.tape
    track!(yt, y, tp)
    track!(xt, x, tp)
    f!(yt, xt)
    out = construct_jacobian_output(yt, xt)
    jacobian_reverse_pass!(out, yt, xt, tp)
    map!(value, y, yt)
    empty!(tp)
    return out
end

function jacobian(f!, y, xs::Tuple, opts::Options = Options(y, xs))
    yt, xts = opts.state
    tp = opts.tape
    track!(yt, y, tp)
    track!(xts, xs, tp)
    f!(yt, xts...)
    outs = construct_jacobian_output(yt, xts)
    jacobian_reverse_pass!(outs, yt, xts, tp)
    map!(value, y, yt)
    empty!(tp)
    return out
end

# jacobian! #
#-----------#

function jacobian!(out, f!, y, x, opts::Options = Options(y, x))
    yt, xt = opts.state
    tp = opts.tape
    track!(yt, y, tp)
    track!(xt, x, tp)
    f!(yt, xt)
    jacobian_reverse_pass!(out, yt, xt, tp)
    jacobian_extract_value!(out, y, yt)
    empty!(tp)
    return out
end

function jacobian!(outs::Tuple, f!, y, xs::Tuple, opts::Options = Options(y, xs))
    yt, xts = opts.state
    tp = opts.tape
    track!(yt, y, tp)
    track!(xts, xs, tp)
    f!(yt, xts...)
    for i in eachindex(outs)
        out = outs[i]
        jacobian_reverse_pass!(out, yt, xts[i], tp)
        jacobian_extract_value!(out, y, yt)
    end
    empty!(tp)
    return outs
end

#############
# Utilities #
#############

function jacobian_reverse_pass!(outs::Tuple, yts::Tuple, xts::Tuple, tp)
    error("Taking the Jacobian of a function which returns multiple arrays is not yet supported.")
end

function jacobian_reverse_pass!(out::DiffResult, yt, xt, tp::Tape)
    return jacobian_reverse_pass!(DiffBase.jacobian(out), yt, xt, tp)
end

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

function jacobian_reverse_pass!(outs::Tuple, yt, xts::Tuple, tp)
    for i in eachindex(outs)
        jacobian_reverse_pass!(outs[i], yt, xts[i], tp)
    end
    return outs
end

construct_jacobian_output(yt, xts::Tuple) = map(xt -> construct_jacobian_output(yt, xt), xts)
construct_jacobian_output(yt, xt) = similar(yt, valtype(eltype(yt)), length(yt), length(xt))

jacobian_extract_value!(out::DiffResult, yt) = DiffBase.value!(value, out, yt)
jacobian_extract_value!(out, yt) = nothing

function jacobian_extract_value!(out, y, yt)
    map!(value, y, yt)
    jacobian_copy_value!(out, y)
end

jacobian_copy_value!(out::DiffResult, y) = DiffBase.value!(out, y)
jacobian_copy_value!(out, y) = nothing
