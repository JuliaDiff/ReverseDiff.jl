######################################################
# Jacobian of `f(::AbstractArray...)::AbstractArray` #
######################################################

# jacobian #
#----------#

function jacobian(f, x, opts::Options = Options(x))
    r = Record(f, x, opts)
    xt, yt, tp = r.inputs, r.outputs, r.tape
    out = construct_jacobian_output(yt, xt)
    jacobian_reverse_pass!(out, yt, xt, tp)
    empty!(tp)
    return out
end

function jacobian(f, xs::Tuple, opts::Options = Options(xs))
    r = Record(f, xs, opts)
    xts, yt, tp = r.inputs, r.outputs, r.tape
    outs = construct_jacobian_output(yt, xts)
    jacobian_reverse_pass!(outs, yt, xts, tp)
    empty!(tp)
    return outs
end

# jacobian! #
#-----------#

function jacobian!(out, f, x, opts::Options = Options(x, eltype(out)))
    r = Record(f, x, opts)
    xt, yt, tp = r.inputs, r.outputs, r.tape
    jacobian_reverse_pass!(out, yt, xt, tp)
    jacobian_extract_value!(out, yt)
    empty!(tp)
    return out
end

function jacobian!(outs::Tuple, f, xs::Tuple, opts::Options = Options(xs, eltype(first(outs))))
    r = Record(f, xs, opts)
    xts, yt, tp = r.inputs, r.outputs, r.tape
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
    r = Record(f!, y, x, opts)
    xt, yt, tp = r.inputs, r.outputs, r.tape
    out = construct_jacobian_output(yt, xt)
    jacobian_reverse_pass!(out, yt, xt, tp)
    map!(value, y, yt)
    empty!(tp)
    return out
end

function jacobian(f!, y, xs::Tuple, opts::Options = Options(y, xs))
    r = Record(f!, y, xs, opts)
    xts, yt, tp = r.inputs, r.outputs, r.tape
    outs = construct_jacobian_output(yt, xts)
    jacobian_reverse_pass!(outs, yt, xts, tp)
    map!(value, y, yt)
    empty!(tp)
    return out
end

# jacobian! #
#-----------#

function jacobian!(out, f!, y, x, opts::Options = Options(y, x))
    r = Record(f!, y, x, opts)
    xt, yt, tp = r.inputs, r.outputs, r.tape
    jacobian_reverse_pass!(out, yt, xt, tp)
    jacobian_extract_value!(out, y, yt)
    empty!(tp)
    return out
end

function jacobian!(outs::Tuple, f!, y, xs::Tuple, opts::Options = Options(y, xs))
    r = Record(f!, y, xs, opts)
    xts, yt, tp = r.inputs, r.outputs, r.tape
    for i in eachindex(outs)
        out = outs[i]
        jacobian_reverse_pass!(out, yt, xts[i], tp)
        jacobian_extract_value!(out, y, yt)
    end
    empty!(tp)
    return outs
end

########################
# Jacobian of `Record` #
########################

# We can't support changing `y` values for recorded `f!(y, x)` because, in
# general, our tracked `y` input will get dereferenced/mutated such that
# the tracked `y` values only reference output nodes, not input nodes. Thus,
# we have no "hook" into `y` values as we do with the `x` values.

function jacobian!(r::Record, x)
    out = construct_jacobian_output(r.outputs, r.inputs)
    return jacobian!(out, r, x)
end

function jacobian!(r::Record, xs::Tuple)
    outs = construct_jacobian_output(r.outputs, r.inputs)
    return jacobian!(outs, r, xs)
end

function jacobian!(out, r::Record, x)
    xt, yt, tp = r.inputs, r.outputs, r.tape
    run_jacobian_passes!(out, xt, yt, tp, x)
    return out
end

function jacobian!(outs::Tuple, r::Record, xs::Tuple)
    xts, yt, tp = r.inputs, r.outputs, r.tape
    run_jacobian_passes!(outs, xts, yt, tp, xs)
    return outs
end

#############
# Utilities #
#############

function run_jacobian_passes!(out, xt, yt, tp, x)
    unseed!(xt)
    setvalue!(xt, x)
    forward_pass!(tp)
    jacobian_reverse_pass!(out, yt, xt, tp)
    jacobian_extract_value!(out, yt)
end

function run_jacobian_passes!(outs::Tuple, xts::Tuple, yt, tp, xs::Tuple)
    for i in eachindex(xts)
        xt = xts[i]
        unseed!(xt)
        setvalue!(xt, xs[i])
    end
    forward_pass!(tp)
    for i in eachindex(outs)
        out = outs[i]
        jacobian_reverse_pass!(out, yt, xts[i], tp)
        jacobian_extract_value!(out, yt)
    end
end

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

function jacobian_reverse_pass!(outs::Tuple, yt, xts::Tuple, tp::Tape)
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
