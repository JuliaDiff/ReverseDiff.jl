##################################################
# Hessian of `f(::AbstractArray)::AbstractArray` #
##################################################

# hessian #
#---------#

function hessian(f, x::AbstractArray, opts::HessianOptions = HessianOptions(x))
    ∇f = y -> gradient(f, y, gradient_options(opts))
    return jacobian(∇f, x, jacobian_options(opts))
end

# hessian! #
#----------#

function hessian!(out, f, x::AbstractArray, opts::HessianOptions = HessianOptions(x))
    ∇f = y -> gradient(f, y, gradient_options(opts))
    jacobian!(out, ∇f, x, jacobian_options(opts))
    return out
end

function hessian!(out::DiffResult, f, x::AbstractArray, opts::HessianOptions = HessianOptions(out, x))
    ∇f! = (y, z) -> begin
        result = DiffResult(zero(eltype(y)), y)
        gradient!(result, f, z, gradient_options(opts))
        DiffBase.value!(out, value(DiffBase.value(result)))
        return y
    end
    jacobian!(DiffBase.hessian(out), ∇f!, DiffBase.gradient(out), x, jacobian_options(opts))
    return out
end

##############################
# Hessian of `HessianRecord` #
##############################

function hessian!(r::HessianRecord, x::AbstractArray)
    jr = jacobian_record(r)
    xt, yt = jr.inputs, jr.outputs
    out = similar(x, length(yt), length(xt))
    return hessian!(out, r, x)
end

for T in (:AbstractArray, :DiffResult) # done to avoid ambiguity errors
    @eval function hessian!(out::$T, r::HessianRecord, x::AbstractArray)
        jr = jacobian_record(r)
        gr = gradient_record(r)
        xt, yt, jtp = jr.inputs, jr.outputs, jr.tape
        xtt, ytt, gtp = gr.inputs, gr.outputs, gr.tape
        setvalue!(xt, x)
        run_gradient_passes!(xtt, ytt, gtp, xt)
        hessian_extract_value!(out, ytt)
        adjoint!(yt, xtt)
        unseed!(xt)
        forward_pass!(jtp)
        hessian_reverse_pass!(out, yt, xt, jtp)
        return out
    end
end

hessian_extract_value!(out, ytt) = nothing
hessian_extract_value!(out::DiffResult, ytt) = DiffBase.value!(out, value(value(ytt)))

hessian_reverse_pass!(out, yt, xt, jtp) = jacobian_reverse_pass!(out, yt, xt, jtp)

function hessian_reverse_pass!(out::DiffResult, yt, xt, jtp)
    result = DiffResult(DiffBase.gradient(out), DiffBase.hessian(out))
    jacobian_reverse_pass!(result, yt, xt, jtp)
    jacobian_extract_value!(result, yt)
    return out
end

######################
# Hessian API Errors #
######################

const HESS_MULTI_ARG_ERR_MSG = "Taking the Hessian of a function with multiple arguments is not yet supported"

hessian(f, xs::Tuple, ::HessianOptions) = error(HESS_MULTI_ARG_ERR_MSG)
hessian(f, xs::Tuple) = error(HESS_MULTI_ARG_ERR_MSG)
hessian!(outs::Tuple, f, xs::Tuple, ::HessianOptions) = error(HESS_MULTI_ARG_ERR_MSG)
hessian!(outs::Tuple, f, xs::Tuple) = error(HESS_MULTI_ARG_ERR_MSG)

const HESS_RECORD_ERR_MSG = "To take the Hessian of a recorded function, use `HessianRecord` instead of `Record`."

hessian(r::Record, x, ::HessianOptions) = error(HESS_RECORD_ERR_MSG)
hessian(r::Record, x) = error(HESS_RECORD_ERR_MSG)
hessian!(out, r::Record, x, ::HessianOptions) = error(HESS_RECORD_ERR_MSG)
hessian!(out, r::Record, x) = error(HESS_RECORD_ERR_MSG)

const HESS_OPTIONS_ERR_MSG = "To take a Hessian with options, use `HessianOptions` instead of `Options`."

hessian(f, x, ::Options) = error(HESS_OPTIONS_ERR_MSG)
hessian!(out, f, x, ::Options) = error(HESS_OPTIONS_ERR_MSG)
