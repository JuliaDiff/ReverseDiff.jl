##################################################
# Hessian of `f(::AbstractArray)::AbstractArray` #
##################################################

# hessian #
#---------#

function hessian(f, x, opts::HessianOptions = HessianOptions(x))
    ∇f = y -> gradient(f, y, gradient_options(opts))
    return jacobian(∇f, x, jacobian_options(opts))
end

# hessian! #
#----------#

function hessian!(out, f, x, opts::HessianOptions = HessianOptions(x))
    ∇f = y -> gradient(f, y, gradient_options(opts))
    jacobian!(out, ∇f, x, jacobian_options(opts))
    return out
end

function hessian!(out::DiffResult, f, x, opts::HessianOptions = HessianOptions(out, x))
    ∇f! = (y, z) -> begin
        result = DiffResult(zero(eltype(y)), y)
        gradient!(result, f, z, gradient_options(opts))
        DiffBase.value!(out, value(DiffBase.value(result)))
        return y
    end
    jacobian!(DiffBase.hessian(out), ∇f!, DiffBase.gradient(out), x, jacobian_options(opts))
    return out
end

#####################################################
# Hessian of `f(::AbstractArray...)::AbstractArray` #
#####################################################

const HESS_ERR_MSG = "Taking the Hessian of a function with multiple arguments is not yet supported"

hessian(f, xs::Tuple, args...) = error(HESS_ERR_MSG)
hessian!(outs::Tuple, f, xs::Tuple, args...) = error(HESS_ERR_MSG)
