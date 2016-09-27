#####################################################
# Hessian of `f(::AbstractArray...)::AbstractArray` #
#####################################################

# hessian #
#---------#

function hessian(f, x, opts::HessianOptions = HessianOptions(x))
    ∇f = y -> gradient(f, y, opts.gradient_options)
    return jacobian(∇f, x, opts.jacobian_options)
end

# hessian! #
#----------#

function hessian!(out, f, x, opts::HessianOptions = HessianOptions(out, x))
    ∇f = y -> gradient(f, y, opts.gradient_options)
    return jacobian!(out, ∇f, x, opts.jacobian_options)
end

# function hessian!(out::DiffResult, f, x, opts::HessianOptions = Options(outs, Tape(), xs))
#     ∇f! = (y, z) -> begin
#         result = DiffResult(zero(eltype(y)), y)
#         gradient!(result, f, z, inner_opts)
#         DiffBase.value!(out, value(DiffBase.value(result)))
#         return y
#     end
#     jacobian!(DiffBase.hessian(out), ∇f!, DiffBase.gradient(out), x, outer_opts)
#     return out
# end
