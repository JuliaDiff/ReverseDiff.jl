######################
# gradient/gradient! #
######################

load_grad_result!(out, result, xtr) = adjoint!(out, xtr)

function load_grad_result!(out::GradientResult, result, xtr)
    out.value = value(result)
    adjoint!(out.gradient, xtr)
    return out
end

function gradient(f, x, xtr = wrap(x))
    result = f(xtr)
    seed!(result)
    backprop!(get(trace(result)))
    return adjoint(xtr)
end

function gradient!(out, f, x, xtr = wrap(eltype(out), x))
    result = f(xtr)
    seed!(result)
    backprop!(get(trace(result)))
    return load_grad_result!(out, result, xtr)
end

######################
# jacobian/jacobian! #
######################

function load_jacobian!(out, xtr, y, tr::Trace)
    outmatrix = reshape(out, length(y), length(xtr))
    for i in eachindex(y)
        n = y[i]
        seed!(n)
        backprop!(tr)
        for j in eachindex(xtr)
            m = xtr[j]
            out[i, j] = adjoint(m)
        end
        unseed!(tr)
    end
    return out
end

load_jac_result!(out, xtr, ytr, tr) = load_jacobian!(out, xtr, ytr, tr)

function load_jac_result!(out::JacobianResult, xtr, ytr, tr)
    value!(out.value, ytr)
    load_jacobian!(out.jacobian, xtr, ytr, tr)
    return out
end

function jacobian(f, x, xtr = wrap(x))
    tr = get(trace(first(xtr)))
    ytr = f(xtr)
    out = similar(ytr, eltype(x), length(ytr), length(x))
    return load_jacobian!(out, xtr, ytr, tr)
end

function jacobian!(out, f, x, xtr = wrap(eltype(out), x))
    tr = get(trace(first(xtr)))
    ytr = f(xtr)
    return load_jac_result!(out, xtr, ytr, tr)
end

####################
# hessian/hessian! #
####################

hessian(f, x, xtr = wrap(x)) = jacobian(y -> gradient(f, y), x, xtr)

hessian!(out, f, x, xtr = wrap(x)) = jacobian!(out, y -> gradient(f, y), x, xtr)

function hessian!(out::HessianResult, f, x, xtr = wrap(x))
    outgrad = GradientResult(out.value, out.gradient)
    jacobian!(out.hessian, y -> gradient!(outgrad, y), x, xtr)
    out.value = outgrad.value
    return out
end
