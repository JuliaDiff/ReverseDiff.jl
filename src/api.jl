Base.eltype{V}(::GradientResult{V}) = V
Base.eltype{V,G}(::Type{GradientResult{V,G}}) = V
Base.eltype{V}(::JacobianResult{V}) = eltype(V)
Base.eltype{V,J}(::Type{JacobianResult{V,J}}) = eltype(V)

#############################################
# Gradient of `f(::AbstractArray...)::Real` #
#############################################

# utilities #
#-----------#

load_grad_result!(out, result, xtr) = adjoint!(out, xtr)

function load_grad_result!(out::GradientResult, result, xtr)
    out.value = value(result)
    adjoint!(out.gradient, xtr)
    return out
end

# gradient #
#----------#

function gradient(f, x, tr::Trace = Trace(), xtr = wrap(x, tr))
    result = f(xtr)
    seed!(result)
    backprop!(tr)
    return adjoint(xtr)
end

function gradient(f, xs::Tuple, tr::Trace = Trace(),
                  xtrs::Tuple = map(x -> wrap(x, tr), xs))
    result = f(xtrs...)
    seed!(result)
    backprop!(tr)
    return map(adjoint, xtrs)
end

# gradient! #
#-----------#

function gradient!(out, f, x, tr::Trace = Trace(), xtr = wrap(eltype(out), x, tr))
    result = f(xtr)
    seed!(result)
    backprop!(tr)
    return load_grad_result!(out, result, xtr)
end

function gradient!(outs::Tuple, f, xs::Tuple, tr::Trace = Trace(),
                   xtrs::Tuple = map((out, x) -> wrap(eltype(out), x, tr), outs, xs))
    result = f(xtrs...)
    seed!(result)
    backprop!(tr)
    for i in eachindex(outs)
        load_grad_result!(outs[i], result, xtrs[i])
    end
    return outs
end

######################################################
# Jacobian of `f(::AbstractArray...)::AbstractArray` #
######################################################

# utilities #
#-----------#

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

# jacobian #
#----------#

function jacobian(f, x, tr::Trace = Trace(), xtr = wrap(x, tr))
    ytr = f(xtr)
    out = similar(ytr, eltype(x), length(ytr), length(x))
    return load_jacobian!(out, xtr, ytr, tr)
end

function jacobian(f, xs::Tuple, tr::Trace = Trace(),
                  xtrs::Tuple = map(x -> wrap(x, tr), xs))
    ytr = f(xtrs...)
    outs = map(x -> similar(ytr, eltype(x), length(ytr), length(x)), xs)
    for i in eachindex(outs)
        load_jacobian!(outs[i], xtrs[i], ytr, tr)
    end
    return outs
end

# jacobian! #
#-----------#

function jacobian!(out, f, x, tr::Trace = Trace(), xtr = wrap(eltype(out), x, tr))
    tr = get(trace(first(xtr)))
    ytr = f(xtr)
    return load_jac_result!(out, xtr, ytr, tr)
end

function jacobian!(outs::Tuple, f, xs::Tuple, tr::Trace = Trace(),
                   xtrs::Tuple = map((out, x) -> wrap(eltype(out), x, tr), outs, xs))
   ytr = f(xtrs...)
   for i in eachindex(outs)
       load_jac_result!(outs[i], xtrs[i], ytr, tr)
   end
   return outs
end
