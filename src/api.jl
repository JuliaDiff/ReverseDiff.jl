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

function gradient(f, x, tp::Tape = Tape(), xtr = track(x, tp))
    result = f(xtr)
    seed!(result)
    backprop!(tp)
    return adjoint(xtr)
end

function gradient(f, xs::Tuple, tp::Tape = Tape(),
                  xtrs::Tuple = map(x -> track(x, tp), xs))
    result = f(xtrs...)
    seed!(result)
    backprop!(tp)
    return map(adjoint, xtrs)
end

# gradient! #
#-----------#

function gradient!(out, f, x, tp::Tape = Tape(), xtr = track(eltype(out), x, tp))
    result = f(xtr)
    seed!(result)
    backprop!(tp)
    return load_grad_result!(out, result, xtr)
end

function gradient!(outs::Tuple, f, xs::Tuple, tp::Tape = Tape(),
                   xtrs::Tuple = map((out, x) -> track(eltype(out), x, tp), outs, xs))
    result = f(xtrs...)
    seed!(result)
    backprop!(tp)
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

function load_jacobian!(out, xtr, y, tp::Tape)
    outmatrix = reshape(out, length(y), length(xtr))
    for i in eachindex(y)
        n = y[i]
        seed!(n)
        backprop!(tp)
        for j in eachindex(xtr)
            m = xtr[j]
            out[i, j] = adjoint(m)
        end
        unseed!(tp)
    end
    return out
end

load_jac_result!(out, xtr, ytr, tp) = load_jacobian!(out, xtr, ytr, tp)

function load_jac_result!(out::JacobianResult, xtr, ytr, tp)
    value!(out.value, ytr)
    load_jacobian!(out.jacobian, xtr, ytr, tp)
    return out
end

# jacobian #
#----------#

function jacobian(f, x, tp::Tape = Tape(), xtr = track(x, tp))
    ytr = f(xtr)
    out = similar(ytr, eltype(x), length(ytr), length(x))
    return load_jacobian!(out, xtr, ytr, tp)
end

function jacobian(f, xs::Tuple, tp::Tape = Tape(),
                  xtrs::Tuple = map(x -> track(x, tp), xs))
    ytr = f(xtrs...)
    outs = map(x -> similar(ytr, eltype(x), length(ytr), length(x)), xs)
    for i in eachindex(outs)
        load_jacobian!(outs[i], xtrs[i], ytr, tp)
    end
    return outs
end

# jacobian! #
#-----------#

function jacobian!(out, f, x, tp::Tape = Tape(), xtr = track(eltype(out), x, tp))
    tp = get(tape(first(xtr)))
    ytr = f(xtr)
    return load_jac_result!(out, xtr, ytr, tp)
end

function jacobian!(outs::Tuple, f, xs::Tuple, tp::Tape = Tape(),
                   xtrs::Tuple = map((out, x) -> track(eltype(out), x, tp), outs, xs))
   ytr = f(xtrs...)
   for i in eachindex(outs)
       load_jac_result!(outs[i], xtrs[i], ytr, tp)
   end
   return outs
end
