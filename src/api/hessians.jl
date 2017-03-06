##################################################
# Hessian of `f(::AbstractArray)::AbstractArray` #
##################################################

# hessian #
#---------#

"""
    ReverseDiff.hessian(f, input::AbstractArray, cfg::HessianConfig = HessianConfig(input))

Given `f(input::AbstractArray{<:Real})::Real`, return `f`s Hessian w.r.t. to the given
`input`.

Note that `cfg` can be preallocated and reused for subsequent calls.

If possible, it is highly recommended to use `ReverseDiff.HessianTape` to prerecord `f`.
Otherwise, this method will have to re-record `f`'s execution trace for every subsequent
call.
"""
function hessian(f, input::AbstractArray, cfg::HessianConfig = HessianConfig(input))
    ∇f = x -> gradient(f, x, cfg.gradient_config)
    return jacobian(∇f, input, cfg.jacobian_config)
end

# hessian! #
#----------#

"""
    ReverseDiff.hessian!(result::AbstractArray, f, input::AbstractArray, cfg::HessianConfig = HessianConfig(input))

    ReverseDiff.hessian!(result::DiffResult, f, input::AbstractArray, cfg::HessianConfig = HessianConfig(result, input))

Returns `result`. This method is exactly like `ReverseDiff.hessian(f, input, cfg)`, except
it stores the resulting Hessian in `result` rather than allocating new memory.

If `result` is a `DiffBase.DiffResult`, the primal value `f(input)` and the gradient
`∇f(input)` will be stored in it along with the Hessian `H(f)(input)`.
"""
function hessian!(result, f, input::AbstractArray, cfg::HessianConfig = HessianConfig(input))
    ∇f = x -> gradient(f, x, cfg.gradient_config)
    jacobian!(result, ∇f, input, cfg.jacobian_config)
    return result
end

function hessian!(result::DiffResult, f, input::AbstractArray,
                  cfg::HessianConfig = HessianConfig(result, input))
    ∇f! = (y, x) -> begin
        gradient_result = DiffResult(zero(eltype(y)), y)
        gradient!(gradient_result, f, x, cfg.gradient_config)
        DiffBase.value!(result, value(DiffBase.value(gradient_result)))
        return y
    end
    jacobian!(DiffBase.hessian(result), ∇f!,
              DiffBase.gradient(result), input,
              cfg.jacobian_config)
    return result
end

##########################
# Executing HessianTapes #
##########################

"""
    ReverseDiff.hessian!(tape::Union{HessianTape,CompiledHessian}, input)

Assuming `tape` represents a function of the form `f(::AbstractArray{<:Real})::Real`,
return the Hessian `H(f)(input)`.
"""
function hessian!(tape::Union{HessianTape,CompiledHessian}, input::AbstractArray)
    result = construct_result(output_hook(tape), input_hook(tape))
    hessian!(result, tape, input)
    return result
end

"""
    ReverseDiff.hessian!(result::AbstractArray, tape::Union{HessianTape,CompiledHessian}, input)

    ReverseDiff.hessian!(result::DiffResult, tape::Union{HessianTape,CompiledHessian}, input)

Returns `result`. This method is exactly like `ReverseDiff.hessian!(tape, input)`, except
it stores the resulting Hessian in `result` rather than allocating new memory.

If `result` is a `DiffBase.DiffResult`, the primal value `f(input)` and the gradient
`∇f(input)` will be stored in it along with the Hessian `H(f)(input)`.
"""
function hessian!(result::AbstractArray, tape::Union{HessianTape,CompiledHessian}, input::AbstractArray)
    seeded_forward_pass!(tape, input)
    seeded_reverse_pass!(result, tape)
    return result
end

function hessian!(result::DiffResult, tape::Union{HessianTape,CompiledHessian}, input::AbstractArray)
    seeded_forward_pass!(tape, input)
    seeded_reverse_pass!(DiffResult(DiffBase.gradient(result), DiffBase.hessian(result)), tape)
    DiffBase.value!(result, tape.func(input))
    return result
end

######################
# Hessian API Errors #
######################

const HESS_MULTI_ARG_ERR_MSG = "Taking the Hessian of a function with multiple arguments is not yet supported"

hessian(f, xs::Tuple, ::HessianConfig) = error(HESS_MULTI_ARG_ERR_MSG)
hessian(f, xs::Tuple) = error(HESS_MULTI_ARG_ERR_MSG)
hessian!(outs::Tuple, f, xs::Tuple, ::HessianConfig) = error(HESS_MULTI_ARG_ERR_MSG)
hessian!(outs::Tuple, f, xs::Tuple) = error(HESS_MULTI_ARG_ERR_MSG)
