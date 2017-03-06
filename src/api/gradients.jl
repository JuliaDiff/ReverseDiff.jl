#############################################
# Gradient of `f(::AbstractArray...)::Real` #
#############################################

"""
    ReverseDiff.gradient(f, input, cfg::GradientConfig = GradientConfig(input))

If `input` is an `AbstractArray`, assume `f` has the form `f(::AbstractArray{<:Real})::Real`
and return `∇f(input)`.

If `input` is a tuple of `AbstractArray`s, assume `f` has the form
`f(::AbstractArray{<:Real}...)::Real` (such that it can be called as `f(input...)`) and return
a `Tuple` where the `i`th element is the gradient of `f` w.r.t. `input[i].`

Note that `cfg` can be preallocated and reused for subsequent calls.

If possible, it is highly recommended to use `ReverseDiff.GradientTape` to prerecord `f`.
Otherwise, this method will have to re-record `f`'s execution trace for every subsequent
call.
"""
function gradient(f, input, cfg::GradientConfig = GradientConfig(input))
    tape = GradientTape(f, input, cfg)
    result = construct_result(input_hook(tape))
    seeded_reverse_pass!(result, tape)
    empty!(cfg.tape)
    return result
end

"""
    ReverseDiff.gradient!(result, f, input, cfg::GradientConfig = GradientConfig(input))

Returns `result`. This method is exactly like `ReverseDiff.gradient(f, input, cfg)`, except
it stores the resulting gradient(s) in `result` rather than allocating new memory.

`result` can be an `AbstractArray` or a `Tuple` of `AbstractArray`s. The `result` (or any
of its elements, if `isa(result, Tuple)`), can also be a `DiffBase.DiffResult`, in which
case the primal value `f(input)` (or `f(input...)`, if `isa(input, Tuple)`) will be stored
in it as well.
"""
function gradient!(result, f, input, cfg::GradientConfig = GradientConfig(input))
    tape = GradientTape(f, input, cfg)
    seeded_reverse_pass!(result, tape)
    empty!(cfg.tape)
    return result
end

###########################
# Executing GradientTapes #
###########################

"""
    ReverseDiff.gradient!(tape::Union{GradientTape,CompiledGradient}, input)

If `input` is an `AbstractArray`, assume `tape` represents a function of the form
`f(::AbstractArray)::Real` and return `∇f(input)`.

If `input` is a tuple of `AbstractArray`s, assume `tape` represents a function of the form
`f(::AbstractArray...)::Real` and return a `Tuple` where the `i`th element is the gradient
of `f` w.r.t. `input[i].`
"""
function gradient!(tape::Union{GradientTape,CompiledGradient}, input)
    result = construct_result(input_hook(tape))
    gradient!(result, tape, input)
    return result
end

"""
    ReverseDiff.gradient!(result, tape::Union{GradientTape,CompiledGradient}, input)

Returns `result`. This method is exactly like `ReverseDiff.gradient!(tape, input)`, except it
stores the resulting gradient(s) in `result` rather than allocating new memory.

`result` can be an `AbstractArray` or a `Tuple` of `AbstractArray`s. The `result` (or any
of its elements, if `isa(result, Tuple)`), can also be a `DiffBase.DiffResult`, in which
case the primal value `f(input)` (or `f(input...)`, if `isa(input, Tuple)`) will be stored
in it as well.
"""
function gradient!(result, tape::Union{GradientTape,CompiledGradient}, input)
    seeded_forward_pass!(tape, input)
    seeded_reverse_pass!(result, tape)
    return result
end
