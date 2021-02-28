######################################################
# Jacobian of `f(::AbstractArray...)::AbstractArray` #
######################################################

"""
    ReverseDiff.jacobian(f, input, cfg::JacobianConfig = JacobianConfig(input))

If `input` is an `AbstractArray`, assume `f` has the form
`f(::AbstractArray{<:Real})::AbstractArray{<:Real}` and return `J(f)(input)`.

If `input` is a tuple of `AbstractArray`s, assume `f` has the form
`f(::AbstractArray{<:Real}...)::AbstractArray{<:Real}` (such that it can be called as
`f(input...)`) and return a `Tuple` where the `i`th element is the  Jacobian of `f` w.r.t.
`input[i].`

Note that `cfg` can be preallocated and reused for subsequent calls.

If possible, it is highly recommended to use `ReverseDiff.JacobianTape` to prerecord `f`.
Otherwise, this method will have to re-record `f`'s execution trace for every subsequent
call.
"""
function jacobian(f, input, cfg::JacobianConfig = JacobianConfig(input))
    tape = JacobianTape(f, input, cfg)
    isa(input, TrackedArray) && empty!(input.tape)
    result = jacobian!(tape, input)
    empty!(tape.tape)
    return result
end

"""
    ReverseDiff.jacobian!(result, f, input, cfg::JacobianConfig = JacobianConfig(input))

Returns `result`. This method is exactly like `ReverseDiff.jacobian(f, input, cfg)`, except
it stores the resulting Jacobian(s) in `result` rather than allocating new memory.

`result` can be an `AbstractArray` or a `Tuple` of `AbstractArray`s. The `result` (or any
of its elements, if `isa(result, Tuple)`), can also be a `DiffResults.DiffResult`, in which
case the primal value `f(input)` (or `f(input...)`, if `isa(input, Tuple)`) will be stored
in it as well.
"""
function jacobian!(result, f, input, cfg::JacobianConfig = JacobianConfig(input))
    tape = JacobianTape(f, input, cfg)
    isa(input, TrackedArray) && empty!(input.tape)
    jacobian!(result, tape, input)
    empty!(tape.tape)
    return result
end

#########################################################
# Jacobian of `f!(::AbstractArray, ::AbstractArray...)` #
#########################################################

"""
    ReverseDiff.jacobian(f!, output, input, cfg::JacobianConfig = JacobianConfig(output, input))

Exactly like `ReverseDiff.jacobian(f, input, cfg)`, except the target function has the
form `f!(output::AbstractArray{<:Real}, input::AbstractArray{<:Real}...)`.
"""
function jacobian(f!, output, input, cfg::JacobianConfig = JacobianConfig(output, input))
    tape = JacobianTape(f!, output, input, cfg)
    isa(input, TrackedArray) && empty!(input.tape)
    result = jacobian!(tape, input)
    extract_result_value!(output, output_hook(tape))
    empty!(tape.tape)
    return result
end

"""
    ReverseDiff.jacobian!(result, f!, output, input, cfg::JacobianConfig = JacobianConfig(output, input))

Exactly like `ReverseDiff.jacobian!(result, f, input, cfg)`, except the target function has the
form `f!(output::AbstractArray{<:Real}, input::AbstractArray{<:Real}...)`.
"""
function jacobian!(result, f!, output, input, cfg::JacobianConfig = JacobianConfig(output, input))
    tape = JacobianTape(f!, output, input, cfg)
    isa(input, TrackedArray) && empty!(input.tape)
    jacobian!(result, tape, input)
    extract_result_value!(output, output_hook(tape))
    empty!(tape.tape)
    return result
end

###########################
# Executing JacobianTapes #
###########################

"""
    ReverseDiff.jacobian!(tape::Union{JacobianTape,CompiledJacobian}, input)

If `input` is an `AbstractArray`, assume `tape` represents a function of the form
`f(::AbstractArray{<:Real})::AbstractArray{<:Real}` or `f!(::AbstractArray{<:Real},
::AbstractArray{<:Real})` and return `tape`'s Jacobian w.r.t. `input`.

If `input` is a tuple of `AbstractArray`s, assume `tape` represents a function of the form
`f(::AbstractArray{<:Real}...)::AbstractArray{<:Real}` or `f!(::AbstractArray{<:Real},
::AbstractArray{<:Real}...)` and return a `Tuple` where the `i`th element is `tape`'s Jacobian
w.r.t. `input[i].`

Note that if `tape` represents a function of the form `f!(output, input...)`, you can only
execute `tape` with new `input` values. There is no way to re-run `tape`'s tape with new
`output` values; since `f!` can mutate `output`, there exists no stable "hook" for loading
new `output` values into the tape.
"""
function jacobian!(tape::Union{JacobianTape,CompiledJacobian}, input)
    result = construct_result(output_hook(tape), input_hook(tape))
    jacobian!(result, tape, input)
    return result
end

"""
    ReverseDiff.jacobian!(result, tape::Union{JacobianTape,CompiledJacobian}, input)

Returns `result`. This method is exactly like `ReverseDiff.jacobian!(tape, input)`, except it
stores the resulting Jacobian(s) in `result` rather than allocating new memory.

`result` can be an `AbstractArray` or a `Tuple` of `AbstractArray`s. The `result` (or any
of its elements, if `isa(result, Tuple)`), can also be a `DiffResults.DiffResult`, in which
case the primal value of the target function will be stored in it as well.
"""
function jacobian!(result, tape::Union{JacobianTape,CompiledJacobian}, input)
    seeded_forward_pass!(tape, input)
    seeded_reverse_pass!(result, tape)
    return result
end

##################################################
# unused (but faster) versions of the above code #
##################################################

#=
These commented-out versions of `jacobian` are faster than the ones we're
actually using above, because they avoid a redundant forward pass. This extra
forward pass should be unneccesary - since no input values are changing,
the record pass should be sufficient on its own. However, for some unknown
reason, getting rid of the superfluous forward pass breaks nested
differentation.
=#

# function jacobian(f, input, cfg::JacobianConfig = JacobianConfig(input))
#     tape = JacobianTape(f, input, cfg)
#     result = construct_result(output_hook(tape), input_hook(tape))
#     seeded_reverse_pass!(result, output_hook(tape), input_hook(tape), tape.tape)
#     empty!(tape.tape)
#     return result
# end
#
# function jacobian!(result, f, input, cfg::JacobianConfig = JacobianConfig(input))
#     tape = JacobianTape(f, input, cfg)
#     seeded_reverse_pass!(result, output_hook(tape), input_hook(tape), tape.tape)
#     empty!(tape.tape)
#     return result
# end
#
# function jacobian(f!, output, input, cfg::JacobianConfig = JacobianConfig(output, input))
#     tape = JacobianTape(f!, output, input, cfg)
#     result = construct_result(output_hook(tape), input_hook(tape))
#     seeded_reverse_pass!(result, output_hook(tape), input_hook(tape), tape.tape)
#     extract_result_value!(output, output_hook(tape))
#     empty!(tape.tape)
#     return result
# end
#
# function jacobian!(result, f!, output, input, cfg::JacobianConfig = JacobianConfig(output, input))
#     tape = JacobianTape(f!, output, input, cfg)
#     seeded_reverse_pass!(result, output_hook(tape), input_hook(tape), tape.tape)
#     extract_result_value!(output, output_hook(tape))
#     empty!(tape.tape)
#     return result
# end
