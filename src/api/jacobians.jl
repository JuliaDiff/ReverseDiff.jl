######################################################
# Jacobian of `f(::AbstractArray...)::AbstractArray` #
######################################################

function jacobian(f, input, cfg::JacobianConfig = JacobianConfig(input))
    rec = JacobianRecord(f, input, cfg)
    isa(input, TrackedArray) && empty!(input.tape)
    result = jacobian!(rec, input)
    empty!(rec.tape)
    return result
end

function jacobian!(result, f, input, cfg::JacobianConfig = JacobianConfig(input))
    rec = JacobianRecord(f, input, cfg)
    isa(input, TrackedArray) && empty!(input.tape)
    jacobian!(result, rec, input)
    empty!(rec.tape)
    return result
end

#########################################################
# Jacobian of `f!(::AbstractArray, ::AbstractArray...)` #
#########################################################

function jacobian(f!, output, input, cfg::JacobianConfig = JacobianConfig(output, input))
    rec = JacobianRecord(f!, output, input, cfg)
    isa(input, TrackedArray) && empty!(input.tape)
    result = jacobian!(rec, input)
    extract_result_value!(output, rec.output)
    empty!(rec.tape)
    return result
end

function jacobian!(result, f!, output, input, cfg::JacobianConfig = JacobianConfig(output, input))
    rec = JacobianRecord(f!, output, input, cfg)
    isa(input, TrackedArray) && empty!(input.tape)
    jacobian!(result, rec, input)
    extract_result_value!(output, rec.output)
    empty!(rec.tape)
    return result
end

#############################
# Executing JacobianRecords #
#############################

#=
We can't support changing `y` values for recorded `f!(y, x)` because, in general, our
tracked `y` input will get dereferenced/mutated such that the tracked `y` values only
reference output instructions, not input instructions. Thus, we have no "hook" into `y`
values as we do with the `x` values.
=#

function jacobian!(rec::Union{JacobianRecord,CompiledJacobian}, input)
    result = construct_result(rec.output, rec.input)
    jacobian!(result, rec, input)
    return result
end

function jacobian!(result, rec::Union{JacobianRecord,CompiledJacobian}, input)
    seeded_forward_pass!(rec, input)
    seeded_reverse_pass!(result, rec)
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
#     rec = JacobianRecord(f, input, cfg)
#     result = construct_result(rec.output, rec.input)
#     seeded_reverse_pass!(result, rec.output, rec.input, rec.tape)
#     empty!(rec.tape)
#     return result
# end
#
# function jacobian!(result, f, input, cfg::JacobianConfig = JacobianConfig(input))
#     rec = JacobianRecord(f, input, cfg)
#     seeded_reverse_pass!(result, rec.output, rec.input, rec.tape)
#     empty!(rec.tape)
#     return result
# end
#
# function jacobian(f!, output, input, cfg::JacobianConfig = JacobianConfig(output, input))
#     rec = JacobianRecord(f!, output, input, cfg)
#     result = construct_result(rec.output, rec.input)
#     seeded_reverse_pass!(result, rec.output, rec.input, rec.tape)
#     extract_result_value!(output, rec.output)
#     empty!(rec.tape)
#     return result
# end
#
# function jacobian!(result, f!, output, input, cfg::JacobianConfig = JacobianConfig(output, input))
#     rec = JacobianRecord(f!, output, input, cfg)
#     seeded_reverse_pass!(result, rec.output, rec.input, rec.tape)
#     extract_result_value!(output, rec.output)
#     empty!(rec.tape)
#     return result
# end
