#######################
# seeded reverse pass #
#######################

# derivative (input is an scalar, output is an array/scalar) #
#------------------------------------------------------------#

function seeded_reverse_pass!(result::AbstractArray, output::AbstractArray, input::TrackedReal, tape)
    result_vector = reshape(result, length(output))
    for i in eachindex(output)
        result_vector[i] = seeded_reverse_pass!(output[i], input, tape)
    end
    return result
end

function seeded_reverse_pass!(output::TrackedReal, input::TrackedReal, tape)
    pull_value!(output)
    unseed!(input)
    seed!(output)
    reverse_pass!(tape)
    return deriv(input)
end

# gradient (input is an array, output is a scalar) #
#--------------------------------------------------#

function seeded_reverse_pass!(result, output::TrackedReal, input, tape)
    pull_value!(output)
    unseed!(input)
    seed!(output)
    reverse_pass!(tape)
    result = extract_result!(result, output, input)
    return result
end

function seeded_reverse_pass!(result, output::Number, input, tape)
    result = extract_result!(result, output)
    return result
end

# jacobian (input and output are both arrays) #
#---------------------------------------------#

function seeded_reverse_pass!(result::AbstractArray, output::AbstractArray, input::TrackedArray, tape)
    result_matrix = reshape(result, length(output), length(input))
    input_deriv = deriv(input)
    pull_value!(output)
    for i in eachindex(output)
        unseed!(input)
        seed!(output, i)
        reverse_pass!(tape)
        for j in eachindex(input)
            result_matrix[i, j] = input_deriv[j]
        end
        unseed!(output, i)
    end
    return result
end

function seeded_reverse_pass!(result::DiffResult, output::AbstractArray, input::TrackedArray, tape)
    result = seeded_reverse_pass!(DiffResults.jacobian(result), output, input, tape)
    result = extract_result_value!(result, output)
    return result
end

function seeded_reverse_pass!(result::Tuple, output::AbstractArray, input::Tuple, tape)
    result = map(eachindex(result, input)) do i
        seeded_reverse_pass!(result[i], output, input[i], tape)
    end
    return result
end

#####################
# result extraction #
#####################

function extract_result!(result::Tuple, output, input::Tuple)
    result = map(eachindex(result, input)) do i
        extract_result!(result[i], output, input[i])
    end
    return result
end

function extract_result!(result::Tuple, output)
    result = map(eachindex(result)) do i
        extract_result!(result[i], output)
    end
    return result
end

function extract_result!(result::AbstractArray, output::TrackedReal, input::TrackedArray)
    copyto!(result, deriv(input))
    return result
end

function extract_result!(result::DiffResult, output::TrackedReal, input::TrackedArray)
    result = DiffResults.value!(result, value(output))
    copyto!(DiffResults.gradient(result), deriv(input))
    return result
end

function extract_result!(result::AbstractArray, output::Number)
    fill_zeros!(result)
    return result
end

function extract_result!(result::DiffResult, output::Number)
    result = DiffResults.value!(result, output)
    fill_zeros!(DiffResults.gradient(result))
    return result
end

function extract_result_value!(result::Tuple, output)
    result = map(eachindex(result)) do i
        extract_result_value!(result[i], output)
    end
    return result
end

function extract_result_value!(result::DiffResult, output::AbstractArray)
    result = DiffResults.value!(value, result, output)
    return result
end

function extract_result_value!(result::DiffResult, output::TrackedArray)
    result = DiffResults.value!(result, value(output))
    return result
end

function extract_result_value!(result::AbstractArray, output::AbstractArray)
    map!(value, result, output)
    return result
end

function extract_result_value!(result::AbstractArray, output::TrackedArray)
    copyto!(result, value(output))
    return result
end

fill_zeros!(result::AbstractArray) = fill!(result, zero(eltype(result)))

#######################
# result construction #
#######################

construct_result(output::AbstractArray, input::Tuple) = map(x -> construct_result(output, x), input)

construct_result(output::AbstractArray, input::TrackedArray) = similar(deriv(input), length(output), length(input))

construct_result(input::TrackedArray) = similar(deriv(input))

construct_result(input::Tuple) = map(construct_result, input)
