#######################
# seeded reverse pass #
#######################

# derivative (input is an scalar, output is an array/scalar) #
#------------------------------------------------------------#

function seeded_reverse_pass!(result::AbstractArray, output::AbstractArray, input::TrackedReal, tp::Tape)
    result_vector = reshape(result, length(output))
    for i in eachindex(output)
        result_vector[i] = seeded_reverse_pass!(output[i], input, tp)
    end
    return result
end

function seeded_reverse_pass!(output::TrackedReal, input::TrackedReal, tp::Tape)
    seed!(output)
    unseed!(input)
    reverse_pass!(tp)
    unseed!(output)
    return deriv(input)
end

# gradient (input is an array, output is a scalar) #
#--------------------------------------------------#

function seeded_reverse_pass!(result, output::TrackedReal, input, tp::Tape)
    seed!(output)
    unseed!(input)
    reverse_pass!(tp)
    extract_result!(result, output, input)
    return result
end

function seeded_reverse_pass!(result, output::Number, input, tp::Tape)
    extract_result!(result, output)
    return result
end

# jacobian (input and output are both arrays) #
#---------------------------------------------#

function seeded_reverse_pass!(result::AbstractArray, output::AbstractArray, input::TrackedArray, tp::Tape)
    result_matrix = reshape(result, length(output), length(input))
    input_deriv = deriv(input)
    for i in eachindex(output)
        output_element = output[i]
        seed!(output_element)
        unseed!(input)
        reverse_pass!(tp)
        for j in eachindex(input)
            result_matrix[i, j] = input_deriv[j]
        end
        unseed!(output_element)
    end
    return result
end

function seeded_reverse_pass!(result::DiffResult, output::AbstractArray, input::TrackedArray, tp::Tape)
    seeded_reverse_pass!(DiffBase.jacobian(result), output, input, tp)
    extract_result_value!(result, output)
    return result
end

function seeded_reverse_pass!(result::Tuple, output::AbstractArray, input::Tuple, tp::Tape)
    for i in eachindex(result)
        seeded_reverse_pass!(result[i], output, input[i], tp)
    end
    return result
end

#####################
# result extraction #
#####################

function extract_result!(result::Tuple, output, input::Tuple)
    for i in eachindex(result)
        extract_result!(result[i], output, input[i])
    end
    return result
end

function extract_result!(result::Tuple, output)
    for i in eachindex(result)
        extract_result!(result[i], output)
    end
    return result
end

function extract_result!(result::AbstractArray, output::TrackedReal, input::TrackedArray)
    copy!(result, deriv(input))
    return result
end

function extract_result!(result::DiffResult, output::TrackedReal, input::TrackedArray)
    DiffBase.value!(result, value(output))
    copy!(DiffBase.gradient(result), deriv(input))
    return result
end

function extract_result!(result::AbstractArray, output::Number)
    fill_zeros!(result)
    return result
end

function extract_result!(result::DiffResult, output::Number)
    DiffBase.value!(result, output)
    fill_zeros!(DiffBase.gradient(result))
    return result
end

function extract_result_value!(result::Tuple, output)
    for i in eachindex(result)
        extract_result_value!(result[i], output)
    end
    return result
end

function extract_result_value!(result::DiffResult, output::AbstractArray)
    DiffBase.value!(value, result, output)
    return result
end

function extract_result_value!(result::DiffResult, output::TrackedArray)
    DiffBase.value!(result, value(output))
    return result
end

function extract_result_value!(result::AbstractArray, output::AbstractArray)
    map!(value, result, output)
    return result
end

function extract_result_value!(result::AbstractArray, output::TrackedArray)
    copy!(result, value(output))
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
