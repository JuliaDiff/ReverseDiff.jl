#############################################
# Gradient of `f(::AbstractArray...)::Real` #
#############################################

function gradient(f, input, cfg::GradientConfig = GradientConfig(input))
    rec = GradientRecord(f, input, cfg)
    result = construct_result(rec.input)
    seeded_reverse_pass!(result, rec.output, rec.input, rec.tape)
    empty!(rec.tape)
    return result
end

function gradient!(result, f, input, cfg::GradientConfig = GradientConfig(input))
    rec = GradientRecord(f, input, cfg)
    seeded_reverse_pass!(result, rec.output, rec.input, rec.tape)
    empty!(rec.tape)
    return result
end

#############################
# Executing GradientRecords #
#############################

function gradient!(rec::GradientRecord, input)
    result = construct_result(rec.input)
    gradient!(result, rec, input)
    return result
end

function gradient!(result, rec::GradientRecord, input)
    value!(rec.input, input)
    forward_pass!(rec.tape)
    seeded_reverse_pass!(result, rec.output, rec.input, rec.tape)
    return result
end
