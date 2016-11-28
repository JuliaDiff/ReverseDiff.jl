#############################################
# Gradient of `f(::AbstractArray...)::Real` #
#############################################

function gradient(f, input, cfg::GradientConfig = GradientConfig(input))
    rec = GradientRecord(f, input, cfg)
    result = construct_result(rec.input)
    seeded_reverse_pass!(result, rec)
    empty!(cfg.tape)
    return result
end

function gradient!(result, f, input, cfg::GradientConfig = GradientConfig(input))
    rec = GradientRecord(f, input, cfg)
    seeded_reverse_pass!(result, rec)
    empty!(cfg.tape)
    return result
end

#############################
# Executing GradientRecords #
#############################

function gradient!(rec::Union{GradientRecord,CompiledGradient}, input)
    result = construct_result(rec.input)
    gradient!(result, rec, input)
    return result
end

function gradient!(result, rec::Union{GradientRecord,CompiledGradient}, input)
    seeded_forward_pass!(rec, input)
    seeded_reverse_pass!(result, rec)
    return result
end
