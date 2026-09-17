module StatisticsExt

using ReverseDiff: ReverseDiff, SpecialInstruction, TrackedArray, deriv, increment_deriv!,
                   istracked, record!, tape, track, unseed!, value, value!
using Statistics: Statistics, mean

function Statistics.mean(x::TrackedArray{V,D}) where {V,D}
    tp = tape(x)
    out = track(mean(value(x)), D, tp)
    record!(tp, SpecialInstruction, mean, x, out)
    return out
end

@noinline function ReverseDiff.special_reverse_exec!(instruction::SpecialInstruction{typeof(mean)})
    input = instruction.input
    output = instruction.output
    istracked(input) && increment_deriv!(input, inv(length(input)) * deriv(output))
    unseed!(output)
    return nothing
end

@noinline function ReverseDiff.special_forward_exec!(instruction::SpecialInstruction{typeof(mean)})
    input = instruction.input
    value!(instruction.output, mean(value(input)))
    return nothing
end

end # module
