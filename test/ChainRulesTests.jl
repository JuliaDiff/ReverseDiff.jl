module ChainRulesTest

using LinearAlgebra
using ChainRules
using DiffResults
using ReverseDiff
using Test

f(x) = sum(4x .+ 1)

function ChainRules.rrule(::typeof(f), x)
    r = f(x)
    function back(d)
        #=
        The proper derivative of `f` is 4, but in order to
        check if `ChainRules.rrule` had taken over the compuation,
        we define a rrule that returns 3 as `f`'s derivative.

        After importing this rrule into ReverseDiff, if we get 3
        rather than 4 when we compute the derivative of `f`, it means
        the importing mechanism works.
        =#
        return ChainRules.NoTangent(), fill(3 * d, size(x))
    end
    return r, back
end

ReverseDiff.@grad_from_chainrules f(::ReverseDiff.TrackedArray)

@testset "rrule in ChainRules and ReverseDiff" begin
    # ChainRules
    input = rand(3, 3)
    output, back = ChainRules.rrule(f, input);
    _, d = back(1)
    @test output == f(input)
    @test d == fill(3, size(input))

    # ReverseDiff
    inputs = (rand(3, 3), )

    results = (similar(inputs[1]),)
    f_tape = ReverseDiff.GradientTape(x -> f(x) + 2, (rand(3, 3),))
    ReverseDiff.gradient!(results, f_tape, inputs)

    @test results[1] == fill(3, size(inputs[1]))

    results = (similar(inputs[1]),)
    compiled_tape = ReverseDiff.CompiledTape(f_tape)
    ReverseDiff.gradient!(results, compiled_tape, inputs)
    @test results[1] == fill(3, size(inputs[1]))
end

### Functions from ChainRules

# import rrule from ChainRules
const FUNCS_FROM_CHAINRULES = [
    :(LinearAlgebra.norm1(::ReverseDiff.TrackedArray)),
]

for func in FUNCS_FROM_CHAINRULES
    @eval ReverseDiff.@grad_from_chainrules $func
end

@testset "test imported rrules" begin
    inputs = (rand(3, 3), )
    results = (similar(inputs[1]),)

    g = (x) -> LinearAlgebra.norm1(x)
    g_tape = ReverseDiff.GradientTape(g, (rand(3, 3),))
    ReverseDiff.gradient!(results, g_tape, inputs)
    @test results[1] == fill(1, size(inputs[1]))
end

end
