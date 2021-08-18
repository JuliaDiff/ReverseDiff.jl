module ChainRulesTest

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

ReverseDiff.@grad_from_chainrules f

begin # test ChainRules.rrule
    input = rand(3, 3)
    output, back = ChainRules.rrule(f, input);
    _, d = back(1)
    @test output == f(input)
    @test d == fill(3, size(input))
end

begin # test ReverseDiff
    const f_tape = ReverseDiff.GradientTape(x -> f(x) + 2, (rand(3, 3),))
    inputs = (rand(3, 3), )
    results = (similar(inputs[1]),)

    ReverseDiff.gradient!(results, f_tape, inputs)

    @test results[1] == fill(3, size(inputs[1]))
end

end
