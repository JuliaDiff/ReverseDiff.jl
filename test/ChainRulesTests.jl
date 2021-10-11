module ChainRulesTest

using LinearAlgebra
using ChainRulesCore
using ChainRules
using DiffResults
using ReverseDiff
using Test

f(x) = sum(4x .+ 1)

function ChainRulesCore.rrule(::typeof(f), x)
    r = f(x)
    function back(d)
        #=
        The proper derivative of `f` is 4, but in order to
        check if `ChainRulesCore.rrule` had taken over the compuation,
        we define a rrule that returns 3 as `f`'s derivative.

        After importing this rrule into ReverseDiff, if we get 3
        rather than 4 when we compute the derivative of `f`, it means
        the importing mechanism works.
        =#
        return ChainRulesCore.NoTangent(), fill(3 * d, size(x))
    end
    return r, back
end

ReverseDiff.@grad_from_chainrules f(x::ReverseDiff.TrackedArray)


g(x, y) = sum(4x .+ 4y)

function ChainRulesCore.rrule(::typeof(g), x, y)
    r = g(x, y)
    function back(d)
        # same as above, use 3 and 5 as the derivatives
        return ChainRulesCore.NoTangent(), fill(3 * d, size(x)), fill(5 * d, size(x))
    end
    return r, back
end

ReverseDiff.@grad_from_chainrules g(x::ReverseDiff.TrackedArray, y)
ReverseDiff.@grad_from_chainrules g(x, y::ReverseDiff.TrackedArray)
ReverseDiff.@grad_from_chainrules g(x::ReverseDiff.TrackedArray, y::ReverseDiff.TrackedArray)

@testset "rrule in ChainRules and ReverseDiff" begin
    ## ChainRules
    # function f
    input = rand(3, 3)
    output, back = ChainRulesCore.rrule(f, input);
    _, d = back(1)
    @test output == f(input)
    @test d == fill(3, size(input))
    # function g
    inputs = rand(3, 3), rand(3, 3)
    output, back = ChainRulesCore.rrule(g, inputs...);
    _, d1, d2 = back(1)
    @test output == g(inputs...)
    @test d1 == fill(3, size(inputs[1]))
    @test d2 == fill(5, size(inputs[2]))


    ## ReverseDiff
    #function f
    inputs = (rand(3, 3), )

    results = (similar(inputs[1]),)
    f_tape = ReverseDiff.GradientTape(x -> f(x) + 2, (rand(3, 3),))
    ReverseDiff.gradient!(results, f_tape, inputs)

    @test results[1] == fill(3, size(inputs[1]))

    results = (similar(inputs[1]),)
    compiled_tape = ReverseDiff.CompiledTape(f_tape)
    ReverseDiff.gradient!(results, compiled_tape, inputs)
    @test results[1] == fill(3, size(inputs[1]))

    # function g
    inputs = rand(3, 3), rand(3, 3)

    results = (similar(inputs[1]), similar(inputs[2]))
    f_tape = ReverseDiff.GradientTape((x, y) -> g(x, y) + 2, (rand(3, 3), rand(3, 3)))
    ReverseDiff.gradient!(results, f_tape, inputs)

    @test results[1] == fill(3, size(inputs[1]))
    @test results[2] == fill(5, size(inputs[2]))

    results = (similar(inputs[1]), similar(inputs[2]),)
    compiled_tape = ReverseDiff.CompiledTape(f_tape)
    ReverseDiff.gradient!(results, compiled_tape, inputs)
    @test results[1] == fill(3, size(inputs[1]))
    @test results[2] == fill(5, size(inputs[2]))

end

### Functions from ChainRules

# import rrule from ChainRules
ReverseDiff.@grad_from_chainrules LinearAlgebra.norm1(x::ReverseDiff.TrackedArray)

@testset "test imported rrules" begin
    inputs = (rand(3, 3), )
    results = (similar(inputs[1]),)

    g = (x) -> LinearAlgebra.norm1(x)
    g_tape = ReverseDiff.GradientTape(g, (rand(3, 3),))
    ReverseDiff.gradient!(results, g_tape, inputs)
    @test results[1] == fill(1, size(inputs[1]))
end

## Mix @grad and @grad_from_chainrules

h(x) = 10x
h(x::ReverseDiff.TrackedArray) = ReverseDiff.track(h, x)
ReverseDiff.@grad function h(x)
    xv = ReverseDiff.value(x)
    return h(xv), Δ -> (Δ * 7,) # use 7 asits derivatives
end

@testset "ReverseDiff and ChainRules Mixed" begin
    t(x) = g(x, h(x))
    inputs = (rand(3, 3), )
    results = (similar(inputs[1]),)

    g_tape = ReverseDiff.GradientTape(t, (rand(3, 3),))
    ReverseDiff.gradient!(results, g_tape, inputs)
    @test results[1] == fill(38, size(inputs[1])) # 38 = 3 + 5 * 7
end

module IsolatedModuleForTestingScoping
using ChainRulesCore
using ReverseDiff: @grad_from_chainrules

f(x) = sum(4x .+ 1)

function ChainRulesCore.rrule(::typeof(f), x)
    r = f(x)
    function back(d)
        # return a distinguishable but improper grad
        return ChainRulesCore.NoTangent(), fill(3 * d, size(x))
    end
    return r, back
end

@grad_from_chainrules f(x::TrackedArray)

module SubModule
using Test
using ReverseDiff: TrackedArray, GradientTape, gradient!
using ..IsolatedModuleForTestingScoping: f
@testset "rrule in Isolated Scope" begin
    inputs = (rand(3, 3), )

    results = (similar(inputs[1]),)
    f_tape = GradientTape(x -> f(x) + 2, (rand(3, 3),))
    gradient!(results, f_tape, inputs)

    @test results[1] == fill(3, size(inputs[1]))
end

end # end if SubModule
end # end of IsolatedModuleForTestingScoping

end
