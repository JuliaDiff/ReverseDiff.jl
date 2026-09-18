module ChainRulesTest

using LinearAlgebra
using ChainRulesCore
using DiffResults
using ReverseDiff
using Test

struct MyStruct end
f(::MyStruct, x) = sum(4x .+ 1)
f(x, y::MyStruct) = sum(4x .+ 1)
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
        return NoTangent(), fill(3 * d, size(x))
    end
    return r, back
end
function ChainRulesCore.rrule(::typeof(f), ::MyStruct, x)
    r = f(MyStruct(), x)
    function back(d)
        return NoTangent(), NoTangent(), fill(3 * d, size(x))
    end
    return r, back
end
function ChainRulesCore.rrule(::typeof(f), x, ::MyStruct)
    r = f(x, MyStruct())
    function back(d)
        return NoTangent(), fill(3 * d, size(x)), NoTangent()
    end
    return r, back
end

ReverseDiff.@grad_from_chainrules f(x::ReverseDiff.TrackedArray)
# test arg type hygiene
ReverseDiff.@grad_from_chainrules f(::MyStruct, x::ReverseDiff.TrackedArray)
ReverseDiff.@grad_from_chainrules f(x::ReverseDiff.TrackedArray, y::MyStruct)

g(x, y) = sum(4x .+ 4y)

function ChainRulesCore.rrule(::typeof(g), x, y)
    r = g(x, y)
    function back(d)
        # same as above, use 3 and 5 as the derivatives
        return NoTangent(), fill(3 * d, size(x)), fill(5 * d, size(x))
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

@testset "custom struct input" begin
    input = rand(3, 3)
    output, back = ChainRulesCore.rrule(f, MyStruct(), input);
    _, _, d = back(1)
    @test output == f(MyStruct(), input)
    @test d == fill(3, size(input))

    output, back = ChainRulesCore.rrule(f, input, MyStruct());
    _, d, _ = back(1)
    @test output == f(input, MyStruct())
    @test d == fill(3, size(input))
end

### Tape test
@testset "Tape test: Ensure ordinary call is not tracked" begin
    tp = ReverseDiff.InstructionTape()

    f(x) = sum(2x .+ g([1, 2], [3, 4]))
    x = rand(3, 3)
    xt = ReverseDiff.track(copy(x), tp)
    # record
    yt = f(xt)
    @test length(tp) == 3 # sum, broadcast+, broadcast*, but not `g`
end

### Functions with varargs and kwargs
# Varargs
f_vararg(x, args...) = sum(4x .+ sum(args))

function ChainRulesCore.rrule(::typeof(f_vararg), x, args...)
    r = f_vararg(x, args...)
    function back(d)
        return NoTangent(), fill(3 * d, size(x))
    end
    return r, back
end

ReverseDiff.@grad_from_chainrules f_vararg(x::ReverseDiff.TrackedArray, args...)

@testset "Function with Varargs" begin
    inputs = (rand(3, 3), )

    results = (similar(inputs[1]),)
    f_tape = ReverseDiff.GradientTape(x -> f_vararg(x, 1, 2, 3) + 2, (rand(3, 3),))
    ReverseDiff.gradient!(results, f_tape, inputs)

    @test results[1] == fill(3, size(inputs[1]))
end


# Vargs and kwargs
f_kw(x, args...; k=1, kwargs...) = sum(4x .+ sum(args) .+ (k + kwargs[:j]))

function ChainRulesCore.rrule(::typeof(f_kw), x, args...; k=1, kwargs...)
    r = f_kw(x, args...; k=k, kwargs...)
    function back(d)
        return NoTangent(), fill(3 * d, size(x))
    end
    return r, back
end

ReverseDiff.@grad_from_chainrules f_kw(x::ReverseDiff.TrackedArray, args...; k=1, kwargs...)

@testset "Function with Varargs and kwargs" begin
    inputs = (rand(3, 3), )

    results = (similar(inputs[1]),)
    f_tape = ReverseDiff.GradientTape(x -> f_kw(x, 1, 2, 3; k=2, j=3) + 2, (rand(3, 3),))
    ReverseDiff.gradient!(results, f_tape, inputs)

    @test results[1] == fill(3, size(inputs[1]))
end

### Mix @grad and @grad_from_chainrules

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

### Thunked tangents
# Lazy tangents have to be unthunked: `increment_deriv!` only accepts arrays and reals.

f_thunk(x) = sum(4x .+ 1)
function ChainRulesCore.rrule(::typeof(f_thunk), x)
    # `InplaceableThunk` is what ChainRules' matrix rules hand back
    pb_f_thunk(Δ) = (NoTangent(), InplaceableThunk(ȳ -> ȳ .+= 3Δ, @thunk(fill(3Δ, size(x)))))
    return f_thunk(x), pb_f_thunk
end
ReverseDiff.@grad_from_chainrules f_thunk(x::ReverseDiff.TrackedArray)

f_thunk(x::Real) = 4x + 1
function ChainRulesCore.rrule(::typeof(f_thunk), x::Real)
    pb_f_thunk_scalar(Δ) = (NoTangent(), @thunk(3Δ))
    return f_thunk(x), pb_f_thunk_scalar
end
ReverseDiff.@grad_from_chainrules f_thunk(x::ReverseDiff.TrackedReal)

# `@grad_from_chainrules` dispatches on `TrackedArray`, so only `@grad` reaches an array
# that merely *holds* tracked elements
g_thunk(a, b) = sum(a) + sum(b)
function g_thunk(a::AbstractVector, b::AbstractVector)
    if ReverseDiff.istracked(a) || ReverseDiff.istracked(b)
        return ReverseDiff.track(g_thunk, a, b)
    else
        return sum(a) + sum(b)
    end
end
ReverseDiff.@grad function g_thunk(a, b)
    av, bv = ReverseDiff.value(a), ReverseDiff.value(b)
    pb_g_thunk(Δ) = (@thunk(fill(3Δ, size(av))), @thunk(fill(5Δ, size(bv))))
    return sum(av) + sum(bv), pb_g_thunk
end

@testset "thunked tangents" begin
    x = rand(3, 3)
    @test ReverseDiff.gradient(f_thunk, x) == fill(3, size(x))
    @test ReverseDiff.gradient(u -> f_thunk(u[1]), [2.0]) == [3.0]

    # the thunk for the untracked argument is dropped rather than forced
    @test ReverseDiff.gradient(u -> g_thunk(Real[u[1], u[2]], [1.0, 2.0]), [1.0, 2.0]) == [3.0, 3.0]
    @test ReverseDiff.gradient(u -> g_thunk([1.0, 2.0], Real[u[1], u[1]]), [1.0, 2.0]) == [10.0, 0.0]
end

### Isolated Scope
module IsolatedModuleForTestingScoping
using ChainRulesCore
using ReverseDiff: ReverseDiff, @grad_from_chainrules

f(x) = sum(4x .+ 1)

function ChainRulesCore.rrule(::typeof(f), x)
    r = f(x)
    function back(d)
        # return a distinguishable but improper grad
        return NoTangent(), fill(3 * d, size(x))
    end
    return r, back
end

@grad_from_chainrules f(x::ReverseDiff.TrackedArray)

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

end # end of SubModule
end # end of IsolatedModuleForTestingScoping

end
