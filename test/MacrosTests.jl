module MacrosTests

using ReverseDiff, ForwardDiff, Test, StaticArrays
using ForwardDiff: Dual, Partials, partials

include(joinpath(dirname(@__FILE__), "utils.jl"))

tp = InstructionTape()
x, a, b = rand(3)

############
# @forward #
############

f0(x) = 1. / (1. + exp(-x))
f0(a, b) = sqrt(a^2 + b^2)

ReverseDiff.@forward f1(x::T) where {T<:Real} = 1. / (1. + exp(-x))
ReverseDiff.@forward f1(a::A, b::B) where {A,B<:Real} = sqrt(a^2 + b^2)

ReverseDiff.@forward f2(x) = 1. / (1. + exp(-x))
ReverseDiff.@forward f2(a, b) = sqrt(a^2 + b^2)

ReverseDiff.@forward function f3(x::T) where T<:Real
    return 1. / (1. + exp(-x))
end

ReverseDiff.@forward function f3(a::A, b::B) where {A,B<:Real}
    return sqrt(a^2 + b^2)
end

ReverseDiff.@forward function f4(x)
    return 1. / (1. + exp(-x))
end

ReverseDiff.@forward function f4(a, b)
    return sqrt(a^2 + b^2)
end

function test_forward(f, x, tp)
    xt = ReverseDiff.TrackedReal(x, zero(x), tp)

    y = f(x)
    @test isempty(tp)

    yt = f(xt)
    @test yt == y
    dual = f(Dual(x, one(x)))
    @test length(tp) == 1
    instruction = first(tp)
    @test typeof(instruction) <: ReverseDiff.ScalarInstruction
    @test instruction.input === xt
    @test instruction.output === yt
    @test instruction.cache[] === partials(dual, 1)
    empty!(tp)
end

function test_forward(f, a, b, tp)
    at = ReverseDiff.TrackedReal(a, zero(a), tp)
    bt = ReverseDiff.TrackedReal(b, zero(b), tp)

    c = f(a, b)
    dual = f(Dual(a, one(a), zero(a)), Dual(b, zero(b), one(b)))
    @test isempty(tp)

    tc = f(at, b)
    @test tc == c
    @test length(tp) == 1
    instruction = first(tp)
    @test typeof(instruction) <: ReverseDiff.ScalarInstruction
    @test instruction.input === (at, b)
    @test instruction.output === tc
    @test instruction.cache[] === SVector(partials(dual, 1), partials(dual, 1))
    empty!(tp)

    tc = f(a, bt)
    @test tc == c
    @test length(tp) == 1
    instruction = first(tp)
    @test typeof(instruction) <: ReverseDiff.ScalarInstruction
    @test instruction.input === (a, bt)
    @test instruction.output === tc
    @test instruction.cache[] === SVector(partials(dual, 2), partials(dual, 2))
    empty!(tp)

    tc = f(at, bt)
    @test tc == c
    @test length(tp) == 1
    instruction = first(tp)
    @test typeof(instruction) <: ReverseDiff.ScalarInstruction
    @test instruction.input === (at, bt)
    @test instruction.output === tc
    @test instruction.cache[] === SVector(partials(dual)...)
    empty!(tp)
end

for f in (ReverseDiff.@forward(f0), f1, f2, f3, f4, ReverseDiff.@forward(-))
    test_println("@forward named functions", f)
    test_forward(f, x, tp)
    test_forward(f, a, b, tp)
end

ReverseDiff.@forward f5 = (x) -> 1. / (1. + exp(-x))
test_println("@forward anonymous functions", f5)
test_forward(f5, x, tp)

ReverseDiff.@forward f6 = (a, b) -> sqrt(a^2 + b^2)
test_println("@forward anonymous functions", f6)
test_forward(f6, a, b, tp)

#########
# @skip #
#########

g0 = f0

ReverseDiff.@skip g1(x::T) where {T<:Real} = 1. / (1. + exp(-x))
ReverseDiff.@skip g1(a::A, b::B) where {A,B<:Real} = sqrt(a^2 + b^2)

ReverseDiff.@skip g2(x) = 1. / (1. + exp(-x))
ReverseDiff.@skip g2(a, b) = sqrt(a^2 + b^2)

ReverseDiff.@skip function g3(x::T) where T<:Real
    return 1. / (1. + exp(-x))
end

ReverseDiff.@skip function g3(a::A, b::B) where {A,B<:Real}
    return sqrt(a^2 + b^2)
end

ReverseDiff.@skip function g4(x)
    return 1. / (1. + exp(-x))
end

ReverseDiff.@skip function g4(a, b)
    return sqrt(a^2 + b^2)
end

function test_skip(g, x, tp)
    xt = ReverseDiff.TrackedReal(x, zero(x), tp)

    y = g(x)
    @test isempty(tp)

    yt = g(xt)
    @test yt === y
    @test isempty(tp)
end

function test_skip(g, a, b, tp)
    at = ReverseDiff.TrackedReal(a, zero(a), tp)
    bt = ReverseDiff.TrackedReal(b, zero(b), tp)

    c = g(a, b)
    @test isempty(tp)

    tc = g(at, b)
    @test tc === c
    @test isempty(tp)

    tc = g(a, bt)
    @test tc === c
    @test isempty(tp)

    tc = g(at, bt)
    @test tc === c
    @test isempty(tp)
end

for g in (ReverseDiff.@skip(g0), g1, g2, g3, g4)
    test_println("@skip named functions", g)
    test_skip(g, x, tp)
    test_skip(g, a, b, tp)
end

ReverseDiff.@skip g5 = (x) -> 1. / (1. + exp(-x))
test_println("@skip anonymous functions", g5)
test_skip(g5, x, tp)

ReverseDiff.@skip g6 = (a, b) -> sqrt(a^2 + b^2)
test_println("@skip anonymous functions", g6)
test_skip(g6, a, b, tp)

end # module
