using ReverseDiff, ForwardDiff, Test, StaticArrays, DiffResults
using ForwardDiff: Dual, Partials, partials

include(joinpath(dirname(@__FILE__), "utils.jl"))

tp = InstructionTape()
x, a, b = rand(3)

############
# @forward #
############

f0(x) = 1.0 / (1.0 + exp(-x))
f0(a, b) = sqrt(a^2 + b^2)

ReverseDiff.@forward f1(x::T) where {T <: Real} = 1.0 / (1.0 + exp(-x))
ReverseDiff.@forward f1(a::A, b::B) where {A, B <: Real} = sqrt(a^2 + b^2)

ReverseDiff.@forward f2(x) = 1.0 / (1.0 + exp(-x))
ReverseDiff.@forward f2(a, b) = sqrt(a^2 + b^2)

ReverseDiff.@forward function f3(x::T) where {T <: Real}
    return 1.0 / (1.0 + exp(-x))
end

ReverseDiff.@forward function f3(a::A, b::B) where {A, B <: Real}
    return sqrt(a^2 + b^2)
end

ReverseDiff.@forward function f4(x)
    return 1.0 / (1.0 + exp(-x))
end

ReverseDiff.@forward function f4(a, b)
    return sqrt(a^2 + b^2)
end

function test_forward(f, x, tp)
    xt = ReverseDiff.TrackedReal(x, zero(x), tp)

    y = f(x)
    @test isempty(take_recorded!(tp))

    yt = f(xt)
    @test yt == y
    dual = f(Dual(x, one(x)))
    recorded = take_recorded!(tp)
    @test length(recorded) == 1
    instruction = first(recorded)
    @test typeof(instruction) <: ReverseDiff.ScalarInstruction
    @test instruction.input === xt
    @test instruction.output === yt
    @test instruction.cache[] === partials(dual, 1)
    return nothing
end

function test_forward(f, a, b, tp)
    at = ReverseDiff.TrackedReal(a, zero(a), tp)
    bt = ReverseDiff.TrackedReal(b, zero(b), tp)

    c = f(a, b)
    dual = f(Dual(a, one(a), zero(a)), Dual(b, zero(b), one(b)))
    @test isempty(take_recorded!(tp))

    tc = f(at, b)
    @test tc == c
    recorded = take_recorded!(tp)
    @test length(recorded) == 1
    instruction = first(recorded)
    @test typeof(instruction) <: ReverseDiff.ScalarInstruction
    @test instruction.input === (at, b)
    @test instruction.output === tc
    @test instruction.cache[] === SVector(partials(dual, 1), partials(dual, 1))

    tc = f(a, bt)
    @test tc == c
    recorded = take_recorded!(tp)
    @test length(recorded) == 1
    instruction = first(recorded)
    @test typeof(instruction) <: ReverseDiff.ScalarInstruction
    @test instruction.input === (a, bt)
    @test instruction.output === tc
    @test instruction.cache[] === SVector(partials(dual, 2), partials(dual, 2))

    tc = f(at, bt)
    @test tc == c
    recorded = take_recorded!(tp)
    @test length(recorded) == 1
    instruction = first(recorded)
    @test typeof(instruction) <: ReverseDiff.ScalarInstruction
    @test instruction.input === (at, bt)
    @test instruction.output === tc
    @test instruction.cache[] === SVector(partials(dual)...)
    return nothing
end

for f in (ReverseDiff.@forward(f0), f1, f2, f3, f4, ReverseDiff.@forward(-))
    test_println("@forward named functions", f)
    test_forward(f, x, tp)
    test_forward(f, a, b, tp)
end

ReverseDiff.@forward f5 = (x) -> 1.0 / (1.0 + exp(-x))
test_println("@forward anonymous functions", f5)
test_forward(f5, x, tp)

ReverseDiff.@forward f6 = (a, b) -> sqrt(a^2 + b^2)
test_println("@forward anonymous functions", f6)
test_forward(f6, a, b, tp)

#########
# @skip #
#########

g0 = f0

ReverseDiff.@skip g1(x::T) where {T <: Real} = 1.0 / (1.0 + exp(-x))
ReverseDiff.@skip g1(a::A, b::B) where {A, B <: Real} = sqrt(a^2 + b^2)

ReverseDiff.@skip g2(x) = 1.0 / (1.0 + exp(-x))
ReverseDiff.@skip g2(a, b) = sqrt(a^2 + b^2)

ReverseDiff.@skip function g3(x::T) where {T <: Real}
    return 1.0 / (1.0 + exp(-x))
end

ReverseDiff.@skip function g3(a::A, b::B) where {A, B <: Real}
    return sqrt(a^2 + b^2)
end

ReverseDiff.@skip function g4(x)
    return 1.0 / (1.0 + exp(-x))
end

ReverseDiff.@skip function g4(a, b)
    return sqrt(a^2 + b^2)
end

function test_skip(g, x, tp)
    xt = ReverseDiff.TrackedReal(x, zero(x), tp)

    y = g(x)
    @test isempty(take_recorded!(tp))

    yt = g(xt)
    @test yt === y
    @test isempty(take_recorded!(tp))
    return nothing
end

function test_skip(g, a, b, tp)
    at = ReverseDiff.TrackedReal(a, zero(a), tp)
    bt = ReverseDiff.TrackedReal(b, zero(b), tp)

    c = g(a, b)
    @test isempty(take_recorded!(tp))

    tc = g(at, b)
    @test tc === c
    @test isempty(take_recorded!(tp))

    tc = g(a, bt)
    @test tc === c
    @test isempty(take_recorded!(tp))

    tc = g(at, bt)
    @test tc === c
    @test isempty(take_recorded!(tp))
    return nothing
end

for g in (ReverseDiff.@skip(g0), g1, g2, g3, g4)
    test_println("@skip named functions", g)
    test_skip(g, x, tp)
    test_skip(g, a, b, tp)
end

ReverseDiff.@skip g5 = (x) -> 1.0 / (1.0 + exp(-x))
test_println("@skip anonymous functions", g5)
test_skip(g5, x, tp)

ReverseDiff.@skip g6 = (a, b) -> sqrt(a^2 + b^2)
test_println("@skip anonymous functions", g6)
test_skip(g6, a, b, tp)

#########
# @grad #
#########

using LinearAlgebra
using ReverseDiff: @grad, TrackedReal, TrackedVector, TrackedMatrix, TrackedArray

@testset "@grad macro" begin
    x = rand(3)
    A = rand(3, 3)
    A_x = [vec(A); x]
    global custom_grad_called

    f1(x) = dot(x, x)
    f1(x::TrackedVector) = ReverseDiff.track(f1, x)
    @grad function f1(x::AbstractVector)
        global custom_grad_called = true
        xv = ReverseDiff.value(x)
        dot(xv, xv), Δ -> (Δ * 2 * xv,)
    end

    custom_grad_called = false
    g1 = ReverseDiff.gradient(f1, x)
    g2 = ReverseDiff.gradient(x -> dot(x, x), x)
    @test g1 == g2
    @test custom_grad_called

    f2(A, x) = A * x
    f2(A, x::TrackedVector) = ReverseDiff.track(f2, A, x)
    f2(A::TrackedMatrix, x) = ReverseDiff.track(f2, A, x)
    f2(A::TrackedMatrix, x::TrackedVector) = ReverseDiff.track(f2, A, x)
    @grad function f2(A::AbstractMatrix, x::AbstractVector)
        global custom_grad_called = true
        Av = ReverseDiff.value(A)
        xv = ReverseDiff.value(x)
        Av * xv, Δ -> (Δ * xv', Av' * Δ)
    end

    custom_grad_called = false
    g1 = ReverseDiff.gradient(x -> sum(f2(A, x)), x)
    g2 = ReverseDiff.gradient(x -> sum(A * x), x)
    @test g1 == g2
    @test custom_grad_called

    custom_grad_called = false
    g1 = ReverseDiff.gradient(A -> sum(f2(A, x)), A)
    g2 = ReverseDiff.gradient(A -> sum(A * x), A)
    @test g1 == g2
    @test custom_grad_called

    custom_grad_called = false
    g1 = ReverseDiff.gradient(A_x -> sum(f2(reshape(A_x[1:9], 3, 3), A_x[10:end])), A_x)
    g2 = ReverseDiff.gradient(A_x -> sum(reshape(A_x[1:9], 3, 3) * A_x[10:end]), A_x)
    @test g1 == g2
    @test custom_grad_called

    f3(A; dims) = sum(A, dims = dims)
    f3(A::TrackedMatrix; dims) = ReverseDiff.track(f3, A; dims = dims)
    @grad function f3(A::AbstractMatrix; dims)
        global custom_grad_called = true
        Av = ReverseDiff.value(A)
        sum(Av, dims = dims), Δ -> (zero(Av) .+ Δ,)
    end
    custom_grad_called = false
    g1 = ReverseDiff.gradient(A -> sum(f3(A, dims = 1)), A)
    g2 = ReverseDiff.gradient(A -> sum(sum(A, dims = 1)), A)
    @test g1 == g2
    @test custom_grad_called

    f4(::typeof(log), A; dims) = sum(log, A, dims = dims)
    f4(::typeof(log), A::TrackedMatrix; dims) = ReverseDiff.track(f4, log, A; dims = dims)
    @grad function f4(::typeof(log), A::AbstractMatrix; dims)
        global custom_grad_called = true
        Av = ReverseDiff.value(A)
        sum(log, Av, dims = dims), Δ -> (nothing, 1 ./ Av .* Δ)
    end
    custom_grad_called = false
    g1 = ReverseDiff.gradient(A -> sum(f4(log, A, dims = 1)), A)
    g2 = ReverseDiff.gradient(A -> sum(sum(log, A, dims = 1)), A)
    @test g1 == g2
    @test custom_grad_called

    f5(x) = log(x)
    f5(x::TrackedReal) = ReverseDiff.track(f5, x)
    @grad function f5(x::Real)
        global custom_grad_called = true
        xv = ReverseDiff.value(x)
        log(xv), Δ -> (1 / xv * Δ,)
    end
    custom_grad_called = false
    g1 = ReverseDiff.gradient(x -> f5(x[1]) * f5(x[2]) + exp(x[3]), x)
    g2 = ReverseDiff.gradient(x -> log(x[1]) * log(x[2]) + exp(x[3]), x)
    @test g1 == g2
    @test custom_grad_called

    f6(x) = sum(x)
    f6(x::TrackedArray{<:AbstractFloat}) = ReverseDiff.track(f6, x)
    @grad function f6(x::TrackedArray{T}) where {T <: AbstractFloat}
        global custom_grad_called = true
        xv = ReverseDiff.value(x)
        sum(xv), Δ -> (one.(xv) .* Δ,)
    end

    custom_grad_called = false
    g1 = ReverseDiff.gradient(f6, x)
    g2 = ReverseDiff.gradient(sum, x)
    @test g1 == g2
    @test custom_grad_called

    x2 = round.(Int, x)
    custom_grad_called = false
    g1 = ReverseDiff.gradient(f6, x2)
    g2 = ReverseDiff.gradient(sum, x2)
    @test g1 == g2
    @test !custom_grad_called
    f6(x::TrackedArray) = ReverseDiff.track(f6, x)
    @test_throws MethodError ReverseDiff.gradient(f6, x2)

    f7(x...) = +(x...)
    f7(x::TrackedReal{<:AbstractFloat}...) = ReverseDiff.track(f7, x...)
    @grad function f7(x::TrackedReal{T}...) where {T <: AbstractFloat}
        global custom_grad_called = true
        xv = ReverseDiff.value.(x)
        +(xv...), Δ -> one.(xv) .* Δ
    end
    custom_grad_called = false
    g1 = ReverseDiff.gradient(x -> f7(x...), x)
    g2 = ReverseDiff.gradient(sum, x)
    @test g1 == g2
    @test custom_grad_called

    f8(A; kwargs...) = sum(A, kwargs...)
    f8(A::TrackedMatrix; kwargs...) = ReverseDiff.track(f8, A; kwargs...)
    @grad function f8(A::AbstractMatrix; kwargs...)
        global custom_grad_called = true
        Av = ReverseDiff.value(A)
        sum(Av; kwargs...), Δ -> (zero(Av) .+ Δ,)
    end
    custom_grad_called = false
    g1 = ReverseDiff.gradient(A -> sum(f8(A, dims = 1)), A)
    g2 = ReverseDiff.gradient(A -> sum(sum(A, dims = 1)), A)
    @test g1 == g2
    @test custom_grad_called

    f9(x::AbstractVector) = sum(abs2, x)
    f9(x::AbstractVector{<:TrackedReal}) = ReverseDiff.track(f9, x)
    @grad function f9(x::AbstractVector{<:TrackedReal})
        global custom_grad_called = true
        xv = ReverseDiff.value(x)
        f9(xv), Δ -> ((2 * Δ) .* xv,)
    end
    custom_grad_called = false
    g1 = ReverseDiff.gradient(A) do x
        sum(i -> f9(view(x, :, i)), axes(x, 2))
    end
    g2 = ReverseDiff.gradient(x -> sum(abs2, x), A)
    @test g1 == g2
    @test custom_grad_called
end

# unnamed argument
f10(x, ::Val{N}) where {N} = N * x
f10(x::TrackedReal, v::Val) = ReverseDiff.track(f10, x, v)
@grad function f10(x::Real, ::Val{N}) where {N}
    return N * ReverseDiff.value(x), Δ -> (N * Δ, nothing)
end

# argument named like the function
f11(x) = 2 * x
f11(x::TrackedReal) = ReverseDiff.track(f11, x)
@grad function f11(f11::Real)
    return 2 * ReverseDiff.value(f11), Δ -> (2 * Δ,)
end

# positional default
f12(x, y = 2) = x * y
f12(x::TrackedReal) = ReverseDiff.track(f12, x)
@grad function f12(x::Real, y::Real = 2)
    return ReverseDiff.value(x) * y, Δ -> (y * Δ, nothing)
end

# return type annotation
f13(x) = x^2
f13(x::TrackedReal) = ReverseDiff.track(f13, x)
@grad function f13(x::Real)::Tuple
    xv = ReverseDiff.value(x)
    return xv^2, Δ -> (2 * xv * Δ,)
end

# `where` parameter used only by a keyword argument
f14(x; k = 1) = k * x
f14(x::TrackedReal; k = 1) = ReverseDiff.track(f14, x; k = k)

# arguments named like the internals of `@grad`
f15(tp, args, back, output, closure, func; kwargs = 1) = kwargs * (tp + args + back + output + closure + func)
function f15(tp::TrackedReal, args::TrackedReal, back::TrackedReal, output::TrackedReal, closure::TrackedReal, func::TrackedReal; kwargs = 1)
    return ReverseDiff.track(f15, tp, args, back, output, closure, func; kwargs = kwargs)
end
@grad function f15(tp::Real, args::Real, back::Real, output::Real, closure::Real, func::Real; kwargs = 1)
    xv = ReverseDiff.value.((tp, args, back, output, closure, func))
    return kwargs * sum(xv), Δ -> map(_ -> kwargs * Δ, xv)
end

# keyword argument used in the forward pass of a replayed tape
# (the pullback is reused, so it must read the input array that the replay updates)
f16(x; p = 2) = sum(x .^ p)
f16(x::TrackedArray; p = 2) = ReverseDiff.track(f16, x; p = p)
@grad function f16(x::AbstractVector; p = 2)
    xv = ReverseDiff.value(x)
    return f16(xv; p = p), Δ -> (p .* xv .^ (p - 1) .* Δ,)
end

# qualified function name (#318)
module GradQualified
    f(x) = sum(abs2, x)
end
GradQualified.f(x::TrackedArray) = ReverseDiff.track(GradQualified.f, x)
@grad function GradQualified.f(x)
    xv = ReverseDiff.value(x)
    return GradQualified.f(xv), Δ -> (2 .* Δ .* xv,)
end

@testset "@grad macro: signatures" begin
    @test ReverseDiff.gradient(x -> f10(x[1], Val(3)), [2.0]) == [3.0]
    @test ReverseDiff.gradient(x -> f11(x[1]), [2.0]) == [2.0]
    @test ReverseDiff.gradient(x -> f12(x[1]), [3.0]) == [2.0]
    @test ReverseDiff.gradient(x -> f13(x[1]), [3.0]) == [6.0]

    @test_nowarn @eval @grad function f14(x::Real; k::T = 1) where {T}
        return k * ReverseDiff.value(x), Δ -> (k * Δ,)
    end
    @test ReverseDiff.gradient(x -> f14(x[1]; k = 2), [3.0]) == [2.0]

    @test ReverseDiff.gradient(x -> f15(x...; kwargs = 3), rand(6)) == fill(3.0, 6)

    tape = ReverseDiff.GradientTape(x -> f16(x; p = 3), [1.0, 1.0])
    for t in (tape, ReverseDiff.compile(tape))
        result = DiffResults.GradientResult([0.0, 0.0])
        ReverseDiff.gradient!(result, t, [2.0, 3.0])
        @test DiffResults.value(result) == 35.0
        @test DiffResults.gradient(result) == [12.0, 27.0]
    end

    x = [1.0, 2.0, 3.0]
    @test ReverseDiff.gradient(GradQualified.f, x) == 2 .* x
    tape = ReverseDiff.GradientTape(GradQualified.f, x)
    for t in (tape, ReverseDiff.compile(tape))
        @test ReverseDiff.gradient!(t, [-2.0, 0.5, 4.0]) == [-4.0, 1.0, 8.0]
    end

    @test_throws "at least one positional argument" @eval @grad function f17(; k)
        return k, Δ -> ()
    end
end
