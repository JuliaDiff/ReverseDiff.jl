module MacrosTests

using ReverseDiff, ForwardDiff, Test, StaticArrays
using ForwardDiff: Dual, Partials, partials

include(joinpath(dirname(@__FILE__), "utils.jl"))

tp = InstructionTape()
x, a, b = rand(3)

############
# @forward #
############

f0(x) = 1.0 / (1.0 + exp(-x))
f0(a, b) = sqrt(a^2 + b^2)

ReverseDiff.@forward f1(x::T) where {T<:Real} = 1.0 / (1.0 + exp(-x))
ReverseDiff.@forward f1(a::A, b::B) where {A,B<:Real} = sqrt(a^2 + b^2)

ReverseDiff.@forward f2(x) = 1.0 / (1.0 + exp(-x))
ReverseDiff.@forward f2(a, b) = sqrt(a^2 + b^2)

ReverseDiff.@forward function f3(x::T) where {T<:Real}
    return 1.0 / (1.0 + exp(-x))
end

ReverseDiff.@forward function f3(a::A, b::B) where {A,B<:Real}
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
    return empty!(tp)
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
    return empty!(tp)
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

ReverseDiff.@skip g1(x::T) where {T<:Real} = 1.0 / (1.0 + exp(-x))
ReverseDiff.@skip g1(a::A, b::B) where {A,B<:Real} = sqrt(a^2 + b^2)

ReverseDiff.@skip g2(x) = 1.0 / (1.0 + exp(-x))
ReverseDiff.@skip g2(a, b) = sqrt(a^2 + b^2)

ReverseDiff.@skip function g3(x::T) where {T<:Real}
    return 1.0 / (1.0 + exp(-x))
end

ReverseDiff.@skip function g3(a::A, b::B) where {A,B<:Real}
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
        return dot(xv, xv), Δ -> (Δ * 2 * xv,)
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
        return Av * xv, Δ -> (Δ * xv', Av' * Δ)
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

    f3(A; dims) = sum(A; dims=dims)
    f3(A::TrackedMatrix; dims) = ReverseDiff.track(f3, A; dims=dims)
    @grad function f3(A::AbstractMatrix; dims)
        global custom_grad_called = true
        Av = ReverseDiff.value(A)
        return sum(Av; dims=dims), Δ -> (zero(Av) .+ Δ,)
    end
    custom_grad_called = false
    g1 = ReverseDiff.gradient(A -> sum(f3(A; dims=1)), A)
    g2 = ReverseDiff.gradient(A -> sum(sum(A; dims=1)), A)
    @test g1 == g2
    @test custom_grad_called

    f4(::typeof(log), A; dims) = sum(log, A; dims=dims)
    f4(::typeof(log), A::TrackedMatrix; dims) = ReverseDiff.track(f4, log, A; dims=dims)
    @grad function f4(::typeof(log), A::AbstractMatrix; dims)
        global custom_grad_called = true
        Av = ReverseDiff.value(A)
        return sum(log, Av; dims=dims), Δ -> (nothing, 1 ./ Av .* Δ)
    end
    custom_grad_called = false
    g1 = ReverseDiff.gradient(A -> sum(f4(log, A; dims=1)), A)
    g2 = ReverseDiff.gradient(A -> sum(sum(log, A; dims=1)), A)
    @test g1 == g2
    @test custom_grad_called

    f5(x) = log(x)
    f5(x::TrackedReal) = ReverseDiff.track(f5, x)
    @grad function f5(x::Real)
        global custom_grad_called = true
        xv = ReverseDiff.value(x)
        return log(xv), Δ -> (1 / xv * Δ,)
    end
    custom_grad_called = false
    g1 = ReverseDiff.gradient(x -> f5(x[1]) * f5(x[2]) + exp(x[3]), x)
    g2 = ReverseDiff.gradient(x -> log(x[1]) * log(x[2]) + exp(x[3]), x)
    @test g1 == g2
    @test custom_grad_called

    f6(x) = sum(x)
    f6(x::TrackedArray{<:AbstractFloat}) = ReverseDiff.track(f6, x)
    @grad function f6(x::TrackedArray{T}) where {T<:AbstractFloat}
        global custom_grad_called = true
        xv = ReverseDiff.value(x)
        return sum(xv), Δ -> (one.(xv) .* Δ,)
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
    @grad function f7(x::TrackedReal{T}...) where {T<:AbstractFloat}
        global custom_grad_called = true
        xv = ReverseDiff.value.(x)
        return +(xv...), Δ -> one.(xv) .* Δ
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
        return sum(Av; kwargs...), Δ -> (zero(Av) .+ Δ,)
    end
    custom_grad_called = false
    g1 = ReverseDiff.gradient(A -> sum(f8(A; dims=1)), A)
    g2 = ReverseDiff.gradient(A -> sum(sum(A; dims=1)), A)
    @test g1 == g2
    @test custom_grad_called

    f9(x::AbstractVector) = sum(abs2, x)
    f9(x::AbstractVector{<:TrackedReal}) = ReverseDiff.track(f9, x)
    @grad function f9(x::AbstractVector{<:TrackedReal})
        global custom_grad_called = true
        xv = ReverseDiff.value(x)
        return f9(xv), Δ -> ((2 * Δ) .* xv,)
    end
    custom_grad_called = false
    g1 = ReverseDiff.gradient(A) do x
        sum(i -> f9(view(x, :, i)), axes(x, 2))
    end
    g2 = ReverseDiff.gradient(x -> sum(abs2, x), A)
    @test g1 == g2
    @test custom_grad_called
end

end # module
