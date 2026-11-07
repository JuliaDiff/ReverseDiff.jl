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
    output, back = ChainRulesCore.rrule(f, input)
    _, d = back(1)
    @test output == f(input)
    @test d == fill(3, size(input))
    # function g
    inputs = rand(3, 3), rand(3, 3)
    output, back = ChainRulesCore.rrule(g, inputs...)
    _, d1, d2 = back(1)
    @test output == g(inputs...)
    @test d1 == fill(3, size(inputs[1]))
    @test d2 == fill(5, size(inputs[2]))


    ## ReverseDiff
    #function f
    inputs = (rand(3, 3),)

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

    results = (similar(inputs[1]), similar(inputs[2]))
    compiled_tape = ReverseDiff.CompiledTape(f_tape)
    ReverseDiff.gradient!(results, compiled_tape, inputs)
    @test results[1] == fill(3, size(inputs[1]))
    @test results[2] == fill(5, size(inputs[2]))

end

@testset "custom struct input" begin
    input = rand(3, 3)
    output, back = ChainRulesCore.rrule(f, MyStruct(), input)
    _, _, d = back(1)
    @test output == f(MyStruct(), input)
    @test d == fill(3, size(input))

    output, back = ChainRulesCore.rrule(f, input, MyStruct())
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
    ReverseDiff.finish!(tp)
    @test length(tp) == 3 # sum, broadcast+, broadcast*, but not `g`
end

### Functions with varargs and kwargs
# Varargs
f_vararg(x, args...) = sum(4x .+ sum(args))

function ChainRulesCore.rrule(::typeof(f_vararg), x, args...)
    r = f_vararg(x, args...)
    function back(d)
        return NoTangent(), fill(3 * d, size(x)), ntuple(_ -> NoTangent(), length(args))...
    end
    return r, back
end

ReverseDiff.@grad_from_chainrules f_vararg(x::ReverseDiff.TrackedArray, args...)

@testset "Function with Varargs" begin
    inputs = (rand(3, 3),)

    results = (similar(inputs[1]),)
    f_tape = ReverseDiff.GradientTape(x -> f_vararg(x, 1, 2, 3) + 2, (rand(3, 3),))
    ReverseDiff.gradient!(results, f_tape, inputs)

    @test results[1] == fill(3, size(inputs[1]))
end


# Vargs and kwargs
f_kw(x, args...; k = 1, kwargs...) = sum(4x .+ sum(args) .+ (k + kwargs[:j]))

function ChainRulesCore.rrule(::typeof(f_kw), x, args...; k = 1, kwargs...)
    r = f_kw(x, args...; k = k, kwargs...)
    function back(d)
        return NoTangent(), fill(3 * d, size(x)), ntuple(_ -> NoTangent(), length(args))...
    end
    return r, back
end

ReverseDiff.@grad_from_chainrules f_kw(x::ReverseDiff.TrackedArray, args...; k = 1, kwargs...)

@testset "Function with Varargs and kwargs" begin
    inputs = (rand(3, 3),)

    results = (similar(inputs[1]),)
    f_tape = ReverseDiff.GradientTape(x -> f_kw(x, 1, 2, 3; k = 2, j = 3) + 2, (rand(3, 3),))
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
    inputs = (rand(3, 3),)
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

### Typed varargs
# `MyStruct` is defined in this module, so the vararg type has to survive macro hygiene,
# and the collected varargs must not be re-asserted against their element type.
f_typed_vararg(x, ys::MyStruct...) = sum(4x .+ length(ys))

function ChainRulesCore.rrule(::typeof(f_typed_vararg), x, ys::MyStruct...)
    function pb_f_typed_vararg(Δ)
        return (NoTangent(), fill(3 * Δ, size(x)), ntuple(_ -> NoTangent(), length(ys))...)
    end
    return f_typed_vararg(x, ys...), pb_f_typed_vararg
end

ReverseDiff.@grad_from_chainrules f_typed_vararg(x::ReverseDiff.TrackedArray, ys::MyStruct...)

@testset "typed varargs" begin
    x = rand(3, 3)
    @test ReverseDiff.gradient(f_typed_vararg, x) == fill(3, size(x))
    @test ReverseDiff.gradient(u -> f_typed_vararg(u, MyStruct(), MyStruct()), x) ==
        fill(3, size(x))
end

### Pullbacks that return the wrong number of tangents
# Without a length check the lone tangent is broadcast over every argument.
f_short(x, ys...) = sum(4x) + sum(sum, ys)

function ChainRulesCore.rrule(::typeof(f_short), x, ys...)
    pb_f_short(Δ) = (NoTangent(), fill(3 * Δ, size(x))) # deliberately short
    return f_short(x, ys...), pb_f_short
end

ReverseDiff.@grad_from_chainrules f_short(
    x::ReverseDiff.TrackedArray, ys::ReverseDiff.TrackedArray...
)

h_short(x, y) = sum(4x) + sum(4y)
function h_short(x::ReverseDiff.TrackedArray, y::ReverseDiff.TrackedArray)
    return ReverseDiff.track(h_short, x, y)
end
ReverseDiff.@grad function h_short(x, y)
    xv, yv = ReverseDiff.value(x), ReverseDiff.value(y)
    pb_h_short(Δ) = (fill(3 * Δ, size(xv)),) # deliberately short
    return sum(4xv) + sum(4yv), pb_h_short
end

g_notuple(x) = sum(4x)
g_notuple(x::ReverseDiff.TrackedArray) = ReverseDiff.track(g_notuple, x)
ReverseDiff.@grad function g_notuple(x)
    xv = ReverseDiff.value(x)
    pb_g_notuple(Δ) = fill(3 * Δ, size(xv)) # deliberately not a tuple
    return sum(4xv), pb_g_notuple
end

@testset "pullback returning the wrong tangents" begin
    x, y = rand(3, 3), rand(3, 3)
    @test_throws "must return a tuple of length 3, not 2" ReverseDiff.gradient(f_short, (x, y))
    @test_throws "must return a tuple of length 2, not 1" ReverseDiff.gradient(h_short, (x, y))
    @test_throws "must return a tuple of tangents" ReverseDiff.gradient(g_notuple, x)
end

### Keyword argument defaults
# The default of the imported signature has to survive, and the caller's `k` has to be
# forwarded by value. The rrule uses a different default, so falling back to it shows.
f_kwdefault(x; k = 1) = sum(4x) * k

function ChainRulesCore.rrule(::typeof(f_kwdefault), x; k = 10)
    pb_f_kwdefault(Δ) = (NoTangent(), fill(3 * k * Δ, size(x)))
    return f_kwdefault(x; k = k), pb_f_kwdefault
end

ReverseDiff.@grad_from_chainrules f_kwdefault(x::ReverseDiff.TrackedArray; k = 1)

@testset "keyword argument defaults" begin
    x = rand(3, 3)
    @test ReverseDiff.gradient(f_kwdefault, x) == fill(3, size(x))
    @test ReverseDiff.gradient(u -> f_kwdefault(u; k = 2), x) == fill(6, size(x))
end

### Positional argument defaults
# A default in the forwarding call would turn the argument into a keyword argument.
f_posdefault(x, y = 2) = sum(4x) * y

function ChainRulesCore.rrule(::typeof(f_posdefault), x, y = 2)
    pb_f_posdefault(Δ) = (NoTangent(), fill(3 * y * Δ, size(x)), NoTangent())
    return f_posdefault(x, y), pb_f_posdefault
end

ReverseDiff.@grad_from_chainrules f_posdefault(x::ReverseDiff.TrackedArray, y::Int = 2)

@testset "positional argument defaults" begin
    x = rand(3, 3)
    @test ReverseDiff.gradient(f_posdefault, x) == fill(6, size(x))
    @test ReverseDiff.gradient(u -> f_posdefault(u, 3), x) == fill(9, size(x))
end

### `AbstractZero` tangents for tracked arguments
g_zero(x, y, z) = sum(4x) + sum(4y) + sum(4z)

function ChainRulesCore.rrule(::typeof(g_zero), x, y, z)
    pb_g_zero(Δ) = (NoTangent(), fill(3 * Δ, size(x)), ZeroTangent(), NoTangent())
    return g_zero(x, y, z), pb_g_zero
end

ReverseDiff.@grad_from_chainrules g_zero(
    x::ReverseDiff.TrackedArray, y::ReverseDiff.TrackedArray, z::ReverseDiff.TrackedArray
)

h_zero(x, y) = x

function ChainRulesCore.rrule(::typeof(h_zero), x, y)
    pb_h_zero(Δ) = (NoTangent(), 3 * Δ, NoTangent())
    return h_zero(x, y), pb_h_zero
end

ReverseDiff.@grad_from_chainrules h_zero(x::ReverseDiff.TrackedReal, y::ReverseDiff.TrackedReal)

k_zero(x, y) = sum(4x) + sum(4y)
function k_zero(x::ReverseDiff.TrackedArray, y::ReverseDiff.TrackedArray)
    return ReverseDiff.track(k_zero, x, y)
end
ReverseDiff.@grad function k_zero(x, y)
    xv, yv = ReverseDiff.value(x), ReverseDiff.value(y)
    pb_k_zero(Δ) = (fill(3 * Δ, size(xv)), nothing)
    return sum(4xv) + sum(4yv), pb_k_zero
end

h_zero_vec(z) = h_zero(z[1], z[2])

@testset "zero tangents for tracked arguments" begin
    x, y, z = rand(3, 3), rand(3, 3), rand(3, 3)
    results = ReverseDiff.gradient(g_zero, (x, y, z))
    @test results[1] == fill(3, size(x))
    @test results[2] == zeros(size(y))
    @test results[3] == zeros(size(z))

    # tracked scalars
    @test ReverseDiff.gradient(h_zero_vec, [1.0, 2.0]) == [3.0, 0.0]

    # `nothing` from a `@grad` pullback
    results = ReverseDiff.gradient(k_zero, (x, y))
    @test results[1] == fill(3, size(x))
    @test results[2] == zeros(size(y))
end

### `where` clauses in the imported signature
f_where(x) = sum(4x)

function ChainRulesCore.rrule(::typeof(f_where), x)
    pb_f_where(Δ) = (NoTangent(), fill(3 * Δ, size(x)))
    return f_where(x), pb_f_where
end

# the static parameter is used in the tracked argument
ReverseDiff.@grad_from_chainrules f_where(x::ReverseDiff.TrackedArray{V}) where {V}

g_where(x, y) = sum(4x) + sum(4y)

function ChainRulesCore.rrule(::typeof(g_where), x, y)
    pb_g_where(Δ) = (NoTangent(), fill(3 * Δ, size(x)), NoTangent())
    return g_where(x, y), pb_g_where
end

# the static parameter is bounded and used only in the *untracked* argument
ReverseDiff.@grad_from_chainrules g_where(
    x::ReverseDiff.TrackedArray, y::AbstractVector{V}
) where {V <: Real}

h_where(x, y = 2) = sum(4x) * y

function ChainRulesCore.rrule(::typeof(h_where), x, y = 2)
    pb_h_where(Δ) = (NoTangent(), fill(3 * y * Δ, size(x)), NoTangent())
    return h_where(x, y), pb_h_where
end

@testset "where clauses" begin
    x = rand(3, 3)
    @test ReverseDiff.gradient(f_where, x) == fill(3, size(x))
    @test ReverseDiff.gradient(u -> g_where(u, [1.0, 2.0]), x) == fill(3, size(x))

    # the static parameter is used only in an argument with a default
    @test_nowarn @eval ReverseDiff.@grad_from_chainrules h_where(
        x::ReverseDiff.TrackedArray, y::V = 2
    ) where {V <: Real}
    @test ReverseDiff.gradient(h_where, x) == fill(6, size(x))
    @test ReverseDiff.gradient(u -> h_where(u, 3), x) == fill(9, size(x))
end

### Qualified function names
module QualifiedNames
    f_qualified(x) = sum(4x)
end

function ChainRulesCore.rrule(::typeof(QualifiedNames.f_qualified), x)
    pb_f_qualified(Δ) = (NoTangent(), fill(3 * Δ, size(x)))
    return QualifiedNames.f_qualified(x), pb_f_qualified
end

ReverseDiff.@grad_from_chainrules QualifiedNames.f_qualified(x::ReverseDiff.TrackedArray)

@testset "qualified function names" begin
    x = rand(3, 3)
    @test ReverseDiff.gradient(QualifiedNames.f_qualified, x) == fill(3, size(x))
end

### Argument names that coincide with the function or with internal names
f_names(f_names, kwargs) = sum(4f_names) + sum(4kwargs)

function ChainRulesCore.rrule(::typeof(f_names), x, y)
    pb_f_names(Δ) = (NoTangent(), fill(3 * Δ, size(x)), fill(5 * Δ, size(y)))
    return f_names(x, y), pb_f_names
end

ReverseDiff.@grad_from_chainrules f_names(
    f_names::ReverseDiff.TrackedArray, kwargs::ReverseDiff.TrackedArray
)

@testset "argument names" begin
    x, y = rand(3, 3), rand(3, 3)
    results = ReverseDiff.gradient(f_names, (x, y))
    @test results[1] == fill(3, size(x))
    @test results[2] == fill(5, size(y))
end

@testset "invalid signatures" begin
    @test_throws LoadError @eval ReverseDiff.@grad_from_chainrules f_where
    @test_throws LoadError @eval ReverseDiff.@grad_from_chainrules f_where()
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
            inputs = (rand(3, 3),)

            results = (similar(inputs[1]),)
            f_tape = GradientTape(x -> f(x) + 2, (rand(3, 3),))
            gradient!(results, f_tape, inputs)

            @test results[1] == fill(3, size(inputs[1]))
        end

    end # end of SubModule
end # end of IsolatedModuleForTestingScoping
