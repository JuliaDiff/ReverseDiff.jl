module BroadcastTests

using ReverseDiff

using ForwardDiff
using LinearAlgebra
using SpecialFunctions
using StaticArrays
using Test

include("../utils.jl")

using Base.Broadcast: AbstractArrayStyle, BroadcastStyle, Broadcasted, DefaultArrayStyle,
    Style, Unknown, result_style
using ReverseDiff: TrackedArray, TrackedReal, TrackedStyle

# stands in for a foreign array style such as `CuArrayStyle` or `StructuredMatrixStyle`
struct ForeignStyle <: AbstractArrayStyle{Any} end
ForeignStyle(::Val) = ForeignStyle()

# a foreign 0-dimensional array, whose style carries `Any` rather than a dimension
struct Foreign0d{T} <: AbstractArray{T, 0}
    x::T
end
Base.size(::Foreign0d) = ()
Base.getindex(f::Foreign0d) = f.x
Base.getindex(f::Foreign0d, ::CartesianIndex{0}) = f.x
Base.Broadcast.BroadcastStyle(::Type{<:Foreign0d}) = ForeignStyle()
Base.similar(::Broadcasted{ForeignStyle}, ::Type{T}, axs) where {T} = similar(Array{T}, axs)

# at top level, since a local function that captures itself takes the scalar rules
inner(t) = ForwardDiff.Dual{typeof(ForwardDiff.Tag(inner, typeof(t)))}(t, one(t))

# a struct with fields, which `NotTracked` declares constant
struct Scale
    s::Float64
end

############################################################################################

@testset "`BroadcastStyle` tracks dimensionality" begin
    tp = InstructionTape()

    @test BroadcastStyle(typeof(track(rand(3), tp))) === TrackedStyle{1}()
    @test BroadcastStyle(typeof(track(rand(3, 3), tp))) === TrackedStyle{2}()
    # `Base` styles scalars as `AbstractArrayStyle{0}`
    @test BroadcastStyle(typeof(track(rand(), tp))) === TrackedStyle{0}()
    @test TrackedStyle{1}(Val(2)) === TrackedStyle{2}()
end

@testset "`BroadcastStyle` precedence" begin
    for (a, b, expected) in (
            (TrackedStyle{0}(), DefaultArrayStyle{0}(), TrackedStyle{0}()),
            (TrackedStyle{0}(), DefaultArrayStyle{2}(), TrackedStyle{2}()),
            (TrackedStyle{1}(), DefaultArrayStyle{1}(), TrackedStyle{1}()),
            (TrackedStyle{1}(), TrackedStyle{2}(), TrackedStyle{2}()),
            (TrackedStyle{1}(), Unknown(), TrackedStyle{1}()),
            (TrackedStyle{1}(), ForeignStyle(), TrackedStyle{Any}()),
            (TrackedStyle{Any}(), ForeignStyle(), TrackedStyle{Any}()),
            (TrackedStyle{Any}(), DefaultArrayStyle{1}(), TrackedStyle{Any}()),
            # a scalar loses to a tuple, as in `Base`
            (TrackedStyle{0}(), Style{Tuple}(), Style{Tuple}()),
            (TrackedStyle{1}(), Style{Tuple}(), TrackedStyle{1}()),
        )
        @test result_style(a, b) === expected
        @test result_style(b, a) === expected
    end
end

@testset "foreign array styles keep their values tracked" begin
    a, v = rand(3, 3), rand(3)
    d = Diagonal(rand(3))
    u = UpperTriangular(rand(3, 3))
    s = SVector(1.0, 2.0, 3.0)

    @test result_style(TrackedStyle{2}(), BroadcastStyle(typeof(d))) === TrackedStyle{2}()
    @test result_style(TrackedStyle{2}(), BroadcastStyle(typeof(u))) === TrackedStyle{2}()
    @test result_style(TrackedStyle{1}(), BroadcastStyle(typeof(s))) === TrackedStyle{1}()
    @test result_style(TrackedStyle{Any}(), BroadcastStyle(typeof(s))) === TrackedStyle{Any}()

    tp = InstructionTape()
    @test track(a, tp) .* d isa TrackedArray
    @test track(v, tp) .* s isa TrackedArray
    # a static container survives into the tracked value
    @test value(track(v, tp) .* s) isa StaticArray

    @test ReverseDiff.gradient(x -> sum(x .* d), a) ≈ Matrix(d)
    @test ReverseDiff.gradient(x -> sum(x .* u), a) ≈ Matrix(u)
    @test ReverseDiff.gradient(x -> sum(x .* s), v) ≈ s

    # a closure takes the scalar rules, and still broadcasts a static value statically
    c = 3.0
    tp = InstructionTape()
    xs = track(SVector(1.0, 2.0), tp)
    y, z = (t -> t * c).(xs), ((t, u) -> t * u * c).(xs, SVector(1.0, 2.0))
    @test y isa StaticArray
    @test z isa StaticArray
    ReverseDiff.seed!(sum(y) + sum(z))
    ReverseDiff.reverse_pass!(ReverseDiff.finish!(tp))
    @test deriv(xs) == [6.0, 9.0]
    @test ReverseDiff.gradient(x -> sum((t -> t * c).(x)), SVector(1.0, 2.0)) == [3.0, 3.0]
end

@testset "no `BroadcastStyle` ambiguities" begin
    # `ReverseDiff` has unrelated pre-existing ambiguities, so filter to broadcasting
    @test isempty(
        filter(Test.detect_ambiguities(ReverseDiff)) do (m1, m2)
            m1.name === :BroadcastStyle || m2.name === :BroadcastStyle
        end
    )
end

@testset "scalar broadcasting matches `Base`" begin
    tr = track(2.0, InstructionTape())

    @test exp.(tr) isa TrackedReal
    @test tr .+ tr isa TrackedReal
    @test tr .+ [1.0, 2.0, 3.0] isa TrackedArray
    @test tr .+ (1, 2, 3) isa Tuple
end

@testset "`f.(x)` and `broadcast(f, x)` agree" begin
    a, b = rand(3), rand(3)

    for (dotted, called) in (
            (tp -> exp.(track(a, tp)), tp -> broadcast(exp, track(a, tp))),
            (tp -> track(a, tp) .+ track(b, tp), tp -> broadcast(+, track(a, tp), track(b, tp))),
            (tp -> atan.(track(a, tp), b), tp -> broadcast(atan, track(a, tp), b)),
        )
        tp1, tp2 = InstructionTape(), InstructionTape()
        x, y = dotted(tp1), called(tp2)
        instrs1, instrs2 = take_recorded!(tp1), take_recorded!(tp2)

        @test value(x) == value(y)
        @test length(instrs1) == length(instrs2) == 1
        @test instrs1[1].func === instrs2[1].func
    end
end

@testset "`@forward` and `@skip` wrappers do not force the fallback path" begin
    tp = InstructionTape()
    x = track(rand(3, 3), tp)
    y = broadcast(ReverseDiff.ForwardOptimize(exp), x)
    @test y isa TrackedArray
    @test length(take_recorded!(tp)) == 1

    # `@skip` results are untracked, as with `map` and scalars
    tp = InstructionTape()
    x = track(rand(3, 3), tp)
    y = broadcast(ReverseDiff.SkipOptimize(exp), x)
    @test y isa Matrix{Float64}
    @test y == exp.(value(x))
    @test isempty(take_recorded!(tp))

    # fused, and at every order
    f(v) = sum(v .* ReverseDiff.@skip(exp).(v))
    a = rand(3)
    @test ReverseDiff.gradient(f, a) ≈ exp.(a)
    @test ReverseDiff.hessian(f, a) ≈ zeros(3, 3)

    # arrays too
    s = ReverseDiff.SkipOptimize(v -> sum(exp, v))
    @test ReverseDiff.gradient(v -> sum(v) * s(v), a) ≈ fill(sum(exp, a), 3)
    @test ReverseDiff.hessian(v -> sum(v) * s(v), a) == zeros(3, 3)
end

@testset "a type broadcast as a function does not force the fallback path" begin
    tp = InstructionTape()
    x = track(rand(3), tp)

    y = Real.(x)
    instrs = take_recorded!(tp)

    @test y isa TrackedArray
    @test length(instrs) == 1
    @test instrs[1].func === ReverseDiff.∇broadcast
    @test ReverseDiff.gradient(v -> sum(Real.(v)), [1.0, 2.0]) == [1.0, 1.0]
end

@testset "scalar-array broadcasting preserves the derivative type" begin
    # `Base` routes scalar-times-array through broadcasting
    a = rand(1, 1)
    f = x -> sum(first(x) * (x * x) + x)   # == x[1]^3 + x[1]

    @test ReverseDiff.hessian(f, a) ≈ [6 * a[1];;]
end

@testset "broadcast expressions stay fused" begin
    tp = InstructionTape()
    x, y = track(rand(3), tp), track(rand(3), tp)

    sum((x .+ y) .* sin.(x))

    # one instruction for the fused expression, one for `sum`
    @test length(take_recorded!(tp)) == 2
end

@testset "`TrackedArray`s are rejected as broadcast destinations" begin
    a = rand(4)
    msg = "`TrackedArray`s do not support `setindex!` and cannot be used as a broadcast destination. Use `y = f.(x)` instead."

    tp = InstructionTape()
    x = track(copy(a), tp)
    @test_throws ArgumentError(msg) track(zeros(4), tp) .= exp.(x)
    @test_throws ArgumentError(msg) track(zeros(4), tp) .= a
    @test_throws ArgumentError(msg) track(fill(1.0), tp) .= 2.0
    # a foreign style brings its own `copyto!`
    @test_throws ArgumentError(msg) track(zeros(2), tp) .= SVector(1.0, 2.0)
    # nothing may be recorded before the failure
    @test isempty(take_recorded!(tp))

    # an untracked container of tracked elements is a valid destination
    dest = Vector{TrackedReal{Float64, Float64, Nothing}}(undef, 4)
    dest .= exp.(x)
    @test value.(dest) ≈ exp.(a)
    # the elements carry the tape, so it must not be dropped on the way in
    @test all(d -> tape(d) === tp, dest)
end

@testset "an untracked argument's `NaN` partial stays out of the tracked partials" begin
    # `DiffRules` gives `besselj` a `NaN` partial for its order, which is untracked here
    nu, z = [0.5, 1.5, 2.5], 0.7
    # J_ν'(z) = (J_{ν-1}(z) - J_{ν+1}(z)) / 2
    expected = sum((besselj(n - 1, z) - besselj(n + 1, z)) / 2 for n in nu)

    @test ReverseDiff.gradient(y -> sum(besselj.(nu, y)), [z]) ≈ [expected]
end

@testset "an array of `TrackedReal`s is differentiated like a `TrackedArray`" begin
    a, b = rand(3), rand(3)
    tp = InstructionTape()
    # a container of tracked elements, which `track` itself never produces
    x = map(xi -> track(xi, tp), a)

    y = x .* b
    s = sum(y)
    ReverseDiff.finish!(tp)

    @test y isa TrackedArray
    @test value(y) ≈ a .* b
    # one instruction for the broadcast, one for `sum`
    @test length(tp) == 2
    @test ReverseDiff.instructions(tp)[1].func === ReverseDiff.∇broadcast

    ReverseDiff.seed!(s)
    ReverseDiff.reverse_pass!(tp)
    # `deriv.(x)` would be traced like any other broadcast, as it is for a `TrackedArray`
    @test map(deriv, x) ≈ b
end

@testset "an array of `TrackedReal`s with an abstract element type" begin
    a = [1.0, 2.0]

    @test ReverseDiff.gradient(u -> sum(TrackedReal[u[1], u[2]] .* 2.0), a) == [2.0, 2.0]
    @test ReverseDiff.gradient(u -> sum(exp.(TrackedReal[u[1], u[2]])), a) ≈ exp.(a)
    @test ReverseDiff.gradient(u -> sum(u .* TrackedReal[u[2], u[1]]), a) == [4.0, 2.0]

    tape = ReverseDiff.GradientTape(u -> sum(exp.(TrackedReal[u[1], u[2]])), a)
    @test ReverseDiff.gradient!(tape, [0.5, 1.5]) ≈ exp.([0.5, 1.5])
end

@testset "functions constant in their argument" begin
    a = rand(3)

    @test ReverseDiff.gradient(x -> sum((t -> 1.0).(x)), a) == zeros(3)
    @test ReverseDiff.gradient(x -> sum(oneunit.(x)), a) == zeros(3)
end

@testset "a type-unstable function keeps its partials" begin
    # the integer literal makes `Base` widen the results to an abstract element type
    relu(t) = t > 0 ? t : 0
    a = [-1.0, 2.0]

    tp = InstructionTape()
    y = relu.(track(copy(a), tp))

    @test y isa TrackedArray
    @test value(y) == [0.0, 2.0]
    @test ReverseDiff.gradient(x -> sum(relu.(x)), a) == [0.0, 1.0]

    # recorded with every element on the constant branch, replayed on the other
    relu0(t) = t > 0 ? t : 0.0
    for f in (
            x -> sum(relu.(x)) + sum(x), x -> sum(relu0.(x)) + sum(x),
            x -> sum(ifelse.(x .> 0, x, 0.0)) + sum(x),
        )
        tape = ReverseDiff.GradientTape(f, [-1.0, -2.0])
        @test ReverseDiff.gradient!(tape, [1.5, 2.5]) == [2.0, 2.0]
        @test ReverseDiff.gradient!(ReverseDiff.compile(tape), [1.5, 2.5]) == [2.0, 2.0]
    end
    @test relu0.(track([-1.0, -2.0], InstructionTape())) isa TrackedArray

    # a static array whose widened element type is not isbits
    tape = ReverseDiff.GradientTape(x -> sum(relu.(x)), MVector(-1.0, 2.0))
    @test ReverseDiff.gradient!(tape, MVector(1.0, 2.0)) == [1.0, 1.0]
end

@testset "the cached partials have a concrete element type" begin
    # the inferred element type decides what is recorded
    tp = InstructionTape()
    track(rand(3), tp) .^ 2
    @test isconcretetype(eltype(first(only(take_recorded!(tp)).cache)))
end

@testset "a non-`Real` result keeps its partials" begin
    a = [1.0, 2.0]

    # `real.(cis.(x))` would fuse into one broadcast with `Real` results, so the `Complex`
    # and `Tuple` results are materialized in a broadcast of their own
    @test ReverseDiff.gradient(x -> (y = cis.(x); sum(real.(y))), a) ≈ -sin.(a)
    @test ReverseDiff.gradient(x -> (y = sincos.(x); sum(first.(y))), a) ≈ cos.(a)
    @test ReverseDiff.gradient(x -> (y = broadcast(t -> (t, 2t), x); sum(last.(y))), a) ==
        [2.0, 2.0]
    @test ReverseDiff.gradient(x -> (y = x .* cis.(x); sum(abs2.(y))), a) ≈ 2 .* a
    @test ReverseDiff.jacobian(x -> (y = cis.(x); imag.(y)), a) ≈ diagm(cos.(a))

    tape = ReverseDiff.GradientTape(x -> (y = cis.(x); sum(real.(y))), a)
    @test ReverseDiff.gradient!(tape, [0.5, 1.5]) ≈ -sin.([0.5, 1.5])
end

@testset "a foreign tag is never read as our own" begin
    # a `Dual` built inside the function buries the broadcast's own partial
    @test_throws ForwardDiff.DualMismatchError ReverseDiff.gradient(
        x -> sum(inner.(x)), [1.0, 2.0]
    )
    # also on one branch only, which is recorded
    @test_throws ForwardDiff.DualMismatchError ReverseDiff.gradient(
        x -> sum((t -> t > 1.5 ? inner(t) : t).(x)), [1.0, 2.0]
    )
    # and next to a constant branch, which widens the results to `Real`
    @test_throws ForwardDiff.DualMismatchError ReverseDiff.gradient(
        x -> sum((t -> t > 1.5 ? inner(t) : 1.0).(x)), [1.0, 2.0]
    )

    # an enclosing differentiation's `Dual` is constant in our arguments
    xs = [1.0, 3.0]
    m(a) = sum(
        ReverseDiff.gradient(
            x -> sum(ifelse.(x .> 2, x, a)), xs, ReverseDiff.GradientConfig(xs, typeof(a))
        )
    )
    @test ForwardDiff.derivative(m, 1.0) == 0.0
end

@testset "a perturbation the derivative cannot hold is rejected (#67, #168)" begin
    # `a` reaches the broadcast as a `Dual`, but the tape's derivatives are `Float64`
    g(a) = sum(ReverseDiff.gradient(x -> sum(x .* a), [1.0, 2.0]))
    E = ForwardDiff.Dual{ForwardDiff.Tag{typeof(g), Float64}, Float64, 1}
    msg = "a broadcast argument with element type $E carries a perturbation that a derivative of type Float64 cannot hold"

    @test_throws ArgumentError(msg) ForwardDiff.derivative(g, 3.0)

    # an abstract element type is checked element by element
    g2(a) = sum(ReverseDiff.gradient(x -> sum(x .* Union{Float64, typeof(a)}[1.0, a]), [1.0, 2.0]))
    E2 = ForwardDiff.Dual{ForwardDiff.Tag{typeof(g2), Float64}, Float64, 1}
    msg2 = "a broadcast argument with element type $E2 carries a perturbation that a derivative of type Float64 cannot hold"
    @test_throws ArgumentError(msg2) ForwardDiff.derivative(g2, 3.0)
    g3(a) = sum(ReverseDiff.gradient(x -> sum(ifelse.(x .> 5, x, Real[1.0, a])), [1.0, 2.0]))
    E3 = ForwardDiff.Dual{ForwardDiff.Tag{typeof(g3), Float64}, Float64, 1}
    msg3 = "a broadcast argument with element type $E3 carries a perturbation that a derivative of type Float64 cannot hold"
    @test_throws ArgumentError(msg3) ForwardDiff.derivative(g3, 3.0)

    # a tape whose derivatives are themselves `Dual`s can hold it, so it is left alone
    h(a) = ReverseDiff.gradient(x -> sum(x .* a), [ForwardDiff.Dual(1.0, 0.0)])
    @test h(3.0) == [ForwardDiff.Dual(3.0, 0.0)]

    # also when the tag without the perturbation was created first
    ReverseDiff.gradient(x -> sum(atan.(x, 2.0)), [1.0, 2.0])
    k(a) = sum(
        ReverseDiff.gradient(
            x -> sum(atan.(x, a)), [1.0, 2.0],
            ReverseDiff.GradientConfig([1.0, 2.0], typeof(a))
        )
    )
    xs = [1.0, 2.0]
    @test ForwardDiff.derivative(k, 3.0) ≈ sum(@. (xs^2 - 9) / (xs^2 + 9)^2)

    # the scalar rules bury the tracked value in the `Dual`, which must not give a zero derivative
    msg4 = "ForwardDiff cannot differentiate through ReverseDiff (see https://github.com/JuliaDiff/ReverseDiff.jl/issues/45)"
    for g4 in (
            a -> sum(ReverseDiff.gradient(x -> x[1] * a, [1.0, 2.0])),
            a -> sum(ReverseDiff.gradient(x -> sum(x .* Real[1.0, a]), [1.0, 2.0])),
            a -> sum(ReverseDiff.jacobian(x -> [x[1] * a, x[2]], [1.0, 2.0])),
        )
        @test_throws ArgumentError(msg4) ForwardDiff.derivative(g4, 3.0)
    end
    # storing the `Dual` as a tracked number would cut it off the tape
    msg5 = "this nesting of ForwardDiff and ReverseDiff is not supported: a `Dual` of tracked numbers cannot be converted to a tracked number (see https://github.com/JuliaDiff/ReverseDiff.jl/issues/45)"
    function g5(a)
        return sum(
            ReverseDiff.gradient([1.0, 2.0]) do x
                v = [x[1], x[2]]
                v[1] = x[1] * a
                return sum(v)
            end
        )
    end
    @test_throws ArgumentError(msg5) ForwardDiff.derivative(g5, 3.0)

    # the other way around is fine, also next to tracked numbers
    @test ReverseDiff.gradient(x -> ForwardDiff.derivative(a -> x[1] * a^2, 2.0), [1.0, 2.0]) ==
        [4.0, 0.0]
    @test ReverseDiff.gradient(
        x -> ForwardDiff.derivative(a -> sum([x[1] * a^2, x[2]]), 2.0),
        [1.0, 2.0]
    ) == [4.0, 0.0]

    # also when the perturbation comes from an enclosing broadcast
    f(x) = sum(broadcast(a -> ReverseDiff.gradient(y -> sum(y .* a), [1.0])[1], x))
    @test_throws ArgumentError ReverseDiff.gradient(f, [2.0, 3.0])
end

@testset "zero-dimensional arrays (#265)" begin
    a = rand(1)

    @test ReverseDiff.gradient(x -> exp.(reshape(vec(x), ())), a) ≈ exp.(a)

    x0 = reshape(vec(track(copy(a), InstructionTape())), ())

    # `Base` collapses a 0-d broadcast to a scalar but keeps `map` 0-dimensional
    @test broadcast(exp, x0) isa TrackedReal
    @test map(exp, x0) isa TrackedArray
    @test ndims(map(exp, x0)) == 0
end

@testset "an untracked scalar argument survives nested differentiation (#214)" begin
    a = [0.5, 1.5, 2.5]
    b = [1.0, 2.0, 3.0]
    c = 4.0
    # `map(*, x, x ./ (b .+ c))` sums to `sum(x .^ 2 ./ (b .+ c))`
    expected = diagm(2 ./ (b .+ c))

    @test ReverseDiff.hessian(x -> sum(map(*, x, x ./ (b .+ c))), a) ≈ expected
    @test ReverseDiff.hessian(x -> sum(map(*, x, x ./ (b .+ [c]))), a) ≈ expected
end

@testset "0-dimensional broadcasts whose style carries no dimension" begin
    tp = InstructionTape()
    tr = track(2.0, tp)
    x0 = reshape(vec(track([3.0], tp)), ())

    # `TrackedStyle{Any}` cannot report 0 dimensions, so only the axes mark these as scalar
    @test result_style(TrackedStyle{0}(), ForeignStyle()) === TrackedStyle{Any}()
    @test tr .+ Foreign0d(1.0) isa TrackedReal
    @test x0 .+ Foreign0d(1.0) isa TrackedReal

    @test ReverseDiff.gradient(v -> sum(v .* Foreign0d(2.0)), [1.0, 2.0]) ≈ [2.0, 2.0]
end

@testset "a `Broadcasted` that reaches `copy` uninstantiated" begin
    # `LinearAlgebra` forwards a `Diagonal`'s broadcast to its diagonal without instantiating it
    a = rand(3)
    tp = InstructionTape()
    x, tr = track(copy(a), tp), track(2.0, tp)

    @test value(copy(Broadcast.broadcasted(exp, x))) ≈ exp.(a)
    @test copy(Broadcast.broadcasted(exp, tr)) isa TrackedReal
    @test copy(Broadcast.broadcasted(+, tr, Foreign0d(1.0))) isa TrackedReal

    @test ReverseDiff.gradient(v -> sum(Diagonal(v) .* 2.0), a) == fill(2.0, 3)

    # alongside a `TrackedArray`, the wrapper is an argument of `∇broadcast`
    m = rand(3, 3)
    gv, gm = ReverseDiff.gradient((v, w) -> sum(Diagonal(v) .* w), (a, m))
    @test gv == diag(m)
    @test gm == Diagonal(a)
end

@testset "`value` of a wrapped `TrackedArray` is not recorded" begin
    wrap(d, e, A) = (
        Diagonal(d), Bidiagonal(d, e, :L), Tridiagonal(e, d, e), SymTridiagonal(d, e),
        UpperTriangular(A), LowerTriangular(A), UnitUpperTriangular(A),
        UnitLowerTriangular(A), UpperHessenberg(A), Symmetric(A, :L),
        Hermitian(A, :L), adjoint(A), transpose(A),
    )
    d, e, A = rand(3), rand(2), rand(3, 3)
    tp = InstructionTape()
    expected = wrap(d, e, A)
    actual = map(value, wrap(track(d, tp), track(e, tp), track(A, tp)))

    @test actual == expected
    @test isempty(take_recorded!(tp))

    # the test covers every `StructuredMatrix` type
    covered = map(nameof ∘ typeof, expected)
    structured = Base.uniontypes(Base.unwrap_unionall(LinearAlgebra.StructuredMatrix))
    @test issubset(map(nameof, structured), covered)
end

@testset "tuple arguments" begin
    a = [1.0, 2.0, 3.0]
    c = (4.0, 5.0, 6.0)

    tp = InstructionTape()
    y = track(copy(a), tp) .* c
    instrs = take_recorded!(tp)
    @test y isa TrackedArray
    @test length(instrs) == 1
    @test instrs[1].func === ReverseDiff.∇broadcast
    @test ReverseDiff.gradient(x -> sum(x .* c), a) == collect(c)

    tape = ReverseDiff.GradientTape(x -> sum(exp.(x .* c)), a)
    @test ReverseDiff.gradient!(tape, a) ≈ collect(c) .* exp.(a .* c)

    # a tracked element is not seeded, so it takes the scalar rules
    f = x -> sum(x .* (x[1], 5.0, 6.0))
    @test ReverseDiff.gradient(f, a) ≈ ForwardDiff.gradient(f, a)

    # a tuple's elements are checked in turn, including a nested tuple's
    h = x -> sum(x .* first.(((x[1], 1.0), (2.0, 3.0), (4.0, 5.0))))
    @test ReverseDiff.gradient(h, a) ≈ ForwardDiff.gradient(h, a)

    tp = InstructionTape()
    track(copy(a), tp) .* first.(((1.0, 2.0), (3.0, 4.0), (5.0, 6.0)))
    @test length(take_recorded!(tp)) == 1

    # an abstract element type is checked element by element, skipping non-numbers
    b = ((1.0, nothing), (2.0, missing), (3.0, nothing))
    g = x -> sum(x .* (p -> first(p)::Float64).(b))
    @test ReverseDiff.gradient(g, a) == [1.0, 2.0, 3.0]

    tp = InstructionTape()
    track(copy(a), tp) .* (p -> first(p)::Float64).(b)
    @test length(take_recorded!(tp)) == 1
end

@testset "a `TrackedReal` without a tape is a constant" begin
    a = [1.0, 2.0]
    c = TrackedReal(2.0, 0.0)

    for (f, expected) in ((x -> x .* c, [2.0, 2.0]), (x -> exp.(x .* c), 2 .* exp.(2 .* a)))
        @test ReverseDiff.gradient(x -> sum(f(x)), a) ≈ expected
    end
end

@testset "`NotTracked`" begin
    f = ReverseDiff.NotTracked(t -> 2t)
    a = [1.0, 2.0, 3.0]

    @test ReverseDiff.gradient(x -> sum(f.(x)), rand(4)) == fill(2.0, 4)
    # a scalar broadcast takes the scalar rules, calling the wrapper itself
    @test ReverseDiff.gradient(x -> f.(x[1]), a) == [2.0, 0.0, 0.0]

    # as an argument, a struct is broadcast as a scalar and an array elementwise
    g = (t, s) -> t * s.s
    @test ReverseDiff.gradient(x -> sum(g.(x, ReverseDiff.NotTracked(Scale(2.0)))), a) ==
        fill(2.0, 3)
    @test ReverseDiff.gradient(x -> sum(g.(x, ReverseDiff.NotTracked(Scale.(a)))), a) == a
end

@testset "a partial known in closed form stores nothing" begin
    n = 10_000
    v, w = rand(n), rand(n)

    function recordbytes(f, args...)
        tp = InstructionTape()
        targs = map(a -> track(a, tp), args)
        f(targs...)
        return @allocated f(targs...)
    end

    # bytes allocated by the whole recording: the output's value and derivative take 8 B per
    # element each, and one `Dual` per element would add at least 16 B more
    for f in (
            t -> t .* 2.0, t -> 2.0 .* t, t -> t .+ 1.0, t -> 1.0 .- t, t -> t ./ 4.0,
            t -> 4.0 .\ t, t -> identity.(t), t -> .-t,
        )
        @test recordbytes(f, v) < 24n
    end
    for f in ((a, b) -> a .- b, (a, b) -> a .* b)
        @test recordbytes(f, v, w) < 24n
    end

    # an untracked array argument is copied onto the tape, 8 B per element more
    for f in (t -> t .+ w, t -> w .- t, t -> t .* w, t -> t ./ w, t -> w .\ t)
        @test recordbytes(f, v) < 32n
    end

    # a partial that depends on the point still needs one `Dual` per element
    for f in (t -> exp.(t), t -> t .^ 2, t -> 2.0 ./ t)
        @test recordbytes(f, v) > 24n
    end
    # a tracked denominator's partial, `-x/y^2`, is no argument of the broadcast
    @test recordbytes(t -> w ./ t, v) > 32n
end

@testset "a known partial differentiates as the general path does" begin
    v, w = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]

    @test ReverseDiff.gradient(t -> sum(t .* 2.0), v) == fill(2.0, 3)
    @test ReverseDiff.gradient(t -> sum(t .+ 1.0), v) == fill(1.0, 3)
    @test ReverseDiff.gradient(t -> sum(1.0 .- t), v) == fill(-1.0, 3)
    @test ReverseDiff.gradient(t -> sum(t ./ 4.0), v) == fill(0.25, 3)
    @test ReverseDiff.gradient(t -> sum(4.0 .\ t), v) == fill(0.25, 3)
    @test ReverseDiff.gradient(t -> sum(.-t), v) == fill(-1.0, 3)
    @test ReverseDiff.gradient(t -> sum(t .+ w), v) == fill(1.0, 3)
    @test ReverseDiff.gradient(t -> sum(t .* w), v) == w
    @test ReverseDiff.gradient(t -> sum(t ./ w), v) == 1 ./ w
    @test ReverseDiff.gradient(t -> sum(w .\ t), v) == 1 ./ w
    @test ReverseDiff.gradient(t -> sum(t .* t), v) == 2 .* v
    @test ReverseDiff.gradient(t -> sum(t .* Real[w...]), v) == w
    @test ReverseDiff.gradient(t -> sum(t .* [t[i] for i in 1:3]), v) == 2 .* v
    @test ReverseDiff.gradient(t -> sum(t' .* t'), v) == 2 .* v

    # `-x/y^2` is no argument of the broadcast, so it is read off a `Dual`
    @test ReverseDiff.gradient(t -> sum(w ./ t), v) ≈ -w ./ v .^ 2

    # an outer broadcast still clamps the reverse pass to the argument's own shape
    @test ReverseDiff.gradient(t -> sum(t .+ w'), v) == fill(3.0, 3)
    @test ReverseDiff.gradient(t -> sum(t .* w'), v) == fill(sum(w), 3)
    @test ReverseDiff.gradient(t -> sum(t .* 2.0), Float64[]) == Float64[]

    # a tracked scalar collects the whole output instead of one element of it
    @test ReverseDiff.gradient(t -> sum(t[1] .+ w), v) == [3.0, 0.0, 0.0]
    @test ReverseDiff.gradient(t -> sum(t .* t[1]), v) == [2 * v[1] + sum(v[2:end]), 1.0, 1.0]

    # an argument `splitargs` holds back is not one a partial may name
    @test ReverseDiff.gradient(t -> sum(Ref(2.0) .* t), v) == fill(2.0, 3)
    @test ReverseDiff.gradient(t -> sum(t .* Ref(2.0)), v) == fill(2.0, 3)

    ga, gb = ReverseDiff.gradient((a, b) -> sum(a .- b), (v, w))
    @test ga == fill(1.0, 3)
    @test gb == fill(-1.0, 3)
end

@testset "a known partial does not overflow on a subnormal divisor" begin
    # materializing `1/y` would give `Inf` before it ever meets the seed
    @test ReverseDiff.gradient(y -> 1.0e-10 * sum(y ./ 1.0e-310), [1.0, 2.0]) ==
        fill(1.0e-10 / 1.0e-310, 2)
end

@testset "a known partial replays its values" begin
    doubled(t) = t .* 2.0
    tape = ReverseDiff.GradientTape(t -> sum(exp.(doubled(t))), [1.0, 2.0, 3.0])
    x = [0.5, 1.5, 2.5]

    # `exp`'s partial reads the doubled values, so a stale replay shows up in the gradient
    @test ReverseDiff.gradient!(tape, x) ≈ 2 .* exp.(2 .* x)

    # an argument a partial names is read at replay time, not at record time
    tape2 = ReverseDiff.GradientTape(t -> sum(t .* t), [1.0, 2.0, 3.0])
    @test ReverseDiff.gradient!(tape2, x) == 2 .* x
    @test ReverseDiff.gradient!(ReverseDiff.compile(tape2), x) == 2 .* x
end

end # module
