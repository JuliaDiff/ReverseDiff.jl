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
struct Foreign0d{T} <: AbstractArray{T,0}
    x::T
end
Base.size(::Foreign0d) = ()
Base.getindex(f::Foreign0d) = f.x
Base.getindex(f::Foreign0d, ::CartesianIndex{0}) = f.x
Base.Broadcast.BroadcastStyle(::Type{<:Foreign0d}) = ForeignStyle()
Base.similar(::Broadcasted{ForeignStyle}, ::Type{T}, axs) where {T} = similar(Array{T}, axs)

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
    for (a, b, expected) in ((TrackedStyle{0}(), DefaultArrayStyle{0}(), TrackedStyle{0}()),
                             (TrackedStyle{0}(), DefaultArrayStyle{2}(), TrackedStyle{2}()),
                             (TrackedStyle{1}(), DefaultArrayStyle{1}(), TrackedStyle{1}()),
                             (TrackedStyle{1}(), TrackedStyle{2}(), TrackedStyle{2}()),
                             (TrackedStyle{1}(), Unknown(), TrackedStyle{1}()),
                             (TrackedStyle{1}(), ForeignStyle(), TrackedStyle{Any}()),
                             (TrackedStyle{Any}(), ForeignStyle(), TrackedStyle{Any}()),
                             (TrackedStyle{Any}(), DefaultArrayStyle{1}(), TrackedStyle{Any}()),
                             # a scalar loses to a tuple, as in `Base`
                             (TrackedStyle{0}(), Style{Tuple}(), Style{Tuple}()),
                             (TrackedStyle{1}(), Style{Tuple}(), TrackedStyle{1}()))
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
end

@testset "no `BroadcastStyle` ambiguities" begin
    # `ReverseDiff` has unrelated pre-existing ambiguities, so filter to broadcasting
    @test isempty(filter(Test.detect_ambiguities(ReverseDiff)) do (m1, m2)
        m1.name === :BroadcastStyle || m2.name === :BroadcastStyle
    end)
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

    for (dotted, called) in ((tp -> exp.(track(a, tp)), tp -> broadcast(exp, track(a, tp))),
                             (tp -> track(a, tp) .+ track(b, tp), tp -> broadcast(+, track(a, tp), track(b, tp))),
                             (tp -> atan.(track(a, tp), b), tp -> broadcast(atan, track(a, tp), b)))
        tp1, tp2 = InstructionTape(), InstructionTape()
        x, y = dotted(tp1), called(tp2)

        @test value(x) == value(y)
        @test length(tp1) == length(tp2) == 1
        @test tp1[1].func === tp2[1].func
    end
end

@testset "`@forward` and `@skip` wrappers do not force the fallback path" begin
    for wrapper in (ReverseDiff.ForwardOptimize, ReverseDiff.SkipOptimize)
        tp = InstructionTape()
        x = track(rand(3, 3), tp)

        y = broadcast(wrapper(exp), x)

        @test y isa TrackedArray
        @test length(tp) == 1
    end
end

@testset "a type broadcast as a function does not force the fallback path" begin
    tp = InstructionTape()
    x = track(rand(3), tp)

    y = Real.(x)

    @test y isa TrackedArray
    @test length(tp) == 1
    @test tp[1].func === ReverseDiff.∇broadcast
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
    @test length(tp) == 2
end

@testset "`TrackedArray`s are rejected as broadcast destinations" begin
    a = rand(4)
    msg = "`TrackedArray`s do not support `setindex!` and cannot be used as a broadcast destination. Use `y = f.(x)` instead."

    tp = InstructionTape()
    x = track(copy(a), tp)
    @test_throws ArgumentError(msg) track(zeros(4), tp) .= exp.(x)
    @test_throws ArgumentError(msg) track(zeros(4), tp) .= a
    # nothing may be recorded before the failure
    @test isempty(tp)

    # an untracked container of tracked elements is a valid destination
    dest = Vector{TrackedReal{Float64,Float64,Nothing}}(undef, 4)
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

    @test y isa TrackedArray
    @test value(y) ≈ a .* b
    @test length(tp) == 1
    @test tp[1].func === ReverseDiff.∇broadcast

    ReverseDiff.seed!(sum(y))
    ReverseDiff.reverse_pass!(tp)
    # `deriv.(x)` would be traced like any other broadcast, as it is for a `TrackedArray`
    @test map(deriv, x) ≈ b
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
end

@testset "a foreign tag is never read as our own" begin
    tagA = typeof(ForwardDiff.Tag(sin, Float64))
    tagB = typeof(ForwardDiff.Tag(cos, Float64))
    # `Tag`s are ordered by first use, so pin the order before relying on it
    @test ForwardDiff.:≺(tagA, tagB)

    concrete = ForwardDiff.Dual{tagA,Float64,1}[ForwardDiff.Dual{tagA}(2.0, 3.0)]
    widened = Union{Float64,ForwardDiff.Dual{tagA,Float64,1}}[
        1.0, ForwardDiff.Dual{tagA}(2.0, 3.0)]
    nested = Union{Float64,ForwardDiff.Dual{tagB,Float64,1}}[1.0]

    @test ReverseDiff.trackresults(tagB, concrete, identity, identity, (), Float64) === concrete
    @test ReverseDiff.trackresults(tagB, widened, identity, identity, (), Float64) === widened
    @test_throws ForwardDiff.DualMismatchError ReverseDiff.trackresults(
        tagA, nested, identity, identity, (), Float64)

    @test ReverseDiff.getpartial(tagA, ForwardDiff.Dual{tagA}(1.0, 2.0), 1) == 2.0
    @test ReverseDiff.getpartial(tagA, 1.0, 1) == 0.0
    @test ReverseDiff.getpartial(tagB, ForwardDiff.Dual{tagA}(1.0, 2.0), 1) == 0.0
    @test_throws ForwardDiff.DualMismatchError ReverseDiff.getpartial(
        tagA, ForwardDiff.Dual{tagB}(1.0, 2.0), 1)
end

@testset "a perturbation the derivative cannot hold is rejected (#67, #168)" begin
    # `a` reaches the broadcast as a `Dual`, but the tape's derivatives are `Float64`
    g(a) = sum(ReverseDiff.gradient(x -> sum(x .* a), [1.0, 2.0]))
    E = ForwardDiff.Dual{ForwardDiff.Tag{typeof(g),Float64},Float64,1}
    msg = "a broadcast argument with element type $E carries a perturbation that a derivative of type Float64 cannot hold"

    @test_throws ArgumentError(msg) ForwardDiff.derivative(g, 3.0)

    # a widened element type is checked member by member
    g2(a) = sum(ReverseDiff.gradient(x -> sum(x .* Union{Float64,typeof(a)}[1.0, a]), [1.0, 2.0]))
    E2 = ForwardDiff.Dual{ForwardDiff.Tag{typeof(g2),Float64},Float64,1}
    msg2 = "a broadcast argument with element type $E2 carries a perturbation that a derivative of type Float64 cannot hold"
    @test_throws ArgumentError(msg2) ForwardDiff.derivative(g2, 3.0)

    # a tape whose derivatives are themselves `Dual`s can hold it, so it is left alone
    h(a) = ReverseDiff.gradient(x -> sum(x .* a), [ForwardDiff.Dual(1.0, 0.0)])
    @test h(3.0) == [ForwardDiff.Dual(3.0, 0.0)]
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

@testset "`NotTracked`" begin
    f = ReverseDiff.NotTracked(t -> 2t)

    @test ReverseDiff.gradient(x -> sum(f.(x)), rand(4)) == fill(2.0, 4)
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

    # the output's value and derivative take 8 B per element each, and one `Dual` per
    # element would add at least 16 B more
    for f in (t -> t .* 2.0, t -> 2.0 .* t, t -> t .+ 1.0, t -> 1.0 .- t, t -> t ./ 4.0,
              t -> 4.0 .\ t, t -> identity.(t), t -> .-t)
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

    # `-x/y^2` is no argument of the broadcast, so it is read off a `Dual`
    @test ReverseDiff.gradient(t -> sum(w ./ t), v) == -w ./ v .^ 2

    # an outer broadcast still clamps the reverse pass to the argument's own shape
    @test ReverseDiff.gradient(t -> sum(t .+ w'), v) == fill(3.0, 3)
    @test ReverseDiff.gradient(t -> sum(t .* w'), v) == fill(sum(w), 3)
    @test ReverseDiff.gradient(t -> sum(t .* 2.0), Float64[]) == Float64[]

    # a tracked scalar collects the whole output instead of one element of it
    @test ReverseDiff.gradient(t -> sum(t[1] .+ w), v) == [3.0, 0.0, 0.0]
    @test ReverseDiff.gradient(t -> sum(t .* t[1]), v) == [2 * v[1] + sum(v[2:end]), 1.0, 1.0]

    # an argument `splitargs` holds back is not one a partial may name
    @test ReverseDiff.gradient(t -> sum(broadcast(*, t, w, Ref(2.0))), v) == 2 .* w

    ga, gb = ReverseDiff.gradient((a, b) -> sum(a .- b), (v, w))
    @test ga == fill(1.0, 3)
    @test gb == fill(-1.0, 3)
end

@testset "a known partial does not overflow on a subnormal divisor" begin
    # materializing `1/y` would give `Inf` before it ever meets the seed
    @test ReverseDiff.gradient(y -> 1e-10 * sum(y ./ 1e-310), [1.0, 2.0]) ==
        fill(1e-10 / 1e-310, 2)
end

@testset "a known partial replays its values" begin
    doubled(t) = t .* 2.0
    tape = ReverseDiff.GradientTape(t -> sum(exp.(doubled(t))), [1.0, 2.0, 3.0])
    x = [0.5, 1.5, 2.5]

    # `exp`'s partial reads the doubled values, so a stale replay shows up in the gradient
    @test ReverseDiff.gradient!(tape, x) ≈ 2 .* exp.(2 .* x)

    # an argument a partial names is read at replay time, not at record time
    scaled(t) = t .* [2.0, 3.0, 4.0]
    tape2 = ReverseDiff.GradientTape(t -> sum(scaled(t)), [1.0, 2.0, 3.0])
    @test ReverseDiff.gradient!(tape2, x) == [2.0, 3.0, 4.0]
end

end # module
