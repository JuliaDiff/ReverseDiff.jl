include("elementwise_utils.jl")

for f in DiffTests.NUMBER_TO_NUMBER_FUNCS
    test_println("DiffTests.NUMBER_TO_NUMBER_FUNCS", f)
    test_elementwise(f, ReverseDiff.@forward(f), x, tp)
    test_elementwise(f, ReverseDiff.@forward(f), a, tp)
end

for (M, fsym, arity) in DiffRules.diffrules(; filter_modules = nothing)
    # ensure that all rules can be tested
    if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), fsym))
        error("$M.$fsym is not available")
    end
    (M, fsym) in ReverseDiff.SKIPPED_DIFFRULES && continue
    if arity == 1
        f = eval(:($M.$fsym))
        test_println("forward-mode unary scalar functions", f)
        test_elementwise(f, f, modify_input(fsym, x), tp)
        test_elementwise(f, f, modify_input(fsym, a), tp)
    end
    # binary rules are tested in ElementwiseBinaryTests.jl
end

for f in DiffTests.BINARY_BROADCAST_OPS
    test_println("built-in broadcast operators", f)
    test_broadcast(f, f, x, y, tp, true)
    test_broadcast(f, f, a, b, tp, true)
    test_broadcast(f, f, x, a, tp, true)
    test_broadcast(f, f, a, x, tp, true)
    test_broadcast(f, f, n, x, tp, true)
    test_broadcast(f, f, x, n, tp, true)
    test_broadcast(f, f, n, a, tp, true)
    test_broadcast(f, f, a, n, tp, true)
end

@testset "broadcasting over types without a definite field count (#289)" begin
    T = ReverseDiff.TrackedReal{Float64, Float64, ReverseDiff.TrackedArray{Float64, Float64, 1, Vector{Float64}, Vector{Float64}}}
    msg = "Converting an instance of $T to Float64 is not defined. Please use `ReverseDiff.value` instead."
    @test_throws ArgumentError(msg) ReverseDiff.gradient(v -> sum(convert.(Float64, v) .* [2.0, 3.0]), [0.3, 0.5])

    @test ReverseDiff.gradient(v -> sum(convert.(Real, v) .* [2.0, 3.0]), [0.3, 0.5]) == [2.0, 3.0]
    @test ReverseDiff.gradient(v -> sum(v .* Any[2.0, 3.0]), [0.3, 0.5]) == [2.0, 3.0]
    @test ReverseDiff.gradient(v -> sum(v .* Base.RefValue{Any}(2.0)), [0.3, 0.5]) == [2.0, 2.0]
end

@testset "`map` over a tracked scalar and an array" begin
    c = [1.5, 2.5]

    # `map` stops at its shortest argument, a scalar counting as a one-element collection,
    # so each case is `atan` of the first elements alone
    cases = (
        (t -> sum(map(atan, t[1], c)), (a, b) -> [c[1], 0] / (a^2 + c[1]^2)),
        (t -> sum(map(atan, c, t[1])), (a, b) -> [-c[1], 0] / (a^2 + c[1]^2)),
        (t -> sum(map(atan, t, t[2])), (a, b) -> [b, -a] / (a^2 + b^2)),
        (t -> sum(map(atan, t[2], t)), (a, b) -> [-b, a] / (a^2 + b^2)),
    )
    for (f, ∇f) in cases
        v = [0.5, 0.7]
        @test ReverseDiff.gradient(f, v) ≈ ∇f(v...)

        # the replay reaches the scalar through the tape
        tape = ReverseDiff.GradientTape(f, v)
        w = [0.2, 0.9]
        @test ReverseDiff.gradient!(tape, w) ≈ ∇f(w...)
    end
end

@testset "`map` over arrays of different lengths" begin
    v = [0.5, 0.7, 0.9]
    c = [1.5, 2.5]

    for f in (
            t -> sum(map(*, t, c)), t -> sum(map(+, c, t)),
            t -> sum(map(atan, t, 2 .* t[2:3])),
        )
        @test ReverseDiff.gradient(f, v) ≈ ForwardDiff.gradient(f, v)
    end
end

@testset "`map` over a `TrackedArray` and an array that may hold `TrackedReal`s" begin
    c = [1.0, 2.0, 3.0]
    ∇hypot(a, b) = [a ./ hypot.(a, b); b ./ hypot.(a, b)]

    cases = (
        (t -> sum(map(hypot, t[1:3], [t[4], t[5], t[6]])), t -> ∇hypot(t[1:3], t[4:6])),
        (t -> sum(map(hypot, [t[1], t[2], t[3]], t[4:6])), t -> ∇hypot(t[1:3], t[4:6])),
        (t -> sum(map(hypot, t[1:3], Real[c...])), t -> [∇hypot(t[1:3], c)[1:3]; zeros(3)]),
    )
    for (f, ∇f) in cases
        v = rand(6)
        @test ReverseDiff.gradient(f, v) ≈ ∇f(v)

        tape = ReverseDiff.GradientTape(f, v)
        w = rand(6)
        @test ReverseDiff.gradient!(tape, w) ≈ ∇f(w)
    end
end

@testset "`map` with a skipped function is untracked" begin
    a, b, c = [1.0, 2.0, 3.5], [2.0, 2.0, 1.0], 2.0

    tp = InstructionTape()
    at, bt = track(a, tp), track(b, tp)

    for fsym in ReverseDiff.SKIPPED_UNARY_SCALAR_FUNCS
        f = eval(fsym)
        @test map(f, at) == map(f, a)
    end
    for fsym in ReverseDiff.SKIPPED_BINARY_SCALAR_FUNCS
        f = eval(fsym)
        for (x, y) in ((at, bt), (a, bt), (at, b), (c, bt), (at, c))
            @test map(f, x, y) == map(f, ReverseDiff.value(x), ReverseDiff.value(y))
        end
    end
    @test isempty(take_recorded!(tp))
end
