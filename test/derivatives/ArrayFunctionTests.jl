using ReverseDiff: track, value, gradient, TrackedVector, TrackedMatrix, TrackedArray
using Test

@testset "fill" begin
    v = fill(track(1), 3)
    @test v isa TrackedArray
    @test value(v) == fill(1, 3)
    @test gradient(x -> sum(fill(x[1], 3)), rand(1)) == [3.0]
end

@testset "any & all" begin
    @test all(iszero, track(zeros(3)))
    @test !all(iszero, track([zeros(2); 1.0]))
    @test !any(iszero, track(ones(3)))
    @test any(iszero, track([ones(2); 0.0]))
end

function testcat(f, args::Tuple{Any, Any}, type, kwargs=NamedTuple())
    x = f(track.(args)...; kwargs...)
    @test x isa type
    @test value(x) == f(args...; kwargs...)

    x = f(track(args[1]), args[2]; kwargs...)
    @test x isa type
    @test value(x) == f(args...; kwargs...)

    x = f(args[1], track(args[2]); kwargs...)
    @test x isa type
    @test value(x) == f(args...; kwargs...)

    args = (args..., args...)
    x = f(track.(args)...; kwargs...)
    @test x isa type
    @test value(x) == f(args...; kwargs...)
end

@testset "cat" begin
    v = rand(3)
    m = rand(3,3)
    a = rand(3,3,3)
    n = rand()

    testcat(cat, (n, n), TrackedVector, (dims=1,))
    testcat(cat, (n, n), TrackedMatrix, (dims=2,))
    testcat(cat, (v, n), TrackedVector, (dims=1,))
    testcat(cat, (n, v), TrackedVector, (dims=1,))

    testcat(cat, (v, v), TrackedVector, (dims=1,))
    testcat(cat, (v, v), TrackedMatrix, (dims=2,))
    testcat(cat, (v, m), TrackedMatrix, (dims=2,))
    testcat(cat, (m, v), TrackedMatrix, (dims=2,))

    testcat(cat, (a, a), TrackedArray, (dims=1,))
    testcat(cat, (a, a), TrackedArray, (dims=2,))
    testcat(cat, (a, a), TrackedArray, (dims=3,))
    testcat(cat, (a, m), TrackedArray, (dims=3,))

    testcat(vcat, (n, n), TrackedVector)
    testcat(vcat, (v, n), TrackedVector)
    testcat(vcat, (n, v), TrackedVector)
    testcat(vcat, (v, v), TrackedVector)

    testcat(hcat, (n, n), TrackedMatrix)
    testcat(hcat, (v, v), TrackedMatrix)
    testcat(hcat, (v, m), TrackedMatrix)
    testcat(hcat, (m, v), TrackedMatrix)
end
