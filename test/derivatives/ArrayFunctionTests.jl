using ForwardDiff
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

function testcat(f, args::Tuple, type, kwargs=NamedTuple())
    x = f(track.(args)...; kwargs...)
    @test x isa type
    @test value(x) == f(args...; kwargs...)

    if length(args) == 1
        x = f(track(args[1]); kwargs...)
        @test x isa type
        @test value(x) == f(args...; kwargs...)
    else
        @assert length(args) == 2

        broken = f == hcat && (args[2] isa AbstractMatrix)
        if broken
            @test_broken f(track(args[1]), args[2]; kwargs...) isa type
            @test_broken value(f(track(args[1]), args[2]; kwargs...)) == f(args...; kwargs...)
        else
            @test f(track(args[1]), args[2]; kwargs...) isa type
            @test value(f(track(args[1]), args[2]; kwargs...)) == f(args...; kwargs...)
        end

        broken = f == hcat && (args[1] isa AbstractMatrix)
        if broken
            @test_broken f(args[1], track(args[2]); kwargs...) isa type
            @test_broken value(f(args[1], track(args[2]); kwargs...)) == f(args...; kwargs...)
        else
            @test f(args[1], track(args[2]); kwargs...) isa type
            @test value(f(args[1], track(args[2]); kwargs...)) == f(args...; kwargs...)
        end
    end

    args = (args..., args...)
    sizes = size.(args)
    broken = (f in (vcat, hcat) && (args[2] isa AbstractArray))
    if broken
        @test_broken f(track.(args)...; kwargs...) isa type
        @test_broken value(f(track.(args)...; kwargs...)) == f(args...; kwargs...)
    else
        @test f(track.(args)...; kwargs...) isa type
        @test value(f(track.(args)...; kwargs...)) == f(args...; kwargs...)
    end

    F = vecx -> sum(f(unpack(sizes, vecx)...; kwargs...))
    X = pack(args)
    @test ForwardDiff.gradient(F, X) == gradient(F, X)
end

function pack(xs)
    return mapreduce(vcat, xs) do x
        x isa Number ? x : vec(x)
    end
end
function unpack(sizes, vecx)
    start = 0
    out = map(sizes) do s
        if s === ()
            x = vecx[start+1]
            start += 1
        else
            x = reshape(vecx[start+1:start+prod(s)], s)
            start += prod(s)
        end
    end
    return out
end

@testset verbose = true "cat" begin
    v = rand(3)
    m = rand(3, 3)
    a = rand(3, 3, 3)
    n = rand()

    testcat(cat, (n,), TrackedVector, (dims=1,))
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

    testcat(vcat, (n,), TrackedVector)
    testcat(vcat, (n, n), TrackedVector)
    testcat(vcat, (v, n), TrackedVector)
    testcat(vcat, (n, v), TrackedVector)
    testcat(vcat, (v, v), TrackedVector)

    testcat(hcat, (n,), TrackedMatrix)
    testcat(hcat, (n, n), TrackedMatrix)
    testcat(hcat, (v, v), TrackedMatrix)
    testcat(hcat, (v, m), TrackedMatrix)
    testcat(hcat, (m, v), TrackedMatrix)
end;
