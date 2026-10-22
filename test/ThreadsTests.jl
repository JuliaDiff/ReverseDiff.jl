using ReverseDiff, Test

# Run on a multithreaded worker, see `test_worker` in runtests.jl
@test Threads.nthreads() > 1

@testset "recording from multiple threads (#218)" begin
    function g(x, i)
        acc = zero(eltype(x))
        for _ in 1:100
            acc += (i * x[1] - x[2])^2
        end
        return acc + sum(abs2, i .* x)
    end
    function f(x)
        s = Vector{Any}(undef, 64)
        @sync for i in eachindex(s)
            Threads.@spawn s[i] = g(x, i)
        end
        return sum(s)
    end
    f_serial(x) = sum(i -> g(x, i), 1:64)

    x = [1.0, 2.0]
    ∇f = ReverseDiff.gradient(f_serial, x)
    for _ in 1:20
        @test ReverseDiff.gradient(f, x) ≈ ∇f
    end

    # tape replay
    tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, x))
    @test ReverseDiff.gradient!(similar(x), tape, x) ≈ ∇f
end
