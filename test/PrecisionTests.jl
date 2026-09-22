module PrecisionTests

using ReverseDiff, Test, LinearAlgebra, Statistics

include("utils.jl")

# gradients must be accurate to the working precision (#239)
setprecision(BigFloat, 256) do
    x = big.(rand(5)) ./ 3
    n = length(x)
    T = eltype(x)
    rtol = 8 * eps(T)

    for (f, ∇f) in ((mean,                    y -> fill(one(T) / n, n)),
                    (y -> mean(y .^ 3),       y -> 3 .* y .^ 2 ./ n),
                    (y -> sum(y ./ 3),        y -> fill(one(T) / 3, n)),
                    (y -> sum(y ./ 3.0),      y -> fill(one(T) / 3, n)),
                    (y -> sum(3 .\ y),        y -> fill(one(T) / 3, n)),
                    (y -> sum(3.0 .\ y),      y -> fill(one(T) / 3, n)),
                    (sum,                     y -> fill(one(T), n)),
                    (prod,                    y -> prod(y) ./ y),
                    (norm,                    y -> y ./ norm(y)),
                    (var,                     y -> 2 .* (y .- mean(y)) ./ (n - 1)),
                    (std,                     y -> (y .- mean(y)) ./ ((n - 1) * std(y))),
                    (y -> dot(y, y),          y -> 2 .* y),
                    (y -> sum(3 ./ y),        y -> -3 ./ y .^ 2),
                    (y -> sum(y .\ 3),        y -> -3 ./ y .^ 2),
                    (y -> sum(y ./ (y .+ 1)), y -> 1 ./ (y .+ 1) .^ 2),
                    (y -> sum(exp.(y)),       y -> exp.(y)))
        test_println("BigFloat gradients", f)
        @test ReverseDiff.gradient(f, x) ≈ ∇f(x) rtol=rtol
    end

    let f = y -> sum(y .^ 2 ./ 7)
        test_println("BigFloat gradients", f)
        @test ReverseDiff.gradient(f, x) ≈ 2 .* x ./ 7 rtol=rtol
    end
end

# a subnormal denominator must not overflow the gradient
let f = y -> 1e-10 * sum(y ./ 1e-310)
    test_println("Float64 gradients", f)
    @test ReverseDiff.gradient(f, [1.0, 2.0]) == fill(1e-10 / 1e-310, 2)
end

# the denominator adjoint is `-n / d^2`, which must not overflow while it is still finite
for (n, d) in ((1e-200, 1e-200), (1e200, 1e200))
    ∂ = Float64(-big(n) / big(d)^2)
    for f in (y -> sum([n] ./ y), y -> sum(y .\ [n]))
        test_println("Float64 gradients", f)
        @test ReverseDiff.gradient(f, [d]) ≈ [∂] rtol=8*eps(Float64)
    end
end

end # module
