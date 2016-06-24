using ReverseDiffPrototype
using Base.Test
using ForwardDiff

x = rand(5)
out = zeros(x)
testf(x) = (exp(x[1]) + log(x[3]) * x[4]) / x[5]

@test_approx_eq ReverseDiffPrototype.gradient!(out, testf, x) ForwardDiff.gradient(testf, x)

x = rand(2)
out = zeros(x)
testf2(x) = x[1]*x[2] + sin(x[1])

@test_approx_eq ReverseDiffPrototype.gradient!(out, testf2, x) ForwardDiff.gradient(testf2, x)

const N = 10
x = rand(2N^2 + N)
out = zeros(x)
function testf3(x)
    k = length(x)
    A = reshape(x[1:N^2], N, N)
    B = reshape(x[N^2 + 1:2N^2], N, N)
    c = x[2N^2+1:end]
    return trace(log(A * B .+ c))
end

@test_approx_eq ReverseDiffPrototype.gradient!(out, testf3, x) ForwardDiff.gradient(testf3, x)
