using ReverseDiffPrototype
using Base.Test
using ForwardDiff

x = rand(5)
out = zeros(x)
testf1(x) = (exp(x[1]) + log(x[3]) * x[4]) / x[5]
g1! = ReverseDiffPrototype.gradient(testf1, x)

@test_approx_eq g1!(out, x) ForwardDiff.gradient(testf1, x)

x = rand(2)
out = zeros(x)
testf2(x) = x[1]*x[2] + sin(x[1])
g2! = ReverseDiffPrototype.gradient(testf2, x)

@test_approx_eq g2!(out, x) ForwardDiff.gradient(testf2, x)

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

g3! = ReverseDiffPrototype.gradient(testf3, x)

@test_approx_eq g3!(out, x) ForwardDiff.gradient(testf3, x)

function rosenbrock(x)
    a = one(eltype(x))
    b = 100 * a
    result = zero(eltype(x))
    for i in 1:length(x)-1
        result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2
    end
    return result
end
g4! = ReverseDiffPrototype.gradient(rosenbrock, Input{Float64,100})
x = rand(100)
out = similar(x)

@test_approx_eq g4!(out, x) ForwardDiff.gradient(rosenbrock, x)

# map of univariates
x = randn(49)
out = zeros(x)
aux_fn5(x) = sqrt(abs(x) + x^2)
function testf5(x)
    k = length(x)
    N = Int(sqrt(k))
    A = reshape(x, N, N)
    return sum(map(aux_fn5, A))
end

g5! = ReverseDiffPrototype.gradient(testf5, x)
@test_approx_eq g5!(out, x) ForwardDiff.gradient(testf5, x)
