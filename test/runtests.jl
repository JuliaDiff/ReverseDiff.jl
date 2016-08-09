using ReverseDiffPrototype
using Base.Test
using ForwardDiff

const RDP = ReverseDiffPrototype

##################################################
println("running test1...")

x = rand(5)
out = zeros(x)

test1(x) = (exp(x[1]) + log(x[3]) * x[4]) / x[5]

@test_approx_eq RDP.gradient!(out, test1, x) ForwardDiff.gradient(test1, x)

##################################################
println("running test2...")

x = rand(2)
out = zeros(x)

test2(x) = x[1]*x[2] + sin(x[1])

@test_approx_eq RDP.gradient!(out, test2, x) ForwardDiff.gradient(test2, x)

##################################################
println("running matrix_test...")

n = 2
x = collect(1:(2n^2 + n))
out = zeros(Float64, x)

function generate_matrix_test(n)
    return x -> begin
        @assert length(x) == 2n^2 + n
        a = reshape(x[1:n^2], n, n)
        b = reshape(x[n^2 + 1:2n^2], n, n)
        return trace(log((a * b) + a - b))
    end
end

matrix_test = generate_matrix_test(n)

@test_approx_eq RDP.gradient!(out, matrix_test, x) ForwardDiff.gradient(test3, x)

##################################################
println("running test4...")

x = rand(1:10, 49)
out = zeros(Float64, 49)

function test4(x)
    k = length(x)
    N = Int(sqrt(k))
    A = reshape(x, N, N)
    return sum(map(n -> sqrt(abs(n) + n^2) * 0.5, A))
end

@test_approx_eq RDP.gradient!(out, test4, x) ForwardDiff.gradient(test4, x)

##################################################
println("testing rosenbrock...")

x = rand(100)
out = similar(x)

function rosenbrock(x)
    a = one(eltype(x))
    b = 100 * a
    result = zero(eltype(x))
    for i in 1:length(x)-1
        result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2
    end
    return result
end

@test_approx_eq RDP.gradient!(out, rosenbrock, x) ForwardDiff.gradient(rosenbrock, x)
