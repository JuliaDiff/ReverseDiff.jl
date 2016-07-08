using ReverseDiffPrototype
using Base.Test
using ForwardDiff

const RDP = ReverseDiffPrototype

##################################################

x = rand(5)
out = zeros(x)

test1(x) = (exp(x[1]) + log(x[3]) * x[4]) / x[5]

@test_approx_eq RDP.gradient!(out, test1, x) ForwardDiff.gradient(test1, x)

##################################################

x = rand(2)
out = zeros(x)

test2(x) = x[1]*x[2] + sin(x[1])

@test_approx_eq RDP.gradient!(out, test2, x) ForwardDiff.gradient(test2, x)

##################################################

n = 2
x = collect(1:(2n^2 + n))
t = RDP.trace_input_array(x, Int)
out = zeros(Float64, length(x))

function generate_test3(n)
    return x -> begin
        @assert length(x) == 2n^2 + n
        A = reshape(x[1:n^2], n, n)
        B = reshape(x[n^2 + 1:2n^2], n, n)
        C = x[2n^2+1:end]
        return trace(log(A * B .+ C))
    end
end

test3 = generate_test3(n)

@test_approx_eq RDP.gradient!(out, test3, x) ForwardDiff.gradient(test3, x)

##################################################

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

##################################################

x = rand(1:10, 49)
out = zeros(Float64, 49)

function test5(x)
    k = length(x)
    N = Int(sqrt(k))
    A = reshape(x, N, N)
    return sum(map(n -> sqrt(abs(n) + n^2) * 0.5, A))
end

@test_approx_eq RDP.gradient!(out, test5, x) ForwardDiff.gradient(test5, x)
