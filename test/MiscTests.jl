module MiscTests

using ReverseDiffPrototype
using Base.Test
using ForwardDiff

const RDP = ReverseDiffPrototype

##################################################
x = rand(5)
out = zeros(x)

test1(x) = (exp(x[1]) + log(x[3]) * x[4]) / x[5]

Main.testprintln(test1)

@test_approx_eq RDP.gradient!(out, test1, x) ForwardDiff.gradient(test1, x)

##################################################
x = rand(2)
out = zeros(x)

test2(x) = x[1]*x[2] + sin(x[1])

Main.testprintln(test2)

@test_approx_eq RDP.gradient!(out, test2, x) ForwardDiff.gradient(test2, x)

##################################################
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

test3 = generate_matrix_test(n)

Main.testprintln(test3)

@test_approx_eq RDP.gradient!(out, test3, x) ForwardDiff.gradient(test3, x)

##################################################
x = rand(1:10, 49)
out = zeros(Float64, 49)

function test4(x)
    k = length(x)
    N = isqrt(k)
    A = reshape(x, N, N)
    return sum(@fastdiff map(n -> sqrt(abs(n) + n^2) * 0.5, A))
end

Main.testprintln(test4)

@test_approx_eq RDP.gradient!(out, test4, x) ForwardDiff.gradient(test4, x)

##################################################
x = rand(10)
out = similar(x)

function rosenbrock1(x)
    a = one(eltype(x))
    b = 100 * a
    result = zero(eltype(x))
    for i in 1:length(x)-1
        result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2
    end
    return result
end

Main.testprintln(rosenbrock1)

@test_approx_eq RDP.gradient!(out, rosenbrock1, x) ForwardDiff.gradient(rosenbrock1, x)

##################################################
x = rand(10)
out = similar(x)

function rosenbrock2(x)
    a = x[1]
    b = 100 * a
    v = map((i, j) -> (a - j)^2 + b*(i - j^2)^2, x[2:end], x[1:end-1])
    return sum(v)
end

Main.testprintln(rosenbrock2)

@test_approx_eq RDP.gradient!(out, rosenbrock2, x) ForwardDiff.gradient(rosenbrock2, x)

end # module
