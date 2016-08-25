using ReverseDiffPrototype

const RDP = ReverseDiffPrototype
const N = 100000

println("benchmarking rosenbrock(x)...")

rosenbrock(x) = sum(@fastdiff map((i, j) -> (1 - j)^2 + 100*(i - j^2)^2, x[2:end], x[1:end-1]))

x = rand(N)
out = zeros(x)
tr = RDP.Trace()
trx = RDP.wrap(eltype(out), x, tr)

# warmup
RDP.seed!(rosenbrock(trx))
RDP.backprop!(tr)
empty!(tr)
RDP.gradient!(out, rosenbrock, x, trx)
empty!(tr)

# timed
gc()
@time RDP.seed!(rosenbrock(trx))
gc()
@time RDP.backprop!(tr)
empty!(tr)
gc()
@time RDP.gradient!(out, rosenbrock, x, trx)
empty!(tr)

println("done")
println("benchmarking ackley(x)...")

function ackley(x::AbstractVector)
    a, b, c = 20.0, -0.2, 2.0*π
    len_recip = inv(length(x))
    sum_sqrs = zero(eltype(x))
    sum_cos = sum_sqrs
    for i in x
        sum_cos += cos(c*i)
        sum_sqrs += i^2
    end
    return (-a * exp(b * sqrt(len_recip*sum_sqrs)) -
            exp(len_recip*sum_cos) + a + e)
end

x = rand(N)
out = zeros(x)
tr = RDP.Trace()
trx = RDP.wrap(eltype(out), x, tr)

# warmup
RDP.seed!(ackley(trx))
RDP.backprop!(tr)
empty!(tr)
RDP.gradient!(out, ackley, x, trx)
empty!(tr)

# timed
gc()
@time RDP.seed!(ackley(trx))
gc()
@time RDP.backprop!(tr)
empty!(tr)
gc()
@time RDP.gradient!(out, ackley, x, trx)
empty!(tr)

println("benchmarking matrix_test(x)...")

function generate_matrix_test(n)
    return x -> begin
        @assert length(x) == 2n^2 + n
        a = reshape(x[1:n^2], n, n)
        b = reshape(x[n^2 + 1:2n^2], n, n)
        return trace(log((a * b) + a - b))
    end
end

n = 100
x = collect(1:(2n^2 + n))
out = zeros(Float64, x)
matrix_test = generate_matrix_test(n)
tr = RDP.Trace()
trx = RDP.wrap(eltype(out), x, tr)

# warmup
RDP.seed!(matrix_test(trx))
RDP.backprop!(tr)
empty!(tr)
RDP.gradient!(out, matrix_test, x, trx)
empty!(tr)

# timed
gc()
@time RDP.seed!(matrix_test(trx))
gc()
@time RDP.backprop!(tr)
empty!(tr)
gc()
@time RDP.gradient!(out, matrix_test, x, trx)
empty!(tr)

println("done")
