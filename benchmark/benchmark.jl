using ReverseDiffPrototype

const N = 100000

println("benchmarking rosenbrock(x)...")

function rosenbrock(x)
    a = one(eltype(x))
    b = 100 * a
    result = zero(eltype(x))
    for i in 1:length(x)-1
        result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2
    end
    return result
end

x = rand(N);
out = zeros(x);
t = ReverseDiffPrototype.trace_array(typeof(rosenbrock), eltype(out), x);

# warmup
ftrace = ReverseDiffPrototype.reset_trace!(rosenbrock)
rosenbrock(t);
ReverseDiffPrototype.backprop!(ftrace);
ReverseDiffPrototype.gradient!(out, rosenbrock, x, t);

# timed
ftrace = ReverseDiffPrototype.reset_trace!(rosenbrock)
gc()
@time rosenbrock(t);
gc()
@time ReverseDiffPrototype.backprop!(ftrace);
gc()
@time ReverseDiffPrototype.gradient!(out, rosenbrock, x, t);

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

x = rand(N);
out = zeros(x);
t = ReverseDiffPrototype.trace_array(typeof(ackley), eltype(out), x);

# warmup
ftrace = ReverseDiffPrototype.reset_trace!(ackley)
ackley(t);
ReverseDiffPrototype.backprop!(ftrace);
ReverseDiffPrototype.gradient!(out, ackley, x, t);

# timed
ftrace = ReverseDiffPrototype.reset_trace!(ackley)
gc()
@time ackley(t);
gc()
@time ReverseDiffPrototype.backprop!(ftrace);
gc()
@time ReverseDiffPrototype.gradient!(out, ackley, x, t);

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
t = ReverseDiffPrototype.trace_array(typeof(matrix_test), eltype(out), x);

# warmup
ftrace = ReverseDiffPrototype.reset_trace!(matrix_test)
matrix_test(t);
ReverseDiffPrototype.backprop!(ftrace);
ReverseDiffPrototype.gradient!(out, matrix_test, x, t);

# timed
ftrace = ReverseDiffPrototype.reset_trace!(matrix_test)
gc()
@time matrix_test(t);
gc()
@time ReverseDiffPrototype.backprop!(ftrace);
gc()
@time ReverseDiffPrototype.gradient!(out, matrix_test, x, t);

println("done")
