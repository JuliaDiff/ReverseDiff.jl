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
t = ReverseDiffPrototype.trace_input_array(rosenbrock, x, eltype(out));

# warmup
trace = ReverseDiffPrototype.reset_trace!(rosenbrock)
rosenbrock(t);
ReverseDiffPrototype.backprop!(trace);
ReverseDiffPrototype.gradient!(out, rosenbrock, x, t);

# timed
trace = ReverseDiffPrototype.reset_trace!(rosenbrock)
gc()
@time rosenbrock(t);
gc()
@time ReverseDiffPrototype.backprop!(trace);
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
t = ReverseDiffPrototype.trace_input_array(ackley, x, eltype(out));

# warmup
trace = ReverseDiffPrototype.reset_trace!(ackley)
ackley(t);
ReverseDiffPrototype.backprop!(trace);
ReverseDiffPrototype.gradient!(out, ackley, x, t);

# timed
trace = ReverseDiffPrototype.reset_trace!(ackley)
gc()
@time ackley(t);
gc()
@time ReverseDiffPrototype.backprop!(trace);
gc()
@time ReverseDiffPrototype.gradient!(out, ackley, x, t);

println("done")
