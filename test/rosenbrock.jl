using ReverseDiffPrototype

function rosenbrock(x)
    a = one(eltype(x))
    b = 100 * a
    result = zero(eltype(x))
    for i in 1:length(x)-1
        result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2
    end
    return result
end

x = rand(100000);
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
