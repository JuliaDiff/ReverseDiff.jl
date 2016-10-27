using ReverseDiff

####################################################################

function grad_benchmark_driver(f, x)
    println("benchmarking ∇$(f)(x) on $(length(x)) elements...")

    out = zeros(x)
    opts = ReverseDiff.Options(x, ReverseDiff.Tape())
    tp = opts.tape
    xt = opts.state

    # warmup
    ReverseDiff.track!(xt, x, tp)
    ReverseDiff.seed!(f(xt))
    ReverseDiff.reverse_pass!(tp)
    empty!(tp)
    ReverseDiff.gradient!(out, f, x, opts)
    empty!(tp)

    # actual
    ReverseDiff.track!(xt, x, tp)
    gc()
    @time ReverseDiff.seed!(f(xt))
    gc()
    @time ReverseDiff.reverse_pass!(tp)
    empty!(tp)
    gc()
    @time ReverseDiff.gradient!(out, f, x, opts)
    empty!(tp)

    println("done.")
end

####################################################################

rosenbrock(x) = sum(map(ReverseDiff.@forward((i, j) -> (1 - j)^2 + 100*(i - j^2)^2), x[2:end], x[1:end-1]))

grad_benchmark_driver(rosenbrock, rand(100000))

####################################################################

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

grad_benchmark_driver(ackley, rand(100000))

####################################################################

function generate_matrix_test(n)
    return x -> begin
        @assert length(x) == 2n^2 + n
        a = reshape(x[1:n^2], n, n)
        b = reshape(x[n^2 + 1:2n^2], n, n)
        return trace(log((a * b) + a - b))
    end
end

n = 100
matrix_test = generate_matrix_test(n)

grad_benchmark_driver(matrix_test, collect(1.0:(2n^2 + n)))

####################################################################

relu(x) = log.(1.0 .+ exp(x))

ReverseDiff.@forward sigmoid(n) = 1. / (1. + exp(-n))

function neural_net(w1, w2, w3, x1)
    x2 = relu(w1 * x1)
    x3 = relu(w2 * x2)
    return sigmoid(dot(w3, x3))
end

neural_net_grads!(outputs, inputs) = ReverseDiff.gradient!(outputs, neural_net, inputs)

inputs = (randn(10,10), randn(10,10), randn(10), rand(10))
outputs = map(similar, inputs)

println("benchmarking neural_net_grads...")

neural_net_grads!(outputs, inputs) # warmup
gc()
@time neural_net_grads!(outputs, inputs) # actual

println("done")
