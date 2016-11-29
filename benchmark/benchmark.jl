using ReverseDiff

####################################################################

function grad_benchmark_driver!(out, f, x)
    println("benchmarking ∇$(f)...")

    cfg = ReverseDiff.GradientConfig(x)
    tp = ReverseDiff.GradientTape(f, x)

    # warmup
    ReverseDiff.gradient!(out, f, x, cfg)
    ReverseDiff.gradient!(out, tp, x)
    ReverseDiff.forward_pass!(tp)
    ReverseDiff.reverse_pass!(tp)

    # actual
    gc()
    print("  gradient! (no precord): ")
    @time ReverseDiff.gradient!(out, f, x, cfg)
    gc()
    print("  gradient!    (precord): ")
    @time ReverseDiff.gradient!(out, tp, x)
    gc()
    print("  forward pass: ")
    @time ReverseDiff.forward_pass!(tp)
    gc()
    print("  reverse pass: ")
    @time ReverseDiff.reverse_pass!(tp)
    gc()

    if length(tp.tape) <= 10000
        ctp = ReverseDiff.compile(tp)

        # warmup
        ReverseDiff.gradient!(out, ctp, x)
        ReverseDiff.forward_pass!(ctp)
        ReverseDiff.reverse_pass!(ctp)

        # actual
        print("  gradient! (compiled): ")
        @time ReverseDiff.gradient!(out, ctp, x)
        gc()
        print("  forward pass (compiled): ")
        @time ReverseDiff.forward_pass!(ctp)
        gc()
        print("  reverse pass (compiled): ")
        @time ReverseDiff.reverse_pass!(ctp)
        gc()
    else
        println("skipping compiled GradientTape benchmark because the tape is too long ($(length(tp.tape)) elements)")
    end
end

####################################################################

rosenbrock(x) = sum(map(ReverseDiff.@forward((i, j) -> (1 - j)^2 + 100*(i - j^2)^2), x[2:end], x[1:end-1]))

# function rosenbrock(x)
#     i = x[2:end]
#     j = x[1:end-1]
#     return sum((1 .- j).^2 + 100*(i - j.^2).^2)
# end
#
# function rosenbrock(x::AbstractVector)
#     a = one(eltype(x))
#     b = 100 * a
#     result = zero(eltype(x))
#     for i in 1:length(x)-1
#         result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2
#     end
#     return result
# end

x = rand(100000)
out = zeros(x)
grad_benchmark_driver!(out, rosenbrock, x)

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

x = rand(100000)
out = zeros(x)
grad_benchmark_driver!(out, ackley, x)

####################################################################

function generate_matrix_test(n)
    return x -> begin
        @assert length(x) == 2n^2 + n
        a = reshape(x[1:n^2], n, n)
        b = reshape(x[n^2 + 1:2n^2], n, n)
        return trace(log.((a * b) + a - b))
    end
end

n = 100
matrix_test = generate_matrix_test(n)
x = collect(1.0:(2n^2 + n))
out = zeros(x)
grad_benchmark_driver!(out, matrix_test, x)

####################################################################

relu(x) = log.(1.0 .+ exp.(x))

ReverseDiff.@forward sigmoid(n) = 1. / (1. + exp(-n))

function neural_net(w1, w2, w3, x1)
    x2 = relu(w1 * x1)
    x3 = relu(w2 * x2)
    return sigmoid(dot(w3, x3))
end

xs = (randn(10,10), randn(10,10), randn(10), rand(10))
outs = map(similar, xs)
grad_benchmark_driver!(outs, neural_net, xs)

####################################################################

println("done")
