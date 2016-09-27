const EPS = 1e-6

# make RNG deterministic, and thus make result inaccuracies
# deterministic so we don't have to retune EPS for arbitrary inputs
srand(1)

testprintln(kind, f, pad = "  ") = println(pad, "testing $(kind): `$(f)`...")

# function test4(x)
#     k = length(x)
#     N = isqrt(k)
#     A = reshape(x, N, N)
#     return sum(map(RDP.@forward(n -> sqrt(abs(n) + n^2) * 0.5), A))
# end
#
# rosenbrock3(x) = sum(map(RDP.@forward((i, j) -> (1 - j)^2 + 100*(i - j^2)^2), x[2:end], x[1:end-1]))
#
# relu(x) = log.(1.0 .+ exp(x))
# RDP.@forward sigmoid(n) = 1. / (1. + exp(-n))
# neural_step(x1, w1, w2) = sigmoid(dot(w2, relu(w1 * x1)))
#
