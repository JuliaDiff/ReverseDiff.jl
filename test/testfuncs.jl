testprintln(kind, f, pad = "  ") = println(pad, "testing $(kind): `$(f)`...")

#############################
# Array -> Number functions #
#############################

test1(x) = (exp(x[1]) + log(x[3]) * x[4]) / x[5]
test2(x) = x[1]*x[2] + sin(x[1])

function test3(x)
    a = reshape(x, length(x), 1)
    b = reshape(copy(x), 1, length(x))
    return trace(log((1 .+ (a * b)) .+ a .- b))
end

function test4(x)
    k = length(x)
    N = isqrt(k)
    A = reshape(x, N, N)
    return sum(map(RDP.@forward(n -> sqrt(abs(n) + n^2) * 0.5), A))
end

test5(x) = norm(x' .* x, 1)

function rosenbrock1(x)
    a = one(eltype(x))
    b = 100 * a
    result = zero(eltype(x))
    for i in 1:length(x)-1
        result += (a - x[i])^2 + b*(x[i+1] - x[i]^2)^2
    end
    return result
end

function rosenbrock2(x)
    a = x[1]
    b = 100 * a
    v = map((i, j) -> (a - j)^2 + b*(i - j^2)^2, x[2:end], x[1:end-1])
    return sum(v)
end

rosenbrock3(x) = sum(map(RDP.@forward((i, j) -> (1 - j)^2 + 100*(i - j^2)^2), x[2:end], x[1:end-1]))

rosenbrock4(x) = sum((1 - x[1:end-1]).^2 + 100*(x[2:end] - x[1:end-1].^2).^2)

function ackley(x)
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

self_weighted_logit(x) = inv(1.0 + exp(-dot(vec(x), vec(x))))

const UNARY_ARR2NUM_FUNCS = (test1, test2, test3, test4, test5,
                             rosenbrock1, rosenbrock2, rosenbrock3, rosenbrock4,
                             ackley, det, self_weighted_logit)

######################################
# (Array, Array) -> Number functions #
######################################

relu(x) = log.(1.0 .+ exp(x))
RDP.@forward sigmoid(n) = 1. / (1. + exp(-n))
neural_step(x1, w1, w2) = sigmoid(dot(w2, relu(w1 * x1)))

const TERNARY_ARR2NUM_FUNCS = (neural_step,)

############################
# Array -> Array functions #
############################

const UNARY_ARR2ARR_FUNCS = (-, inv)

#####################################
# (Array, Array) -> Array functions #
#####################################

const BINARY_ARR2ARR_FUNCS = (+, .+, -, .-, *, .*, ./, .^,
                              A_mul_Bt, At_mul_B, At_mul_Bt,
                              A_mul_Bc, Ac_mul_B, Ac_mul_Bc)
