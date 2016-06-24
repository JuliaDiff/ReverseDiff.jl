using ReverseDiffPrototype
using Base.Test
using ForwardDiff

x = collect(1.0:5.0)
out = zeros(x)
testf(x) = (exp(x[1]) + log(x[3]) * x[4]) / x[5]

@test ReverseDiffPrototype.gradient!(out, testf, x) == ForwardDiff.gradient(testf, x)

x = collect(1.0:2.0)
out = zeros(x)
testf2(x) = x[1]*x[2] + sin(x[1])

@test ReverseDiffPrototype.gradient!(out, testf2, x) == ForwardDiff.gradient(testf2, x)
