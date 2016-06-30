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
g! = ReverseDiffPrototype.gradient(rosenbrock, Input{Float64,100000});
