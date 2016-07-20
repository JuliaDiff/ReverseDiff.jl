using ReverseDiffPrototype

const N = 100
x = rand(2N^2 + N)
function matrixnode(x)
    k = length(x)
    A = reshape(x[1:N^2], N, N)
    B = reshape(x[N^2 + 1:2N^2], N, N)
    c = x[2N^2+1:end]
    return trace(log(A * B .+ c))
end

out = zeros(x);
t = ReverseDiffPrototype.tape_array(matrixnode, x)
g! = ReverseDiffPrototype.gradient(matrixnode)
@time g!(out, x, t)
