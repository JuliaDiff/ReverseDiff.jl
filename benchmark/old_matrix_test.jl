using ReverseDiffPrototype

const N = 100
x = rand(2N^2 + N)
function matrix_test(x)
    k = length(x)
    A = reshape(x[1:N^2], N, N)
    B = reshape(x[N^2 + 1:2N^2], N, N)
    c = x[2N^2+1:end]
    return trace(log(A * B .+ c))
end

out = zeros(x);
t = ReverseDiffPrototype.trace_input_array(matrix_test, x, eltype(out));
@time ReverseDiffPrototype.gradient!(out, matrix_test, x, t);
