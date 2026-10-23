include("elementwise_utils.jl")

for (M, fsym) in binary_diffrules()[1:2:end]
    test_binary(M, fsym)
end
