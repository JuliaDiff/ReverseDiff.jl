module CompatTests

using FillArrays, ReverseDiff, Test

@test ReverseDiff.gradient(fill(2.0, 3)) do x
    sum(abs2.(x .- Zeros(3)))
end == fill(4.0, 3)

@test ReverseDiff.gradient(fill(2.0, 3)) do x
    sum(abs2.(x .- (1:3)))
end == [2, 0, -2]

end
