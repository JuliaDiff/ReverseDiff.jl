using ReverseDiff, Test

# https://github.com/JuliaDiff/ReverseDiff.jl/issues/135

struct Over{T} den::T end

(o::Over)(x) = x ./ o.den

(o::Over)(x::ReverseDiff.TrackedArray) = ReverseDiff.track(o, x)

ReverseDiff.@grad function (o::Over)(x)
    # abused gradient :/ but we can leverage it to test whether it come from custom grad or "normal over".
    ReverseDiff.value(x) ./ o.den, Δ -> (Δ .* o.den,) 
end


struct Over2 end

(o::Over2)(x) = x ./ 2

(o::Over2)(x::ReverseDiff.TrackedArray) = ReverseDiff.track(o, x)

ReverseDiff.@grad function (o::Over2)(x)
    ReverseDiff.value(x) ./ 2, Δ -> (Δ .* 2,)
end

o3 = Over(3.)
o2 = Over2()

g3 = ReverseDiff.gradient([2., 1., 3.]) do x
    sum(o3(x))
end

g2 = ReverseDiff.gradient([2., 1., 3.]) do x
    sum(o2(x))
end

@test g3 == [3., 3., 3.]
@test g2 == [2., 2., 2.]