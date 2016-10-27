module TrackedTests

using ReverseDiff, Base.Test

include("utils.jl")

println("testing Tracked type...")
tic()

############################################################################################

##########################
# Constructors/Accessors #
##########################

a, b, c = rand(3)
tp = Tape()
ntp = Nullable(tp)

@test value(a) === a
@test !(ReverseDiff.hastape(a))

at = Tracked(a)
@test valtype(at) === typeof(a)
@test adjtype(at) === typeof(a)
@test valtype(typeof(at)) === valtype(at)
@test adjtype(typeof(at)) === adjtype(at)
@test value(at) === at.value === a
@test adjoint(at) === at.adjoint === zero(a)
@test tape(at) === at.tape === Nullable{Tape}()
@test !(ReverseDiff.hastape(at))

bt = Tracked(b, Int)
@test valtype(bt) === typeof(b)
@test adjtype(bt) === Int
@test valtype(typeof(bt)) === valtype(bt)
@test adjtype(typeof(bt)) === adjtype(bt)
@test value(bt) === bt.value === b
@test adjoint(bt) === bt.adjoint === zero(Int)
@test tape(bt) === bt.tape === Nullable{Tape}()
@test !(ReverseDiff.hastape(bt))

ct = Tracked(c, Int, ntp)
@test valtype(ct) === typeof(c)
@test adjtype(ct) === Int
@test valtype(typeof(ct)) === valtype(ct)
@test adjtype(typeof(ct)) === adjtype(ct)
@test value(ct) === ct.value === c
@test adjoint(ct) === ct.adjoint === zero(Int)
@test get(tape(ct)) === get(ct.tape) === tp === get(ntp)
@test ReverseDiff.hastape(ct)

@test tape(at) === tape(bt) === Nullable{Tape}()
@test tape(at, bt) === tape(bt, at)
@test tape(ct) === tape(at, ct) === tape(bt, ct) === tape(ct, at) === tape(ct, bt)

@test isempty(tp)

########################
# Conversion/Promotion #
########################

tp = Tape()
ntp = Nullable(tp)
x = 1.0
t = Tracked(x, Int, ntp)

@test convert(typeof(x), Tracked(x)) === x
@test tracked_is(convert(Tracked{typeof(x),Int}, x), Tracked(x, Int))
@test tracked_is(convert(Tracked{Int,Int}, Tracked(x, ntp)), Tracked(Int(x), Int, ntp))

@test convert(typeof(t), t) === t

@test promote_type(Int64, Tracked{Int32,Int32}) === Tracked{Int64,Int32}
@test promote_type(Tracked{Int64,Int32}, Tracked{Int32,Int64}) === Tracked{Int64,Int64}

@test Base.promote_array_type(nothing, Tracked{Int,Int}, Float64) === Tracked{Float64,Int}
@test Base.promote_array_type(nothing, Tracked{Int,Int}, Float64, Int) === Int
@test Base.promote_array_type(nothing, Float64, Tracked{Int,Int}) === Tracked{Float64,Int}
@test Base.promote_array_type(nothing, Float64, Tracked{Int,Int}, Int) === Int

@test isempty(tp)

####################
# `Real` Interface #
####################

tp = Tape()
ntp = Nullable(tp)
x = rand()
t = Tracked(x, ntp)

@test hash(t) === hash(x)
@test hash(t, hash(x)) === hash(x, hash(x))

@test deepcopy(t) === t
@test copy(t) === t

@test float(t) === t
@test tracked_is(float(Tracked(1)), Tracked(1.0, Int))

@test tracked_is(one(t), Tracked(one(x)))
@test tracked_is(zero(t), Tracked(zero(x)))

@test begin
    srand(1)
    rand_t1 = rand(typeof(t))
    srand(1)
    rand_t2 = Tracked(rand(valtype(t)))
    tracked_is(rand_t1, rand_t2)
end

@test begin
    rng = MersenneTwister(1)
    rand_t1 = rand(rng, typeof(t))
    rng = MersenneTwister(1)
    rand_t2 = Tracked(rand(rng, valtype(t)))
    tracked_is(rand_t1, rand_t2)
end

@test eps(t) === eps(x)
@test eps(typeof(t)) === eps(typeof(x))

@test floor(t) === floor(x)
@test floor(Int, t) === floor(Int, x)

@test ceil(t) === ceil(x)
@test ceil(Int, t) === ceil(Int, x)

@test trunc(t) === trunc(x)
@test trunc(Int, t) === trunc(Int, x)

@test round(t) === round(x)
@test round(Int, t) === round(Int, x)

############################################################################################

println("done (took $(toq()) seconds)")

end # module
