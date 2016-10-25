module TrackedTests

using ReverseDiffPrototype, Base.Test
using ReverseDiffPrototype: Tape, Tracked, value, adjoint, tape, valtype, adjtype

const RDP = ReverseDiffPrototype

include("utils.jl")

println("testing Tracked type...")
tic()

############################################################################################

tracked_is(a, b) = value(a) === value(b) && adjoint(a) === adjoint(b) && tape(a) === tape(b)

##########################
# Constructors/Accessors #
##########################

a, b, c = rand(3)
tp = Tape()
ntp = Nullable(tp)

@test value(a) === a
@test !(RDP.hastape(a))

ta = Tracked(a)
@test valtype(ta) === typeof(a)
@test adjtype(ta) === typeof(a)
@test valtype(typeof(ta)) === valtype(ta)
@test adjtype(typeof(ta)) === adjtype(ta)
@test value(ta) === ta.value === a
@test adjoint(ta) === ta.adjoint === zero(a)
@test tape(ta) === ta.tape === Nullable{Tape}()
@test !(RDP.hastape(ta))

tb = Tracked(b, Int)
@test valtype(tb) === typeof(b)
@test adjtype(tb) === Int
@test valtype(typeof(tb)) === valtype(tb)
@test adjtype(typeof(tb)) === adjtype(tb)
@test value(tb) === tb.value === b
@test adjoint(tb) === tb.adjoint === zero(Int)
@test tape(tb) === tb.tape === Nullable{Tape}()
@test !(RDP.hastape(tb))

tc = Tracked(c, Int, ntp)
@test valtype(tc) === typeof(c)
@test adjtype(tc) === Int
@test valtype(typeof(tc)) === valtype(tc)
@test adjtype(typeof(tc)) === adjtype(tc)
@test value(tc) === tc.value === c
@test adjoint(tc) === tc.adjoint === zero(Int)
@test get(tape(tc)) === get(tc.tape) === tp === get(ntp)
@test RDP.hastape(tc)

@test tape(ta) === tape(tb) === Nullable{Tape}()
@test tape(ta, tb) === tape(tb, ta)
@test tape(tc) === tape(ta, tc) === tape(tb, tc) === tape(tc, ta) === tape(tc, tb)

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
