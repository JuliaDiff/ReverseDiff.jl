module TrackedTests

using ReverseDiff, Base.Test
using ReverseDiff: TrackedReal, TrackedArray

include(joinpath(dirname(@__FILE__), "utils.jl"))

samefields(a, b) = a === b

function samefields(a::TrackedReal, b::TrackedReal)
    same_origin = false
    if isdefined(a, :origin) === isdefined(b, :origin)
        if isdefined(a, :origin)
            same_origin = a.origin === b.origin
        else
            same_origin = true
        end
    end
    return (ReverseDiff.value(a) == ReverseDiff.value(b) &&
            ReverseDiff.deriv(a) == ReverseDiff.deriv(b) &&
            ReverseDiff.tape(a) === ReverseDiff.tape(b) &&
            a.index === b.index && same_origin)
end

function samefields(a::TrackedArray, b::TrackedArray)
    return (ReverseDiff.value(a) == ReverseDiff.value(b) &&
            ReverseDiff.deriv(a) == ReverseDiff.deriv(b) &&
            ReverseDiff.tape(a) === ReverseDiff.tape(b))
end

println("testing TrackedReals/TrackedArrays...")
tic()

############################################################################################

################
# Constructors #
################

# TrackedReal #
#-------------#

v = rand()
d = rand(Int)
o = [v]
tp = RawTape()
t = TrackedReal(v, d, tp, 1, o)

@test t.value === v
@test t.deriv === d
@test t.tape === tp
@test t.index === 1
@test t.origin === o
@test isa(t, TrackedReal{Float64,Int,Vector{Float64}})

t = TrackedReal(v, d, tp)

@test t.value === v
@test t.deriv === d
@test t.tape === tp
@test t.index === ReverseDiff.NULL_INDEX
@test t.origin === nothing
@test isa(t, TrackedReal{Float64,Int,Void})

t = TrackedReal(v, d)

@test t.value === v
@test t.deriv === d
@test t.tape === ReverseDiff.NULL_TAPE
@test t.index === ReverseDiff.NULL_INDEX
@test t.origin === nothing
@test isa(t, TrackedReal{Float64,Int,Void})

# TrackedArray #
#--------------#

v = rand(3, 2, 1)
d = rand(Int, 3, 2, 1)
tp = RawTape()

t = TrackedArray(v, d, tp)

@test t.value === v
@test t.deriv === d
@test t.tape === tp
@test isa(t, TrackedArray{Float64,Int,3,Array{Float64,3},Array{Int,3}})

###########
# getters #
###########

# istracked #
#-----------#

@test !(ReverseDiff.istracked(nothing))
@test !(ReverseDiff.istracked(rand()))
@test !(ReverseDiff.istracked([1]))
@test ReverseDiff.istracked(TrackedArray(rand(1), rand(1), RawTape()))
@test ReverseDiff.istracked(TrackedReal(1, 1))
@test ReverseDiff.istracked(Any[1])
@test ReverseDiff.istracked([TrackedReal(1, 1)])

# value #
#-------#

v, d = rand(), rand()
varr, darr = [v], [d]
any_varr = Any[v]

@test ReverseDiff.value(nothing) === nothing
@test ReverseDiff.value(v) === v
@test ReverseDiff.value(TrackedArray(varr, darr, RawTape())) === varr
@test ReverseDiff.value(TrackedReal(v, d)) === v
@test ReverseDiff.value(varr) === varr
@test ReverseDiff.value(any_varr) !== any_varr
@test ReverseDiff.value(any_varr) == any_varr
@test ReverseDiff.value([TrackedReal(v, d)]) == varr

# deriv #
#-------#

v, d = rand(), rand()
varr, darr = [v], [d]
any_varr = Any[v]

@test ReverseDiff.deriv(TrackedArray(varr, darr, RawTape())) === darr
@test ReverseDiff.deriv(TrackedReal(v, d)) === d

# valtype #
#---------#

@test ReverseDiff.valtype(TrackedReal(1,0)) === Int
@test ReverseDiff.valtype(TrackedReal(1.0, 0)) === Float64
@test ReverseDiff.valtype(typeof(TrackedReal(1,0))) === Int
@test ReverseDiff.valtype(typeof(TrackedReal(1.0, 0))) === Float64
@test ReverseDiff.valtype(TrackedArray([1], [0], RawTape())) === Int
@test ReverseDiff.valtype(TrackedArray([1.0], [0], RawTape())) === Float64
@test ReverseDiff.valtype(typeof(TrackedArray([1], [0], RawTape()))) === Int
@test ReverseDiff.valtype(typeof(TrackedArray([1.0], [0], RawTape()))) === Float64

# derivtype #
#-----------#

@test ReverseDiff.derivtype(TrackedReal(1,0)) === Int
@test ReverseDiff.derivtype(TrackedReal(1, 0.0)) === Float64
@test ReverseDiff.derivtype(typeof(TrackedReal(1,0))) === Int
@test ReverseDiff.derivtype(typeof(TrackedReal(1, 0.0))) === Float64
@test ReverseDiff.derivtype(TrackedArray([1], [0], RawTape())) === Int
@test ReverseDiff.derivtype(TrackedArray([1], [0.0], RawTape())) === Float64
@test ReverseDiff.derivtype(typeof(TrackedArray([1], [0], RawTape()))) === Int
@test ReverseDiff.derivtype(typeof(TrackedArray([1], [0.0], RawTape()))) === Float64

# hasorigin #
#-----------#

@test !(ReverseDiff.hasorigin(rand()))
@test !(ReverseDiff.hasorigin(TrackedReal(rand(), rand())))
@test ReverseDiff.hasorigin(TrackedReal(rand(), rand(), RawTape(), 1, [rand()]))

# tape/hastape #
#--------------#

tp = RawTape()

null_tape_items = (nothing, rand(), [1], Any[1], [TrackedReal(1, 1)], TrackedReal(1, 1),
                   TrackedArray(rand(1), rand(1), ReverseDiff.NULL_TAPE))
tape_items = ([3, TrackedReal(1, 1, tp)], TrackedReal(1, 1, tp),
              TrackedArray(rand(1), rand(1), tp))

for i in null_tape_items
    @test ReverseDiff.tape(i) === ReverseDiff.NULL_TAPE
    @test !(ReverseDiff.hastape(i))
end

for i in tape_items
    @test ReverseDiff.tape(i) === tp
    @test ReverseDiff.hastape(i)
    for j in null_tape_items
        @test ReverseDiff.tape(i, j) === tp
        @test ReverseDiff.tape(j, i) === tp
        @test ReverseDiff.tape(i, j, i) === tp
        @test ReverseDiff.tape(j, i, i) === tp
        @test ReverseDiff.tape(i, i, j) === tp
    end
    for j in tape_items
        @test ReverseDiff.tape(i, j) === tp
        @test ReverseDiff.tape(i, j, i) === tp
    end
end

###########
# setters #
###########

v, d = rand(), rand()
v2, d2 = rand(), rand()
varr, darr = rand(3), rand(3)
varr2, darr2 = rand(3), rand(3)

tr = TrackedReal(v, d)
ta = TrackedArray(varr, darr, RawTape())

# value! #
#--------#

ReverseDiff.value!(tr, v2)
@test ReverseDiff.value(tr) === v2

ReverseDiff.value!(ta, varr2)
@test ReverseDiff.value(ta) == varr2
@test ReverseDiff.value(ta) === varr

ReverseDiff.value!(tr, v2)
@test ReverseDiff.value(tr) === v2

varr_copy = copy(varr)
ReverseDiff.value!((ta, tr), (varr, v))
@test ReverseDiff.value(tr) === v
@test ReverseDiff.value(ta) == varr_copy
@test ReverseDiff.value(ta) === varr

# deriv! #
#--------#

ReverseDiff.deriv!(tr, d2)
@test ReverseDiff.deriv(tr) === d2

ReverseDiff.deriv!(ta, darr2)
@test ReverseDiff.deriv(ta) == darr2
@test ReverseDiff.deriv(ta) === darr

ReverseDiff.deriv!(tr, d2)
@test ReverseDiff.deriv(tr) === d2

darr_copy = copy(darr)
ReverseDiff.deriv!((ta, tr), (darr, d))
@test ReverseDiff.deriv(tr) === d
@test ReverseDiff.deriv(ta) == darr_copy
@test ReverseDiff.deriv(ta) === darr

# pulling values from origin #
#----------------------------#

ta = TrackedArray(rand(3), rand(3), RawTape())
ta1 = ta[1]
v_old = ReverseDiff.value(ta)[1]
v_new = rand()
tr = TrackedReal(v_old, v_old)
varr_copy = copy(ReverseDiff.value(ta))
trs = Any[tr, ta[1], ta[2]]

@test ReverseDiff.pull_value!(nothing) === nothing
@test ReverseDiff.pull_value!([1,2,3]) === nothing
@test ReverseDiff.pull_value!(1.0) === nothing

ReverseDiff.value(ta)[1] = v_new
@test ReverseDiff.value(ta1) === v_old
ReverseDiff.pull_value!(ta1)
@test ReverseDiff.value(ta1) === v_new
ReverseDiff.value(ta)[1] = v_old
ReverseDiff.pull_value!(ta1)

@test ReverseDiff.value(tr) === v_old
ReverseDiff.pull_value!(tr)
@test ReverseDiff.value(tr) === v_old

ReverseDiff.pull_value!(ta)
@test ReverseDiff.value(ta) == varr_copy

ReverseDiff.value(ta)[1] = v_new + 1
ReverseDiff.value(ta)[2] = v_new + 2

@test ReverseDiff.value(trs[1]) === ReverseDiff.value(tr)
@test ReverseDiff.value(trs[2]) === varr_copy[1]
@test ReverseDiff.value(trs[3]) === varr_copy[2]

ReverseDiff.pull_value!(trs)

@test ReverseDiff.value(trs[1]) === ReverseDiff.value(tr)
@test ReverseDiff.value(trs[2]) === ReverseDiff.value(ta)[1]
@test ReverseDiff.value(trs[3]) === ReverseDiff.value(ta)[2]

# pulling derivs from origin #
#----------------------------#

ta = TrackedArray(rand(3), rand(3), RawTape())
ta1 = ta[1]
d_old = ReverseDiff.deriv(ta)[1]
d_new = rand()
tr = TrackedReal(d_old, d_old)
darr_copy = copy(ReverseDiff.deriv(ta))
trs = Any[tr, ta[1], ta[2]]

@test ReverseDiff.pull_deriv!(nothing) === nothing
@test ReverseDiff.pull_deriv!([1,2,3]) === nothing
@test ReverseDiff.pull_deriv!(1.0) === nothing

ReverseDiff.deriv(ta)[1] = d_new
@test ReverseDiff.deriv(ta1) === d_old
ReverseDiff.pull_deriv!(ta1)
@test ReverseDiff.deriv(ta1) === d_new
ReverseDiff.deriv(ta)[1] = d_old
ReverseDiff.pull_deriv!(ta1)

@test ReverseDiff.deriv(tr) === d_old
ReverseDiff.pull_deriv!(tr)
@test ReverseDiff.deriv(tr) === d_old

ReverseDiff.pull_deriv!(ta)
@test ReverseDiff.deriv(ta) == darr_copy

ReverseDiff.deriv(ta)[1] = d_new + 1
ReverseDiff.deriv(ta)[2] = d_new + 2

@test ReverseDiff.deriv(trs[1]) === ReverseDiff.deriv(tr)
@test ReverseDiff.deriv(trs[2]) === darr_copy[1]
@test ReverseDiff.deriv(trs[3]) === darr_copy[2]

ReverseDiff.pull_deriv!(trs)

@test ReverseDiff.deriv(trs[1]) === ReverseDiff.deriv(tr)
@test ReverseDiff.deriv(trs[2]) === ReverseDiff.deriv(ta)[1]
@test ReverseDiff.deriv(trs[3]) === ReverseDiff.deriv(ta)[2]

# push derivs from origin #
#-------------------------#

ta = TrackedArray(rand(3), rand(3), RawTape())
ta1 = ta[1]
d_old = ReverseDiff.deriv(ta)[1]
d_new = rand()
tr = TrackedReal(d_old, d_old)
darr_copy = copy(ReverseDiff.deriv(ta))
trs = Any[tr, ta[1], ta[2]]

@test ReverseDiff.push_deriv!(nothing) === nothing
@test ReverseDiff.push_deriv!([1,2,3]) === nothing
@test ReverseDiff.push_deriv!(1.0) === nothing

ReverseDiff.deriv!(ta1, d_new)
@test ReverseDiff.deriv(ta)[1] === d_old
ReverseDiff.push_deriv!(ta1)
@test ReverseDiff.deriv(ta)[1] === d_new
ReverseDiff.deriv(ta)[1] = d_old
ReverseDiff.pull_deriv!(ta1)

@test ReverseDiff.deriv(tr) === d_old
ReverseDiff.push_deriv!(tr)
@test ReverseDiff.deriv(tr) === d_old

ReverseDiff.push_deriv!(ta)
@test ReverseDiff.deriv(ta) == darr_copy
@test ReverseDiff.deriv(ta1) === d_old

ReverseDiff.deriv!(trs[2], d_new + 1)
ReverseDiff.deriv!(trs[3], d_new + 2)

@test ReverseDiff.deriv(ta)[1] === darr_copy[1]
@test ReverseDiff.deriv(ta)[2] === darr_copy[2]

ReverseDiff.push_deriv!(trs)

@test ReverseDiff.deriv(ta)[1] === ReverseDiff.deriv(trs[2])
@test ReverseDiff.deriv(ta)[2] === ReverseDiff.deriv(trs[3])

# seed!/unseed! #
#---------------#

v, d = rand(), rand()
varr, darr = rand(3), rand(3)
varr_copy, darr_copy = copy(varr), copy(darr)
ta = TrackedArray(varr, darr, RawTape())
tr = TrackedReal(v, d)
trs = Any[tr, ta[1], ta[2]]

@test ReverseDiff.seed!(nothing) === nothing
@test ReverseDiff.seed!([1,2,3]) === nothing
@test ReverseDiff.seed!(1.0) === nothing
@test ReverseDiff.unseed!(nothing) === nothing
@test ReverseDiff.unseed!([1,2,3]) === nothing
@test ReverseDiff.unseed!(1.0) === nothing

ReverseDiff.seed!(ta)
@test ReverseDiff.value(ta) === varr == varr_copy
@test ReverseDiff.deriv(ta) === darr == darr_copy
ReverseDiff.unseed!(ta)
@test ReverseDiff.value(ta) === varr == varr_copy
@test ReverseDiff.deriv(ta) === darr == zeros(darr)
ReverseDiff.deriv!(ta, darr_copy)
@test ReverseDiff.value(ta) === varr == varr_copy
@test ReverseDiff.deriv(ta) === darr == darr_copy

ReverseDiff.seed!(tr)
@test ReverseDiff.value(tr) === v
@test ReverseDiff.deriv(tr) === 1.0
ReverseDiff.unseed!(tr)
@test ReverseDiff.value(tr) === v
@test ReverseDiff.deriv(tr) === 0.0
ReverseDiff.deriv!(tr, d)
@test ReverseDiff.value(tr) === v
@test ReverseDiff.deriv(tr) === d

ReverseDiff.seed!(trs)
@test ReverseDiff.value(tr) === v
@test ReverseDiff.deriv(tr) === d
@test ReverseDiff.value(ta) === varr == varr_copy
@test ReverseDiff.deriv(ta) === darr == darr_copy
@test ReverseDiff.value(trs[2]) === varr[1]
@test ReverseDiff.deriv(trs[2]) === darr[1]
@test ReverseDiff.value(trs[3]) === varr[2]
@test ReverseDiff.deriv(trs[3]) === darr[2]

ReverseDiff.unseed!(trs)
@test ReverseDiff.value(tr) === v
@test ReverseDiff.deriv(tr) === 0.0
@test ReverseDiff.value(ta) === varr == varr_copy
@test ReverseDiff.deriv(ta) === darr == [0.0, 0.0, darr_copy[3]]
@test ReverseDiff.value(trs[2]) === varr[1]
@test ReverseDiff.deriv(trs[2]) === 0.0
@test ReverseDiff.value(trs[3]) === varr[2]
@test ReverseDiff.deriv(trs[3]) === 0.0

ReverseDiff.deriv!(ta, darr_copy)
ReverseDiff.deriv!(tr, d)
ReverseDiff.pull_deriv!(trs)
@test trs[1] === tr
@test ReverseDiff.deriv(tr) === d
@test ReverseDiff.deriv(ta) === darr == darr_copy
@test ReverseDiff.deriv(trs[2]) === darr[1]
@test ReverseDiff.deriv(trs[3]) === darr[2]

ReverseDiff.seed!((tr, ta, trs))
@test ReverseDiff.value(tr) === v
@test ReverseDiff.deriv(tr) === d
@test ReverseDiff.value(ta) === varr == varr_copy
@test ReverseDiff.deriv(ta) === darr == darr_copy
@test ReverseDiff.value(trs[2]) === varr[1]
@test ReverseDiff.deriv(trs[2]) === darr[1]
@test ReverseDiff.value(trs[3]) === varr[2]
@test ReverseDiff.deriv(trs[3]) === darr[2]

ReverseDiff.unseed!((tr, ta, trs))
@test ReverseDiff.value(tr) === v
@test ReverseDiff.deriv(tr) === 0.0
@test ReverseDiff.value(ta) === varr == varr_copy
@test ReverseDiff.deriv(ta) === darr == zeros(darr)
@test ReverseDiff.value(trs[2]) === varr[1]
@test ReverseDiff.deriv(trs[2]) === 0.0
@test ReverseDiff.value(trs[3]) === varr[2]
@test ReverseDiff.deriv(trs[3]) === 0.0

# increment deriv #
#-----------------#

x = rand(3)
v, d = rand(), rand()
varr, darr = rand(3), rand(3)
varr_copy, darr_copy = copy(varr), copy(darr)
ta = TrackedArray(varr, darr, RawTape())
tr = TrackedReal(v, d)
trs = Any[tr, ta[1], ta[2]]

ReverseDiff.increment_deriv!(tr, x[1])
@test ReverseDiff.value(tr) === v
@test ReverseDiff.deriv(tr) === d + x[1]
ReverseDiff.deriv!(tr, d)

ta1 = ta[1]
ReverseDiff.deriv!(ta, x)
ReverseDiff.increment_deriv!(ta1, d)
@test ReverseDiff.value(ta1) === ReverseDiff.value(ta)[1] === varr_copy[1]
@test ReverseDiff.deriv(ta1) === ReverseDiff.deriv(ta)[1] === x[1] + d
ReverseDiff.deriv!(ta, darr_copy)

ReverseDiff.increment_deriv!(ta, x)
@test ReverseDiff.value(ta) === varr == varr_copy
@test ReverseDiff.deriv(ta) === darr == darr_copy + x
ReverseDiff.deriv!(ta, darr_copy)

ReverseDiff.increment_deriv!(ta, x[1])
@test ReverseDiff.value(ta) === varr == varr_copy
@test ReverseDiff.deriv(ta) === darr == darr_copy .+ x[1]
ReverseDiff.deriv!(ta, darr_copy)

ReverseDiff.increment_deriv!(trs, x)
@test ReverseDiff.value(tr) === v
@test ReverseDiff.deriv(tr) === d + x[1]
@test ReverseDiff.value(trs[2]) === varr[1] === varr_copy[1]
@test ReverseDiff.deriv(trs[2]) === darr[1] === darr_copy[1] + x[2]
@test ReverseDiff.value(trs[3]) === varr[2] === varr_copy[2]
@test ReverseDiff.deriv(trs[3]) === darr[2] === darr_copy[2] + x[3]
ReverseDiff.deriv!(tr, d)
ReverseDiff.deriv!(ta, darr_copy)
ReverseDiff.pull_deriv!(trs)

ReverseDiff.increment_deriv!(trs, x[1])
@test ReverseDiff.value(tr) === v
@test ReverseDiff.deriv(tr) === d + x[1]
@test ReverseDiff.value(trs[2]) === varr[1] === varr_copy[1]
@test ReverseDiff.deriv(trs[2]) === darr[1] === darr_copy[1] + x[1]
@test ReverseDiff.value(trs[3]) === varr[2] === varr_copy[2]
@test ReverseDiff.deriv(trs[3]) === darr[2] === darr_copy[2] + x[1]
ReverseDiff.deriv!(tr, d)
ReverseDiff.deriv!(ta, darr_copy)
ReverseDiff.pull_deriv!(trs)

# decrement deriv #
#-----------------#

x = rand(3)
v, d = rand(), rand()
varr, darr = rand(3), rand(3)
varr_copy, darr_copy = copy(varr), copy(darr)
ta = TrackedArray(varr, darr, RawTape())
tr = TrackedReal(v, d)
trs = Any[tr, ta[1], ta[2]]

ReverseDiff.decrement_deriv!(tr, x[1])
@test ReverseDiff.value(tr) === v
@test ReverseDiff.deriv(tr) === d - x[1]
ReverseDiff.deriv!(tr, d)

ta1 = ta[1]
ReverseDiff.deriv!(ta, x)
ReverseDiff.decrement_deriv!(ta1, d)
@test ReverseDiff.value(ta1) === ReverseDiff.value(ta)[1] === varr_copy[1]
@test ReverseDiff.deriv(ta1) === ReverseDiff.deriv(ta)[1] === x[1] - d
ReverseDiff.deriv!(ta, darr_copy)

ReverseDiff.decrement_deriv!(ta, x)
@test ReverseDiff.value(ta) === varr == varr_copy
@test ReverseDiff.deriv(ta) === darr == darr_copy - x
ReverseDiff.deriv!(ta, darr_copy)

ReverseDiff.decrement_deriv!(ta, x[1])
@test ReverseDiff.value(ta) === varr == varr_copy
@test ReverseDiff.deriv(ta) === darr == darr_copy .- x[1]
ReverseDiff.deriv!(ta, darr_copy)

ReverseDiff.decrement_deriv!(trs, x)
@test ReverseDiff.value(tr) === v
@test ReverseDiff.deriv(tr) === d - x[1]
@test ReverseDiff.value(trs[2]) === varr[1] === varr_copy[1]
@test ReverseDiff.deriv(trs[2]) === darr[1] === darr_copy[1] - x[2]
@test ReverseDiff.value(trs[3]) === varr[2] === varr_copy[2]
@test ReverseDiff.deriv(trs[3]) === darr[2] === darr_copy[2] - x[3]
ReverseDiff.deriv!(tr, d)
ReverseDiff.deriv!(ta, darr_copy)
ReverseDiff.pull_deriv!(trs)

ReverseDiff.decrement_deriv!(trs, x[1])
@test ReverseDiff.value(tr) === v
@test ReverseDiff.deriv(tr) === d - x[1]
@test ReverseDiff.value(trs[2]) === varr[1] === varr_copy[1]
@test ReverseDiff.deriv(trs[2]) === darr[1] === darr_copy[1] - x[1]
@test ReverseDiff.value(trs[3]) === varr[2] === varr_copy[2]
@test ReverseDiff.deriv(trs[3]) === darr[2] === darr_copy[2] - x[1]
ReverseDiff.deriv!(tr, d)
ReverseDiff.deriv!(ta, darr_copy)
ReverseDiff.pull_deriv!(trs)

###########
# capture #
###########

v, d = rand(), rand()
varr, darr = rand(3), rand(3)
ta = TrackedArray(varr, darr, RawTape())
tr = TrackedReal(v, d)
trs = Any[tr, ta[1], ta[2]]

@test ReverseDiff.capture(ta) === ta
@test ReverseDiff.capture(tr) === v
@test samefields(ReverseDiff.capture(ta[1]), ta[1])
@test all(map(samefields, ReverseDiff.capture(trs), Any[v, ta[1], ta[2]]))

########################
# Conversion/Promotion #
########################

v, d = rand(), rand()
varr, darr = rand(3), rand(3)
tp = RawTape()
ta = TrackedArray(varr, darr, tp)
tr = TrackedReal(v, d)
ta1 = ta[1]
A = TrackedArray{BigInt,Float64,1,Array{BigInt,1},Array{Float64,1}}
T = TrackedReal{BigInt,Float64,A}

t2 = convert(TrackedReal{BigFloat,BigFloat,Void}, ta1)
@test length(tp) == 1
instr = tp[1]
@test instr.func === convert
@test instr.input === ta1
@test samefields(instr.output, TrackedReal(big(varr[1]), big(darr[1]), tp))
@test instr.cache === nothing
empty!(tp)

@test convert(Float64, tr) === v
@test convert(BigFloat, tr) == v

@test samefields(convert(T, 1), T(big(1), 0.0))
@test samefields(convert(TrackedReal{BigInt,Float64,Void}, 1), TrackedReal(big(1), 0.0))

@test convert(typeof(tr), tr) === tr
@test convert(typeof(ta), ta) === ta
@test convert(typeof(ta1), ta1) === ta1

@test promote_type(T, BigFloat) === TrackedReal{BigFloat,Float64,A}
@test promote_type(T, Float64) === TrackedReal{BigFloat,Float64,A}
@test promote_type(T, TrackedReal{BigFloat,BigFloat,Void}) === TrackedReal{BigFloat,BigFloat,Void}
@test promote_type(T, T) === T

@test Base.promote_array_type(1, T, BigFloat) === TrackedReal{BigFloat,Float64,A}
@test Base.promote_array_type(1, T, BigFloat, Int) === Int
@test Base.promote_array_type(1, BigFloat, T) === TrackedReal{BigFloat,Float64,A}
@test Base.promote_array_type(1, BigFloat, T, Int) === Int

@test Base.r_promote(+, tr) === tr
@test Base.r_promote(*, tr) === tr

###########################
# AbstractArray Interface #
###########################

varr, darr = rand(3, 3), rand(3, 3)
tp = RawTape()
ta = TrackedArray(varr, darr, tp)

@test isa(similar(ta), Matrix{eltype(ta)})

@test samefields(ta[2], TrackedReal(varr[2], darr[2], tp, 2, ta))

ta_sub = ta[:,:]
idx = ReverseDiff.index_iterable(indices(ta), (:, :))
@test collect(idx) == [(i, j) for i in 1:3, j in 1:3]
@test samefields(ta_sub, ta)
@test length(tp) == 1
instr = tp[1]
@test instr.func === getindex
@test instr.input === (ta, idx)
@test samefields(instr.output, ta)
@test instr.cache === nothing
empty!(tp)

ta_sub = ta[:,1:2]
idx = ReverseDiff.index_iterable(indices(ta), (:, 1:2))
@test collect(idx) == [(i, j) for i in 1:3, j in 1:2]
@test samefields(ta_sub, TrackedArray(varr[:,1:2], darr[:,1:2], tp))
@test length(tp) == 1
instr = tp[1]
@test instr.func === getindex
@test instr.input === (ta, idx)
@test samefields(instr.output, TrackedArray(varr[:,1:2], darr[:,1:2], tp))
@test instr.cache === nothing
empty!(tp)

ta_sub = ta[2:3,:]
idx = ReverseDiff.index_iterable(indices(ta), (2:3, :))
@test collect(idx) == [(i, j) for i in 2:3, j in 1:3]
@test samefields(ta_sub, TrackedArray(varr[2:3,:], darr[2:3,:], tp))
@test length(tp) == 1
instr = tp[1]
@test instr.func === getindex
@test instr.input === (ta, idx)
@test samefields(instr.output, TrackedArray(varr[2:3,:], darr[2:3,:], tp))
@test instr.cache === nothing
empty!(tp)

ta_sub = ta[1:2,2:3]
idx = ReverseDiff.index_iterable(indices(ta), (1:2, 2:3))
@test collect(idx) == [(i, j) for i in 1:2, j in 2:3]
@test samefields(ta_sub, TrackedArray(varr[1:2,2:3], darr[1:2,2:3], tp))
@test length(tp) == 1
instr = tp[1]
@test instr.func === getindex
@test instr.input === (ta, idx)
@test samefields(instr.output, TrackedArray(varr[1:2,2:3], darr[1:2,2:3], tp))
@test instr.cache === nothing
empty!(tp)

ta_sub = ta[2:6]
idx = ReverseDiff.index_iterable(indices(ta), (2:6,))
@test collect(idx) == [(i,) for i in 2:6]
@test samefields(ta_sub, TrackedArray(varr[2:6], darr[2:6], tp))
@test length(tp) == 1
instr = tp[1]
@test instr.func === getindex
@test instr.input === (ta, idx)
@test samefields(instr.output, TrackedArray(varr[2:6], darr[2:6], tp))
@test instr.cache === nothing
empty!(tp)

ta_sub = ta[:]
idx = ReverseDiff.index_iterable(indices(ta), (:,))
@test collect(idx) == [(i, j) for i in 1:3, j in 1:3]
@test samefields(ta_sub, TrackedArray(varr[:], darr[:], tp))
@test length(tp) == 1
instr = tp[1]
@test instr.func === getindex
@test instr.input === (ta, idx)
@test samefields(instr.output, TrackedArray(varr[:], darr[:], tp))
@test instr.cache === nothing
empty!(tp)

@test Base.linearindexing(ta) === Base.LinearFast()

@test Base.size(ta) === size(varr)

@test Base.copy(ta) === ta

@test ones(ta) == ones(TrackedReal{Float64,Float64,Void}, size(ta))

@test zeros(ta) == zeros(TrackedReal{Float64,Float64,Void}, size(ta))

####################
# `Real` Interface #
####################

v_int, v_float, d = rand(Int), rand(), rand()
tp = RawTape()
tr_int = TrackedReal(v_int, d, tp)
tr_float = TrackedReal(v_float, d, tp)

@test hash(tr_float) === hash(v_float)
@test hash(tr_float, hash(1)) === hash(v_float, hash(1))

@test deepcopy(tr_float) === tr_float
@test copy(tr_float) === tr_float

@test samefields(float(tr_int), TrackedReal{Float64,Float64,Void}(float(v_int)))
@test float(tr_float) === tr_float

@test samefields(one(tr_float), typeof(tr_float)(one(v_float)))

@test samefields(zero(tr_float), typeof(tr_float)(zero(v_float)))

tr_rand = rand(TrackedReal{Int,Float64,Void})
@test samefields(tr_rand, TrackedReal{Int,Float64,Void}(ReverseDiff.value(tr_rand)))

tr_rand = rand(MersenneTwister(1), TrackedReal{Int,Float64,Void})
@test samefields(tr_rand, TrackedReal{Int,Float64,Void}(ReverseDiff.value(tr_rand)))

@test eps(tr_float) === eps(v_float)
@test eps(typeof(tr_float)) === eps(Float64)

@test floor(tr_float) === floor(v_float)
@test floor(Int, tr_float) === floor(Int, v_float)

@test ceil(tr_float) === ceil(v_float)
@test ceil(Int, tr_float) === ceil(Int, v_float)

@test trunc(tr_float) === trunc(v_float)
@test trunc(Int, tr_float) === trunc(Int, v_float)

@test round(tr_float) === round(v_float)
@test round(Int, tr_float) === round(Int, v_float)

################
# track/track! #
################

v, d = rand(), rand()
varr, darr = rand(3), rand(3)
tp = RawTape()

@test samefields(ReverseDiff.track(v, tp), TrackedReal(v, zero(v), tp))
@test samefields(ReverseDiff.track(v, Int, tp), TrackedReal(v, zero(Int), tp))

@test samefields(ReverseDiff.track(varr, tp), TrackedArray(varr, zeros(varr), tp))
@test samefields(ReverseDiff.track(varr, Int, tp), TrackedArray(varr, zeros(Int, size(varr)), tp))

tr = TrackedReal(v, d, tp)
x = rand()
ReverseDiff.track!(tr, x)

@test samefields(tr, TrackedReal(x, zero(x), tp))

ta = TrackedArray(varr, darr, tp)
x = rand(3)
ReverseDiff.track!(ta, x)

@test samefields(ta, TrackedArray(x, zeros(x), tp))

ta = TrackedArray(varr, darr, tp)
trs = similar(ta)
tp2 = RawTape()
@test isa(trs, Vector{eltype(ta)})
trs = similar(ta, TrackedReal{Float64,Float64,Void})
track!(trs, ta, tp2)
for i in eachindex(trs)
    @test samefields(trs[i], track(varr[i], tp2))
end

############################################################################################

println("done (took $(toq()) seconds)")

end
