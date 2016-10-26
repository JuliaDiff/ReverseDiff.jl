module UtilsTests

using ReverseDiffPrototype, Base.Test

include("utils.jl")

println("testing utilities...")
tic()

############################################################################################

################
# track/track! #
################

tp = Tape()
ntp = Nullable(tp)
n, m = rand(), rand()
x, y = rand(3), rand(3, 3)
nt_int, mt_int = Tracked(n, Int, ntp), Tracked(m, Int, ntp)
nt_float, mt_float = Tracked(n, ntp), Tracked(m, ntp)
xt_int, yt_int = map(t -> Tracked(t, Int, ntp), x), map(t -> Tracked(t, Int, ntp), y)
xt_float, yt_float = map(t -> Tracked(t, ntp), x), map(t -> Tracked(t, ntp), y)

@test tracked_is(track(n, tp), nt_float)
@test tracked_is(track(n, Int, tp), nt_int)
@test tracked_is(track(n, Int, ntp), nt_int)

@test tracked_is(track(x, tp), xt_float)
@test tracked_is(track(x, Int, tp), xt_int)
@test tracked_is(track(x, Int, ntp), xt_int)

@test tracked_is(track((n, m), tp), (nt_float, mt_float))
@test tracked_is(track((n, m), Int, tp), (nt_int, mt_int))

@test tracked_is(track((x, y), tp), (xt_float, yt_float))
@test tracked_is(track((x, y), Int, tp), (xt_int, yt_int))

xt_int_sim = similar(xt_int)
xt_float_sim = similar(xt_float)
track!(xt_int_sim, x, tp)
@test tracked_is(xt_int_sim, xt_int)
track!(xt_float_sim, x, tp)
@test tracked_is(xt_float_sim, xt_float)

xt_int_sim = similar(xt_int)
xt_float_sim = similar(xt_float)
track!(xt_int_sim, x, ntp)
@test tracked_is(xt_int_sim, xt_int)
track!(xt_float_sim, x, ntp)
@test tracked_is(xt_float_sim, xt_float)

xt_int_sim, yt_int_sim = similar(xt_int), similar(yt_int)
xt_float_sim, yt_float_sim = similar(xt_float), similar(yt_float)
track!((xt_int_sim, yt_int_sim), (x, y), tp)
@test tracked_is(xt_int_sim, xt_int)
@test tracked_is(yt_int_sim, yt_int)
track!((xt_float_sim, yt_float_sim), (x, y), tp)
@test tracked_is(xt_float_sim, xt_float)
@test tracked_is(yt_float_sim, yt_float)

xt_int_sim, yt_int_sim = similar(xt_int), similar(yt_int)
xt_float_sim, yt_float_sim = similar(xt_float), similar(yt_float)
track!((xt_int_sim, yt_int_sim), (x, y), ntp)
@test tracked_is(xt_int_sim, xt_int)
@test tracked_is(yt_int_sim, yt_int)
track!((xt_float_sim, yt_float_sim), (x, y), ntp)
@test tracked_is(xt_float_sim, xt_float)
@test tracked_is(yt_float_sim, yt_float)

track!(nt_int, m, tp)
@test tracked_is(nt_int, mt_int)
track!(nt_int, n, tp)

track!(nt_float, m, tp)
@test tracked_is(nt_float, mt_float)
track!(nt_float, n, tp)

track!((mt_int, nt_int), (n, m), tp)
@test value(mt_int) === n
@test value(nt_int) === m
track!((mt_int, nt_int), (m, n), tp)

track!(nt_int, m, ntp)
@test tracked_is(nt_int, mt_int)
track!(nt_int, n, ntp)

track!(nt_float, m, ntp)
@test tracked_is(nt_float, mt_float)
track!(nt_float, n, ntp)

track!((mt_int, nt_int), (n, m), ntp)
@test value(mt_int) === n
@test value(nt_int) === m
track!((mt_int, nt_int), (m, n), ntp)

##################################
# array accessors/tape selection #
##################################

tp = Tape()
ntp = Nullable(tp)
x, y = rand(3), rand(3, 2)
xt, yt = track(x, tp), track(y, tp)

@test value(xt) == x
@test value(yt) == y

x_sim, y_sim = similar(x), similar(y)
RDP.value!(x_sim, xt)
RDP.value!(y_sim, yt)
@test x_sim == x
@test y_sim == y

@test all(adjoint(xt) .== zero(first(x)))
@test all(adjoint(yt) .== zero(first(y)))

x_sim, y_sim = similar(x), similar(y)
RDP.adjoint!(x_sim, xt)
RDP.adjoint!(y_sim, yt)
@test all(x_sim .== zero(first(x)))
@test all(y_sim .== zero(first(y)))

@test tape(xt) === ntp
@test tape(xt, yt) === ntp
@test tape(xt, first(xt)) === ntp
@test tape(first(xt), xt) === ntp
@test tape(xt, Tracked(1)) === ntp
@test tape(Tracked(1), xt) === ntp
@test tape(xt, [Tracked(1)]) === ntp
@test tape([Tracked(1)], xt) === ntp
@test tape([Tracked(1), first(xt)]) === ntp
@test isnull(tape([Tracked(1)]))
@test isnull(tape([Tracked(1)], [Tracked(1)]))

#################
# seed!/unseed! #
#################

tp = Tape()
ntp = Nullable(tp)

@test tracked_is(RDP.seed!(Tracked(1, 0, ntp)), Tracked(1, 1, ntp))

node = TapeNode(+, (Tracked(2, ntp), Tracked(1, ntp)), Tracked(3, ntp), nothing)
@test tracked_is(RDP.seed!(node).outputs, Tracked(3, 1, ntp))

@test tracked_is(RDP.unseed!(Tracked(1, 1, ntp)), Tracked(1, 0, ntp))

node = TapeNode(+, (Tracked(2, 2, ntp), Tracked(1, 3, ntp)), Tracked(3, 4, ntp), nothing)
RDP.unseed!(node)
@test adjoint(node.inputs[1]) === 0
@test adjoint(node.inputs[2]) === 0
@test adjoint(node.outputs) === 0

tp2 = [TapeNode(+, (Tracked(2, 2, ntp), Tracked(1, 3, ntp)), Tracked(3, 4, ntp), nothing),
       TapeNode(+, Tracked(1.0, 3.0, ntp), (Tracked(51.4, 3.1, ntp), Tracked(3, 4, ntp)), nothing)]
RDP.unseed!(tp2)
@test adjoint(tp2[1].inputs[1]) === 0
@test adjoint(tp2[1].inputs[2]) === 0
@test adjoint(tp2[1].outputs) === 0
@test adjoint(tp2[2].inputs) === 0.0
@test adjoint(tp2[2].outputs[1]) === 0.0
@test adjoint(tp2[2].outputs[2]) === 0

#######################
# adjoint propagation #
#######################

tp = Tape()
ntp = Nullable(tp)
genarr = () -> [Tracked(rand(), rand(), ntp) for i in 1:3]

x, y = genarr(), genarr()
xadj, yadj = adjoint(x), adjoint(y)
RDP.extract_and_decrement_adjoint!(x, y)
@test adjoint(x) == (xadj - yadj)

x, y = genarr(), genarr()
xadj, yadj = adjoint(x), adjoint(y)
RDP.extract_and_increment_adjoint!(x, y)
@test adjoint(x) == (xadj + yadj)

x, y = genarr(), rand(3)
xadj = adjoint(x)
RDP.increment_adjoint!(x, y)
@test adjoint(x) == (xadj + y)

x = genarr()
xadj = adjoint(x)
RDP.increment_adjoint!(x, 3)
@test adjoint(x) == (xadj + 3)

k = Tracked(1, 3, ntp)
RDP.increment_adjoint!(k, 3)
@test tracked_is(k, Tracked(1, 6, ntp))

############################################################################################

println("done (took $(toq()) seconds)")

end # module
