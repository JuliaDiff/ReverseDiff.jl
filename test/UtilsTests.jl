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
tn_int, tm_int = Tracked(n, Int, ntp), Tracked(m, Int, ntp)
tn_float, tm_float = Tracked(n, ntp), Tracked(m, ntp)
tx_int, ty_int = map(t -> Tracked(t, Int, ntp), x), map(t -> Tracked(t, Int, ntp), y)
tx_float, ty_float = map(t -> Tracked(t, ntp), x), map(t -> Tracked(t, ntp), y)

@test tracked_is(track(n, tp), tn_float)
@test tracked_is(track(n, Int, tp), tn_int)
@test tracked_is(track(n, Int, ntp), tn_int)

@test tracked_is(track(x, tp), tx_float)
@test tracked_is(track(x, Int, tp), tx_int)
@test tracked_is(track(x, Int, ntp), tx_int)

@test tracked_is(track((n, m), tp), (tn_float, tm_float))
@test tracked_is(track((n, m), Int, tp), (tn_int, tm_int))

@test tracked_is(track((x, y), tp), (tx_float, ty_float))
@test tracked_is(track((x, y), Int, tp), (tx_int, ty_int))

tx_int_sim = similar(tx_int)
tx_float_sim = similar(tx_float)
track!(tx_int_sim, x, tp)
@test tracked_is(tx_int_sim, tx_int)
track!(tx_float_sim, x, tp)
@test tracked_is(tx_float_sim, tx_float)

tx_int_sim = similar(tx_int)
tx_float_sim = similar(tx_float)
track!(tx_int_sim, x, ntp)
@test tracked_is(tx_int_sim, tx_int)
track!(tx_float_sim, x, ntp)
@test tracked_is(tx_float_sim, tx_float)

tx_int_sim, ty_int_sim = similar(tx_int), similar(ty_int)
tx_float_sim, ty_float_sim = similar(tx_float), similar(ty_float)
track!((tx_int_sim, ty_int_sim), (x, y), tp)
@test tracked_is(tx_int_sim, tx_int)
@test tracked_is(ty_int_sim, ty_int)
track!((tx_float_sim, ty_float_sim), (x, y), tp)
@test tracked_is(tx_float_sim, tx_float)
@test tracked_is(ty_float_sim, ty_float)

tx_int_sim, ty_int_sim = similar(tx_int), similar(ty_int)
tx_float_sim, ty_float_sim = similar(tx_float), similar(ty_float)
track!((tx_int_sim, ty_int_sim), (x, y), ntp)
@test tracked_is(tx_int_sim, tx_int)
@test tracked_is(ty_int_sim, ty_int)
track!((tx_float_sim, ty_float_sim), (x, y), ntp)
@test tracked_is(tx_float_sim, tx_float)
@test tracked_is(ty_float_sim, ty_float)

track!(tn_int, m, tp)
@test tracked_is(tn_int, tm_int)
track!(tn_int, n, tp)

track!(tn_float, m, tp)
@test tracked_is(tn_float, tm_float)
track!(tn_float, n, tp)

track!((tm_int, tn_int), (n, m), tp)
@test value(tm_int) === n
@test value(tn_int) === m
track!((tm_int, tn_int), (m, n), tp)

track!(tn_int, m, ntp)
@test tracked_is(tn_int, tm_int)
track!(tn_int, n, ntp)

track!(tn_float, m, ntp)
@test tracked_is(tn_float, tm_float)
track!(tn_float, n, ntp)

track!((tm_int, tn_int), (n, m), ntp)
@test value(tm_int) === n
@test value(tn_int) === m
track!((tm_int, tn_int), (m, n), ntp)

##################################
# array accessors/tape selection #
##################################

tp = Tape()
ntp = Nullable(tp)
x, y = rand(3), rand(3, 2)
tx, ty = track(x, tp), track(y, tp)

@test value(tx) == x
@test value(ty) == y

x_sim, y_sim = similar(x), similar(y)
RDP.value!(x_sim, tx)
RDP.value!(y_sim, ty)
@test x_sim == x
@test y_sim == y

@test all(adjoint(tx) .== zero(first(x)))
@test all(adjoint(ty) .== zero(first(y)))

x_sim, y_sim = similar(x), similar(y)
RDP.adjoint!(x_sim, tx)
RDP.adjoint!(y_sim, ty)
@test all(x_sim .== zero(first(x)))
@test all(y_sim .== zero(first(y)))

@test tape(tx) === ntp
@test tape(tx, ty) === ntp
@test tape(tx, first(tx)) === ntp
@test tape(first(tx), tx) === ntp
@test tape(tx, Tracked(1)) === ntp
@test tape(Tracked(1), tx) === ntp
@test tape(tx, [Tracked(1)]) === ntp
@test tape([Tracked(1)], tx) === ntp
@test tape([Tracked(1), first(tx)]) === ntp
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
