module TapeTests

using ReverseDiffPrototype, Base.Test
using ReverseDiffPrototype: TapeNode, Tape, Tracked

const RDP = ReverseDiffPrototype

include("utils.jl")

println("testing Tape/TapeNode types...")
tic()

############################################################################################

#################
# TapeNode/Tape #
#################

x, y, k = [1, 2, 3], [4, 5, 6], 7
z = x + y + k
c = []
tn = TapeNode(+, (x, y, k), z, c)
@test tn.func === +
@test tn.inputs === (x, y, k)
@test tn.outputs === z
@test tn.cache === c

tp = Tape()
ntp = Nullable(tp)
RDP.record!(ntp, +, (x, y, k), z, c)
tp1 = first(tp)
@test tp1 == tn
@test tp1.inputs[1] !== x
@test tp1.inputs[2] !== y
@test tp1.inputs[3] === k
@test tp1.outputs !== z
@test tp1.cache === c

ntp = Nullable{Tape}()
RDP.record!(ntp, +, (x, y, k), z, c)
@test ntp === Nullable{Tape}()

t = Tracked(1)
x = [t, t]
@test RDP.capture(t) === t

cx = RDP.capture(x)
@test cx !== x
@test cx == x
@test cx[1] === x[1]
@test cx[2] === x[2]

cs = RDP.capture((x, t, x))
@test cs[1] !== x
@test cs[1] == x
@test cs[1][1] === x[1]
@test cs[1][2] === x[2]
@test cs[2] === t
@test cs[3] !== x
@test cs[3] == x
@test cs[3][1] === x[1]
@test cs[3][2] === x[2]

############################################################################################

println("done (took $(toq()) seconds)")

end # module
