module OptionsTests

using ReverseDiffPrototype, Base.Test

include("../utils.jl")

println("testing Options...")
tic()

issimilar(x::AbstractArray, y::AbstractArray) = typeof(x) === typeof(y) && size(x) === size(y)
issimilar(x::Options, y::Options) = issimilar(x.state, y.state) && x.tape === y.tape
issimilar(x::Tuple, y::Tuple) = all(map(issimilar, x, y))

############################################################################################

###########
# Options #
###########

tp = Tape()
x, y = rand(3), rand(3, 2)
z = rand(Int, 4)

opts = Options(x, tp)
@test issimilar(opts.state, track(x, tp))
@test opts.tape === tp

opts = Options(x, Int, tp)
@test issimilar(opts.state, track(x, Int, tp))
@test opts.tape === tp

opts = Options((x, y), tp)
@test issimilar(opts.state, track((x, y), tp))
@test opts.tape === tp

opts = Options((x, y), Int, tp)
@test issimilar(opts.state, track((x, y), Int, tp))
@test opts.tape === tp

opts = Options(z, x, tp)
@test issimilar(opts.state, (track(z, tp), track(x, eltype(z), tp)))
@test opts.tape === tp

opts = Options(z, (x, y), tp)
@test issimilar(opts.state, (track(z, tp), track((x, y), eltype(z), tp)))
@test opts.tape === tp

opts1 = Options(z, (x, y), tp)
opts2 = Options(DiffBase.JacobianResult(z), (x, y), tp)
@test issimilar(opts1, opts2)

##################
# HessianOptions #
##################

gtp, jtp = Tape(), Tape()
x, y = rand(3), rand(Int, 3)

opts = HessianOptions(x, gtp, jtp)
@test issimilar(RDP.gradient_options(opts), Options(track(x), gtp))
@test issimilar(RDP.jacobian_options(opts), Options(x, jtp))

opts = HessianOptions(x, Int, gtp, jtp)
@test issimilar(RDP.gradient_options(opts), Options(track(x, Int), gtp))
@test issimilar(RDP.jacobian_options(opts), Options(x, Int, jtp))

opts = HessianOptions(y, x, gtp, jtp)
@test issimilar(RDP.gradient_options(opts), Options(track(x, Int), gtp))
@test issimilar(RDP.jacobian_options(opts), Options(y, x, jtp))

opts1 = HessianOptions(x, Int, gtp, jtp)
opts2 = HessianOptions(DiffBase.HessianResult(x), Int, gtp, jtp)
@test issimilar(RDP.gradient_options(opts1), RDP.gradient_options(opts2))
@test issimilar(RDP.jacobian_options(opts1), RDP.jacobian_options(opts2))

############################################################################################

println("done (took $(toq()) seconds)")

end # module
