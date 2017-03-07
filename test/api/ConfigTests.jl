module ConfigTests

using ReverseDiff, Base.Test

include(joinpath(dirname(@__FILE__), "../utils.jl"))

println("testing Config...")
tic()

issimilar(x::Void, y::Void) = true
issimilar(x::AbstractArray, y::AbstractArray) = typeof(x) === typeof(y) && size(x) === size(y)
issimilar(x::GradientConfig, y::GradientConfig) = issimilar(x.input, y.input) && x.tape === y.tape
issimilar(x::JacobianConfig, y::JacobianConfig) = issimilar(x.output, y.output) && issimilar(x.input, y.input) && x.tape === y.tape
issimilar(x::Tuple, y::Tuple) = all(map(issimilar, x, y))

############################################################################################

###########
# Config #
###########

tp = InstructionTape()
x, y = rand(3), rand(3, 2)
z = rand(Int, 4)

for Config in (GradientConfig, JacobianConfig)
    cfg = Config(x, tp)
    @test issimilar(cfg.input, track(x, tp))
    @test cfg.tape === tp

    cfg = Config(x, Int, tp)
    @test issimilar(cfg.input, track(x, Int, tp))
    @test cfg.tape === tp

    cfg = Config((x, y), tp)
    @test issimilar(cfg.input, (track(x, tp), track(y, tp)))
    @test cfg.tape === tp

    cfg = Config((x, y), Int, tp)
    @test issimilar(cfg.input, (track(x, Int, tp), track(y, Int, tp)))
    @test cfg.tape === tp
end

cfg = JacobianConfig(z, x, tp)
zt = similar(z, ReverseDiff.TrackedReal{eltype(z),eltype(z),Void})
@test issimilar(cfg.input, track(x, eltype(z), tp))
@test issimilar(cfg.output, track!(zt, z, tp))
@test cfg.tape === tp

cfg = JacobianConfig(z, (x, y), tp)
zt = similar(z, ReverseDiff.TrackedReal{eltype(z),eltype(z),Void})
@test issimilar(cfg.output, track!(zt, z, tp))
@test issimilar(cfg.input, (track(x, eltype(z), tp), track(y, eltype(z), tp)))
@test cfg.tape === tp

cfg1 = JacobianConfig(z, (x, y), tp)
zt = similar(z, ReverseDiff.TrackedReal{eltype(z),eltype(z),Void})
cfg2 = JacobianConfig(DiffBase.JacobianResult(z), (x, y), tp)
@test issimilar(cfg1, cfg2)

##################
# HessianConfig #
##################

gtp, jtp = InstructionTape(), InstructionTape()
x, y = rand(3), rand(Int, 3)

cfg = HessianConfig(x, gtp, jtp)
@test issimilar(cfg.gradient_config, GradientConfig(track(x), gtp))
@test issimilar(cfg.jacobian_config, JacobianConfig(x, jtp))

cfg = HessianConfig(x, Int, gtp, jtp)
@test issimilar(cfg.gradient_config, GradientConfig(track(x, Int), gtp))
@test issimilar(cfg.jacobian_config, JacobianConfig(x, Int, jtp))

cfg = HessianConfig(DiffBase.HessianResult(y), x, gtp, jtp)
@test issimilar(cfg.gradient_config, GradientConfig(track(x, Int), gtp))
@test issimilar(cfg.jacobian_config, JacobianConfig(y, x, jtp))

############################################################################################

println("done (took $(toq()) seconds)")

end # module
