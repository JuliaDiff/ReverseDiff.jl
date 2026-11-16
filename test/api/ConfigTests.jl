using ReverseDiff, Test

include(joinpath(dirname(@__FILE__), "../utils.jl"))

issimilar(x::Nothing, y::Nothing) = true
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
zt = similar(z, ReverseDiff.TrackedReal{eltype(z), eltype(z), Nothing})
@test issimilar(cfg.input, track(x, eltype(z), tp))
@test issimilar(cfg.output, track!(zt, z, tp))
@test cfg.tape === tp

cfg = JacobianConfig(z, (x, y), tp)
zt = similar(z, ReverseDiff.TrackedReal{eltype(z), eltype(z), Nothing})
@test issimilar(cfg.output, track!(zt, z, tp))
@test issimilar(cfg.input, (track(x, eltype(z), tp), track(y, eltype(z), tp)))
@test cfg.tape === tp

cfg1 = JacobianConfig(z, (x, y), tp)
zt = similar(z, ReverseDiff.TrackedReal{eltype(z), eltype(z), Nothing})
cfg2 = JacobianConfig(DiffResults.JacobianResult(z), (x, y), tp)
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

cfg = HessianConfig(DiffResults.HessianResult(y), x, gtp, jtp)

# These two act weird: work locally but fail in test mode
#@test issimilar(cfg.gradient_config, GradientConfig(track(x, Int), gtp))
#@test issimilar(cfg.jacobian_config, JacobianConfig(y, x, jtp))

############
# deepcopy #
############

@testset "`deepcopy` of a config is independent" begin
    x, y = [1.0, 2.0], [3.0, 4.0]
    I2 = [2.0 0.0; 0.0 2.0]
    cases = (
        (cfg -> ReverseDiff.gradient!(zeros(2), x -> sum(abs2, x), x, cfg), GradientConfig(x), [2.0, 4.0]),
        (cfg -> ReverseDiff.gradient!((zeros(2), zeros(2)), (x, y) -> sum(x .* y), (x, y), cfg), GradientConfig((x, y)), (y, x)),
        (cfg -> ReverseDiff.jacobian!(zeros(2, 2), x -> 2 .* x, x, cfg), JacobianConfig(x), I2),
        (cfg -> ReverseDiff.jacobian!(zeros(2, 2), (z, x) -> (z .= 2 .* x), zeros(2), x, cfg), JacobianConfig(zeros(2), x), I2),
        (cfg -> ReverseDiff.hessian!(zeros(2, 2), x -> sum(abs2, x), x, cfg), HessianConfig(x), I2),
    )
    for (run, cfg, expected) in cases
        copied = deepcopy(cfg)
        @test run(copied) == expected
        @test run(cfg) == expected
        @test run(copied) == expected
    end
end
