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

#############
# recording #
#############

@testset "a config holding a recorded tape cannot record another" begin
    x = [1.0, 2.0]
    cfg = GradientConfig(x)
    ReverseDiff.GradientTape(sum, x, cfg)
    @test_throws ArgumentError ReverseDiff.GradientTape(sum, x, cfg)
end

############
# deepcopy #
############

# `MCMCThreads` deepcopies models, which may hold configs that were already used

sum_abs2(x) = sum(abs2, x)
double(x) = 2 .* x

@testset "`deepcopy` of a `GradientConfig`" begin
    x = [1.0, 2.0]
    cfg = GradientConfig(x)
    @test ReverseDiff.gradient(sum_abs2, x, deepcopy(cfg)) == [2.0, 4.0]
    @test ReverseDiff.gradient(sum_abs2, x, cfg) == [2.0, 4.0]
    @test ReverseDiff.gradient(sum_abs2, x, deepcopy(cfg)) == [2.0, 4.0]
end

@testset "`deepcopy` of a `JacobianConfig`" begin
    x = [1.0, 2.0]
    cfg = JacobianConfig(x)
    ReverseDiff.jacobian(double, x, cfg)
    @test ReverseDiff.jacobian(double, x, deepcopy(cfg)) == [2.0 0.0; 0.0 2.0]
    @test ReverseDiff.jacobian(double, x, cfg) == [2.0 0.0; 0.0 2.0]
end

@testset "`deepcopy` of a `HessianConfig`" begin
    x = [1.0, 2.0]
    cfg = HessianConfig(x)
    ReverseDiff.hessian(sum_abs2, x, cfg)
    @test ReverseDiff.hessian(sum_abs2, x, deepcopy(cfg)) == [2.0 0.0; 0.0 2.0]
    @test ReverseDiff.hessian(sum_abs2, x, cfg) == [2.0 0.0; 0.0 2.0]
end
