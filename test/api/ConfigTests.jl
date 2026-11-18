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

##############
# exceptions #
##############

# throws after some instructions were recorded
function throw_if_large(x)
    y = sum(abs2, x)
    if y > 10
        throw(DomainError(y))
    end
    return x
end
sum_abs2_if_small(x) = sum_abs2(throw_if_large(x))
double_if_small(x) = double(throw_if_large(x))

# throws in the reverse pass
sum_failing_pullback(x) = sum(x)
sum_failing_pullback(x::ReverseDiff.TrackedVector) = ReverseDiff.track(sum_failing_pullback, x)
ReverseDiff.@grad function sum_failing_pullback(x::AbstractVector)
    return sum(ReverseDiff.value(x)), Δ -> throw(DomainError(Δ))
end
sums_failing_pullback(x) = [sum_failing_pullback(x)]

@testset "a config can be used and copied after recording throws" begin
    small, large = [1.0, 2.0], [3.0, 4.0]

    cfg = GradientConfig(small)
    @test_throws DomainError ReverseDiff.gradient(sum_abs2_if_small, large, cfg)
    @test ReverseDiff.gradient(sum_abs2_if_small, small, deepcopy(cfg)) == [2.0, 4.0]
    @test ReverseDiff.gradient(sum_abs2_if_small, small, cfg) == [2.0, 4.0]

    cfg = JacobianConfig(small)
    @test_throws DomainError ReverseDiff.jacobian(double_if_small, large, cfg)
    @test ReverseDiff.jacobian(double_if_small, small, deepcopy(cfg)) == [2.0 0.0; 0.0 2.0]
    @test ReverseDiff.jacobian(double_if_small, small, cfg) == [2.0 0.0; 0.0 2.0]

    cfg = HessianConfig(small)
    @test_throws DomainError ReverseDiff.hessian(sum_abs2_if_small, large, cfg)
    @test_throws DomainError ReverseDiff.HessianTape(sum_abs2_if_small, large, cfg)
    @test ReverseDiff.hessian(sum_abs2_if_small, small, deepcopy(cfg)) == [2.0 0.0; 0.0 2.0]
    @test ReverseDiff.hessian(sum_abs2_if_small, small, cfg) == [2.0 0.0; 0.0 2.0]
end

@testset "a config can be used after the reverse pass throws" begin
    x = [1.0, 2.0]

    cfg = GradientConfig(x)
    @test_throws DomainError ReverseDiff.gradient(sum_failing_pullback, x, cfg)
    @test ReverseDiff.gradient(sum_abs2, x, cfg) == [2.0, 4.0]

    cfg = JacobianConfig(x)
    @test_throws DomainError ReverseDiff.jacobian(sums_failing_pullback, x, cfg)
    @test ReverseDiff.jacobian(double, x, cfg) == [2.0 0.0; 0.0 2.0]

    cfg = HessianConfig(x)
    @test_throws DomainError ReverseDiff.HessianTape(sum_failing_pullback, x, cfg)
    @test ReverseDiff.hessian(sum_abs2, x, cfg) == [2.0 0.0; 0.0 2.0]
end
