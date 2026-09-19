module QATests

using ReverseDiff, Test
using Aqua: Aqua
using ExplicitImports: ExplicitImports
using JET: JET
using StaticArrays: StaticArrays
using StaticArraysCore: StaticArraysCore

# Activates `StatisticsExt`: `runtests.jl` includes this file after `LinAlgTests`
using Statistics

const STATISTICS_EXT = Base.get_extension(ReverseDiff, :StatisticsExt)

# JET >= 0.11.4 requires Julia >= 1.12, so e.g. the LTS resolves JET 0.9.18 which does not
# define `JET_AVAILABLE`. A version check instead of `isdefined` errors, rather than skips
# silently, if JET removes it again.
const JET_AVAILABLE = pkgversion(JET) >= v"0.12" && JET.JET_AVAILABLE

@test STATISTICS_EXT !== nothing

@testset "Aqua" begin
    # `ambiguities`: methods generated with `@eval` clash with `Base` and `LinearAlgebra`
    Aqua.test_all(ReverseDiff; ambiguities=(; broken=true))
end

@testset "ExplicitImports" begin
    # Using ReverseDiff's internals inside its own extension is fine, but ExplicitImports cannot
    # be told so: a package extension is a top-level module, so `Base.moduleroot(STATISTICS_EXT)`
    # is `StatisticsExt` rather than `ReverseDiff` and `allow_internal_imports`/
    # `allow_internal_accesses` never apply, while `ignore=(STATISTICS_EXT,)` is rejected because
    # an extension is not a submodule. Listing the names is the only way to express it -- and it
    # keeps the extension's dependence on internals visible and reviewable.
    RD_INTERNALS = (
        :SpecialInstruction, :TrackedArray, :deriv, :increment_deriv!, :istracked, :record!,
        :special_forward_exec!, :special_reverse_exec!, :tape, :track, :unseed!, :value, :value!,
    )
    # Non-public names in `Base` and in dependencies that predate the `public` keyword.
    # ReverseDiff itself cannot use `public` either: it requires Julia >= 1.11 and the LTS is 1.10.
    UPSTREAM_INTERNALS = (
        # `Base` and `Base.Broadcast`
        :Broadcasted, :LogicalIndex, :RefValue, :broadcasted, :flatten, :materialize, :rtoldefault,
        # `Core.Compiler`
        :Compiler, :return_type,
        # `DiffResults`
        :DiffResult, :GradientResult, :ImmutableDiffResult, :derivative, :gradient, :gradient!,
        :hessian, :jacobian,
        # `DiffRules`, `ForwardDiff` and `FunctionWrappers`
        :diffrules, :Dual, :derivative!, :FunctionWrapper,
    )
    NONPUBLIC = (RD_INTERNALS..., UPSTREAM_INTERNALS...)

    for m in (ReverseDiff, STATISTICS_EXT)
        @testset "$(nameof(m))" begin
            @test ExplicitImports.check_no_implicit_imports(m) === nothing
            @test ExplicitImports.check_no_stale_explicit_imports(m) === nothing
            @test ExplicitImports.check_no_self_qualified_accesses(m) === nothing
            @test ExplicitImports.check_all_qualified_accesses_via_owners(m) === nothing
            # `SVector` is exported by `StaticArrays` but owned by `StaticArraysCore`
            @test ExplicitImports.check_all_explicit_imports_via_owners(
                m; skip=(Base => Core, StaticArrays => StaticArraysCore)
            ) === nothing
            @test ExplicitImports.check_all_explicit_imports_are_public(m; ignore=NONPUBLIC) ===
                nothing
            @test ExplicitImports.check_all_qualified_accesses_are_public(m; ignore=NONPUBLIC) ===
                nothing
        end
    end
end

@testset "JET" begin
    if JET_AVAILABLE
        target_modules = (ReverseDiff, STATISTICS_EXT)

        f(x) = sum(abs2, x) + prod(x)
        g(x) = x .^ 2 .+ 1
        x = rand(4)

        gradient_tape = ReverseDiff.GradientTape(f, x)
        jacobian_tape = ReverseDiff.JacobianTape(g, x)
        hessian_tape = ReverseDiff.HessianTape(f, x)
        compiled_gradient_tape = ReverseDiff.compile(gradient_tape)
        compiled_jacobian_tape = ReverseDiff.compile(jacobian_tape)

        gradient_result = similar(x)
        jacobian_result = similar(x, 4, 4)
        hessian_result = similar(x, 4, 4)

        # `report_package` reports hundreds of problems in code generated with `@eval`
        JET.@test_call target_modules = target_modules ReverseDiff.GradientConfig(x)
        JET.@test_call target_modules = target_modules ReverseDiff.JacobianConfig(x)
        JET.@test_call target_modules = target_modules ReverseDiff.HessianConfig(x)
        JET.@test_call target_modules = target_modules ReverseDiff.GradientTape(f, x)
        JET.@test_call target_modules = target_modules ReverseDiff.JacobianTape(g, x)
        JET.@test_call target_modules = target_modules ReverseDiff.compile(gradient_tape)
        JET.@test_call target_modules = target_modules ReverseDiff.jacobian(g, x)
        JET.@test_call target_modules = target_modules ReverseDiff.jacobian(
            g, x, ReverseDiff.JacobianConfig(x)
        )
        JET.@test_call target_modules = target_modules ReverseDiff.jacobian!(
            jacobian_result, jacobian_tape, x
        )
        JET.@test_call target_modules = target_modules ReverseDiff.jacobian!(
            jacobian_result, compiled_jacobian_tape, x
        )
        JET.@test_call target_modules = target_modules ReverseDiff.hessian!(
            hessian_result, hessian_tape, x
        )

        # Scalar outputs reach `pull_value!(::TrackedReal{V,D,Nothing})` in `src/tracked.jl`,
        # where `t.origin` is only guarded by the run-time check `hasorigin(t)`
        JET.@test_call broken = true target_modules = target_modules ReverseDiff.HessianTape(
            f, x
        )
        JET.@test_call broken = true target_modules = target_modules ReverseDiff.gradient(f, x)
        JET.@test_call broken = true target_modules = target_modules ReverseDiff.gradient(
            f, x, ReverseDiff.GradientConfig(x)
        )
        JET.@test_call broken = true target_modules = target_modules ReverseDiff.gradient!(
            gradient_result, gradient_tape, x
        )
        JET.@test_call broken = true target_modules = target_modules ReverseDiff.gradient!(
            gradient_result, compiled_gradient_tape, x
        )
        JET.@test_call broken = true target_modules = target_modules ReverseDiff.hessian(f, x)

        # Executing an uncompiled tape dispatches on `AbstractInstruction` at run time
        JET.@test_opt target_modules = target_modules ReverseDiff.gradient!(
            gradient_result, compiled_gradient_tape, x
        )
        JET.@test_opt broken = true target_modules = target_modules ReverseDiff.gradient!(
            gradient_result, gradient_tape, x
        )
        JET.@test_opt broken = true target_modules = target_modules ReverseDiff.jacobian!(
            jacobian_result, jacobian_tape, x
        )
    end
end

end # module
