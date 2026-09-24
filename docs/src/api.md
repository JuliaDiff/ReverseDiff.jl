# ReverseDiff API

```@meta
CurrentModule = ReverseDiff
```

## Gradients of `f(x::AbstractArray{<:Real}...)::Real`

```@docs
ReverseDiff.gradient
ReverseDiff.gradient!
```

### [Example](@id gradient-example)

```jldoctest gradient
julia> f(a, b) = sum(a' * b + a * b');

julia> a, b = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0], [2.0 0.0 1.0; 1.0 3.0 0.0; 0.0 1.0 2.0];

julia> inputs = (a, b);
```

For a one-off gradient, simply call `gradient`:

```jldoctest gradient
julia> using ReverseDiff: gradient

julia> ∇f = gradient(f, inputs);
```

The in-place `gradient!` writes the gradients into pre-allocated `results`:

```jldoctest gradient
julia> using ReverseDiff: gradient!

julia> results = (similar(a), similar(b));

julia> gradient!(results, f, inputs) == ∇f
true
```

In addition to the gradients, `DiffResult` instances from
[DiffResults.jl](https://github.com/JuliaDiff/DiffResults.jl) also store the value
`f(a, b)`:

```jldoctest gradient
julia> using DiffResults

julia> all_results = map(DiffResults.GradientResult, (similar(a), similar(b)));

julia> gradient!(all_results, f, inputs);

julia> map(DiffResults.value, all_results) == (f(a, b), f(a, b))
true

julia> map(DiffResults.gradient, all_results) == ∇f
true
```

Every such call allocates a new [`GradientConfig`](@ref ReverseDiff.GradientConfig) and
records a new tape. For repeated calls with inputs of the same shape and element type, a
pre-allocated `GradientConfig` avoids the former:

```jldoctest gradient
julia> using ReverseDiff: GradientConfig

julia> cfg = GradientConfig(inputs);

julia> gradient!(results, f, inputs, cfg) == ∇f
true
```

A pre-recorded [`GradientTape`](@ref ReverseDiff.GradientTape) also avoids re-recording the
tape. It is recorded once with inputs of the same shape and element type:

```jldoctest gradient
julia> using ReverseDiff: GradientTape

julia> f_tape = GradientTape(f, (rand(3, 3), rand(3, 3)));

julia> gradient!(results, f_tape, inputs) == ∇f
true
```

!!! warning
    A recorded tape, compiled or not, only replays the operations that were executed
    while recording it. It is only valid for inputs that take exactly the same path
    through `f`. For example, a branch that depends on the input values won't be
    re-evaluated. `f` above has no such control flow, so reusing its tape is safe. See
    [The `AbstractTape` API](@ref).

[`compile`](@ref ReverseDiff.compile) turns the tape into a more optimized representation.
Compiling has an upfront cost, but executing the compiled tape is typically the fastest
method when many gradients are needed:

```jldoctest gradient
julia> using ReverseDiff: compile

julia> compiled_f_tape = compile(f_tape);

julia> gradient!(results, compiled_f_tape, inputs) == ∇f
true
```

## Jacobians of `f(x::AbstractArray{<:Real}...)::AbstractArray{<:Real}`

```@docs
ReverseDiff.jacobian
ReverseDiff.jacobian!
```

### [Example](@id jacobian-example)

Consider an out-of-place function `f` and an in-place function `g!` that computes the same
output:

```jldoctest jacobian
julia> f(a, b) = (a + b) * (a * b)';

julia> using LinearAlgebra: mul!

julia> g!(out, a, b) = mul!(out, a + b, (a * b)');

julia> a, b = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0], [2.0 0.0 1.0; 1.0 3.0 0.0; 0.0 1.0 2.0];

julia> inputs = (a, b);

julia> out = similar(a);

julia> g!(out, a, b) == f(a, b)
true
```

For a one-off Jacobian, simply call `jacobian`:

```jldoctest jacobian
julia> using ReverseDiff: jacobian

julia> J = jacobian(f, inputs);

julia> jacobian(g!, out, inputs) == J
true
```

The in-place `jacobian!` writes the Jacobians into pre-allocated `results`:

```jldoctest jacobian
julia> using ReverseDiff: jacobian!

julia> results = (similar(a, 9, 9), similar(b, 9, 9));

julia> jacobian!(results, f, inputs) == J
true

julia> jacobian!(results, g!, out, inputs) == J
true
```

In addition to the Jacobians, `DiffResult` instances from
[DiffResults.jl](https://github.com/JuliaDiff/DiffResults.jl) also store the value
`f(a, b)`:

```jldoctest jacobian
julia> using DiffResults

julia> all_results = map(x -> DiffResults.JacobianResult(out, x), inputs);

julia> jacobian!(all_results, f, inputs);

julia> map(DiffResults.value, all_results) == (f(a, b), f(a, b))
true

julia> map(DiffResults.jacobian, all_results) == J
true

julia> jacobian!(all_results, g!, out, inputs);

julia> map(DiffResults.value, all_results) == (f(a, b), f(a, b))
true

julia> map(DiffResults.jacobian, all_results) == J
true
```

Every such call allocates a new [`JacobianConfig`](@ref ReverseDiff.JacobianConfig) and
records a new tape. For repeated calls with inputs (and, for `g!`, an output) of the same
shape and element type, a pre-allocated `JacobianConfig` avoids the former:

```jldoctest jacobian
julia> using ReverseDiff: JacobianConfig

julia> f_cfg = JacobianConfig(inputs);

julia> jacobian!(results, f, inputs, f_cfg) == J
true

julia> g!_cfg = JacobianConfig(out, inputs);

julia> jacobian!(results, g!, out, inputs, g!_cfg) == J
true
```

A pre-recorded [`JacobianTape`](@ref ReverseDiff.JacobianTape) also avoids re-recording the
tape. It is recorded once with inputs (and, for `g!`, an output) of the same shape and
element type:

```jldoctest jacobian
julia> using ReverseDiff: JacobianTape

julia> f_tape = JacobianTape(f, (rand(3, 3), rand(3, 3)));

julia> jacobian!(results, f_tape, inputs) == J
true

julia> g!_tape = JacobianTape(g!, rand(3, 3), (rand(3, 3), rand(3, 3)));

julia> jacobian!(results, g!_tape, inputs) == J
true
```

!!! warning
    As for gradients, a recorded tape, compiled or not, is only valid for inputs that take
    exactly the same path through the function as the ones it was recorded with. See
    [The `AbstractTape` API](@ref).

[`compile`](@ref ReverseDiff.compile) turns the tapes into more optimized representations.
Compiling has an upfront cost, but executing the compiled tapes is typically the fastest
method when many Jacobians are needed:

```jldoctest jacobian
julia> using ReverseDiff: compile

julia> compiled_f_tape = compile(f_tape);

julia> jacobian!(results, compiled_f_tape, inputs) == J
true

julia> compiled_g!_tape = compile(g!_tape);

julia> jacobian!(results, compiled_g!_tape, inputs) == J
true
```

## Hessians of `f(x::AbstractArray{<:Real})::Real`

```@docs
ReverseDiff.hessian
ReverseDiff.hessian!
```

## The `AbstractTape` API

ReverseDiff works by recording the target function's execution trace to a "tape", then
running the tape forwards and backwards to propagate new input values and derivative
information.

In many cases, it is the recording phase of this process that consumes the most time and
memory, while the forward and reverse execution passes are often fast and non-allocating.
Luckily, ReverseDiff provides the `AbstractTape` family of types, which enable the user to
*pre-record* a reusable tape for a given function and differentiation operation.

**Note that pre-recording a tape can only capture the the execution trace of the target
function with the given input values.** Therefore, re-running the tape (even with new input
values) will only execute the paths that were recorded using the original input values. In
other words, the tape cannot any re-enact branching behavior that depends on the input
values. You can guarantee your own safety in this regard by never using the `AbstractTape`
API with functions that contain control flow based on the input values.

Similarly to the branching issue, a tape is not guaranteed to capture any side-effects
caused or depended on by the target function.

```@docs
ReverseDiff.GradientTape
ReverseDiff.JacobianTape
ReverseDiff.HessianTape
ReverseDiff.compile
```

## The `AbstractConfig` API

For the sake of convenience and performance, all "extra" information used by ReverseDiff's
API methods is bundled up in the `ReverseDiff.AbstractConfig` family of types. These
types allow the user to easily feed several different parameters to ReverseDiff's API
methods, such as work buffers and tape configurations.

ReverseDiff's basic API methods will allocate these types automatically by default, but you
can reduce memory usage and improve performance if you preallocate them yourself.

```@docs
ReverseDiff.GradientConfig
ReverseDiff.JacobianConfig
ReverseDiff.HessianConfig
```

## Optimization Annotations

```@docs
ReverseDiff.@forward
ReverseDiff.@skip
```

## ChainRules integration

```@docs
ReverseDiff.@grad_from_chainrules
```
