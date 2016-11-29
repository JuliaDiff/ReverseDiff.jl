# ReverseDiff API

```@meta
CurrentModule = ReverseDiff
```

## Gradients of `f(x::AbstractArray{Real}...)::Real`

```@docs
ReverseDiff.gradient
ReverseDiff.gradient!
```

## Jacobians of `f(x::AbstractArray{Real}...)::AbstractArray{Real}`

```@docs
ReverseDiff.jacobian
ReverseDiff.jacobian!
```

## Hessians of `f(x::AbstractArray{Real})::Real`

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
