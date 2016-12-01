
<a id='ReverseDiff-API-1'></a>

# ReverseDiff API




<a id='Gradients-of-f(x::AbstractArray{Real}...)::Real-1'></a>

## Gradients of `f(x::AbstractArray{Real}...)::Real`

<a id='ReverseDiff.gradient' href='#ReverseDiff.gradient'>#</a>
**`ReverseDiff.gradient`** &mdash; *Function*.



```
ReverseDiff.gradient(f, input, cfg::GradientConfig = GradientConfig(input))
```

If `input` is an `AbstractArray`, assume `f` has the form `f(::AbstractArray{Real})::Real` and return `∇f(input)`.

If `input` is a tuple of `AbstractArray`s, assume `f` has the form `f(::AbstractArray{Real}...)::Real` (such that it can be called as `f(input...)`) and return a `Tuple` where the `i`th element is the gradient of `f` w.r.t. `input[i].`

Note that `cfg` can be preallocated and reused for subsequent calls.

If possible, it is highly recommended to use `ReverseDiff.GradientTape` to prerecord `f`. Otherwise, this method will have to re-record `f`'s execution trace for every subsequent call.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/gradients.jl#L5-L20' class='documenter-source'>source</a><br>

<a id='ReverseDiff.gradient!' href='#ReverseDiff.gradient!'>#</a>
**`ReverseDiff.gradient!`** &mdash; *Function*.



```
ReverseDiff.gradient!(result, f, input, cfg::GradientConfig = GradientConfig(input))
```

Returns `result`. This method is exactly like `ReverseDiff.gradient(f, input, cfg)`, except it stores the resulting gradient(s) in `result` rather than allocating new memory.

`result` can be an `AbstractArray` or a `Tuple` of `AbstractArray`s. The `result` (or any of its elements, if `isa(result, Tuple)`), can also be a `DiffBase.DiffResult`, in which case the primal value `f(input)` (or `f(input...)`, if `isa(input, Tuple)`) will be stored in it as well.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/gradients.jl#L29-L39' class='documenter-source'>source</a><br>


```
ReverseDiff.gradient!(tape::Union{GradientTape,CompiledGradient}, input)
```

If `input` is an `AbstractArray`, assume `tape` represents a function of the form `f(::AbstractArray)::Real` and return `∇f(input)`.

If `input` is a tuple of `AbstractArray`s, assume `tape` represents a function of the form `f(::AbstractArray...)::Real` and return a `Tuple` where the `i`th element is the gradient of `f` w.r.t. `input[i].`


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/gradients.jl#L51-L60' class='documenter-source'>source</a><br>


```
ReverseDiff.gradient!(result, tape::Union{GradientTape,CompiledGradient}, input)
```

Returns `result`. This method is exactly like `ReverseDiff.gradient!(tape, input)`, except it stores the resulting gradient(s) in `result` rather than allocating new memory.

`result` can be an `AbstractArray` or a `Tuple` of `AbstractArray`s. The `result` (or any of its elements, if `isa(result, Tuple)`), can also be a `DiffBase.DiffResult`, in which case the primal value `f(input)` (or `f(input...)`, if `isa(input, Tuple)`) will be stored in it as well.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/gradients.jl#L67-L77' class='documenter-source'>source</a><br>


<a id='Jacobians-of-f(x::AbstractArray{Real}...)::AbstractArray{Real}-1'></a>

## Jacobians of `f(x::AbstractArray{Real}...)::AbstractArray{Real}`

<a id='ReverseDiff.jacobian' href='#ReverseDiff.jacobian'>#</a>
**`ReverseDiff.jacobian`** &mdash; *Function*.



```
ReverseDiff.jacobian(f, input, cfg::JacobianConfig = JacobianConfig(input))
```

If `input` is an `AbstractArray`, assume `f` has the form `f(::AbstractArray{Real})::AbstractArray{Real}` and return `J(f)(input)`.

If `input` is a tuple of `AbstractArray`s, assume `f` has the form `f(::AbstractArray{Real}...)::AbstractArray{Real}` (such that it can be called as `f(input...)`) and return a `Tuple` where the `i`th element is the  Jacobian of `f` w.r.t. `input[i].`

Note that `cfg` can be preallocated and reused for subsequent calls.

If possible, it is highly recommended to use `ReverseDiff.JacobianTape` to prerecord `f`. Otherwise, this method will have to re-record `f`'s execution trace for every subsequent call.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/jacobians.jl#L5-L21' class='documenter-source'>source</a><br>


```
ReverseDiff.jacobian(f!, output, input, cfg::JacobianConfig = JacobianConfig(output, input))
```

Exactly like `ReverseDiff.jacobian(f, input, cfg)`, except the target function has the form `f!(output::AbstractArray{Real}, input::AbstractArray{Real}...)`.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/jacobians.jl#L53-L58' class='documenter-source'>source</a><br>

<a id='ReverseDiff.jacobian!' href='#ReverseDiff.jacobian!'>#</a>
**`ReverseDiff.jacobian!`** &mdash; *Function*.



```
ReverseDiff.jacobian!(result, f, input, cfg::JacobianConfig = JacobianConfig(input))
```

Returns `result`. This method is exactly like `ReverseDiff.jacobian(f, input, cfg)`, except it stores the resulting Jacobian(s) in `result` rather than allocating new memory.

`result` can be an `AbstractArray` or a `Tuple` of `AbstractArray`s. The `result` (or any of its elements, if `isa(result, Tuple)`), can also be a `DiffBase.DiffResult`, in which case the primal value `f(input)` (or `f(input...)`, if `isa(input, Tuple)`) will be stored in it as well.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/jacobians.jl#L30-L40' class='documenter-source'>source</a><br>


```
ReverseDiff.jacobian!(result, f!, output, input, cfg::JacobianConfig = JacobianConfig(output, input))
```

Exactly like `ReverseDiff.jacobian!(result, f, input, cfg)`, except the target function has the form `f!(output::AbstractArray{Real}, input::AbstractArray{Real}...)`.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/jacobians.jl#L68-L73' class='documenter-source'>source</a><br>


```
ReverseDiff.jacobian!(tape::Union{JacobianTape,CompiledJacobian}, input)
```

If `input` is an `AbstractArray`, assume `tape` represents a function of the form `f(::AbstractArray{Real})::AbstractArray{Real}` or `f!(::AbstractArray{Real}, ::AbstractArray{Real})` and return `tape`'s Jacobian w.r.t. `input`.

If `input` is a tuple of `AbstractArray`s, assume `tape` represents a function of the form `f(::AbstractArray{Real}...)::AbstractArray{Real}` or `f!(::AbstractArray{Real}, ::AbstractArray{Real}...)` and return a `Tuple` where the `i`th element is `tape`'s Jacobian w.r.t. `input[i].`

Note that if `tape` represents a function of the form `f!(output, input...)`, you can only execute `tape` with new `input` values. There is no way to re-run `tape`'s tape with new `output` values; since `f!` can mutate `output`, there exists no stable "hook" for loading new `output` values into the tape.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/jacobians.jl#L87-L103' class='documenter-source'>source</a><br>


```
ReverseDiff.jacobian!(result, tape::Union{JacobianTape,CompiledJacobian}, input)
```

Returns `result`. This method is exactly like `ReverseDiff.jacobian!(tape, input)`, except it stores the resulting Jacobian(s) in `result` rather than allocating new memory.

`result` can be an `AbstractArray` or a `Tuple` of `AbstractArray`s. The `result` (or any of its elements, if `isa(result, Tuple)`), can also be a `DiffBase.DiffResult`, in which case the primal value of the target function will be stored in it as well.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/jacobians.jl#L110-L119' class='documenter-source'>source</a><br>


<a id='Hessians-of-f(x::AbstractArray{Real})::Real-1'></a>

## Hessians of `f(x::AbstractArray{Real})::Real`

<a id='ReverseDiff.hessian' href='#ReverseDiff.hessian'>#</a>
**`ReverseDiff.hessian`** &mdash; *Function*.



```
ReverseDiff.hessian(f, input::AbstractArray, cfg::HessianConfig = HessianConfig(input))
```

Given `f(input::AbstractArray{Real})::Real`, return `f`s Hessian w.r.t. to the given `input`.

Note that `cfg` can be preallocated and reused for subsequent calls.

If possible, it is highly recommended to use `ReverseDiff.HessianTape` to prerecord `f`. Otherwise, this method will have to re-record `f`'s execution trace for every subsequent call.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/hessians.jl#L8-L19' class='documenter-source'>source</a><br>

<a id='ReverseDiff.hessian!' href='#ReverseDiff.hessian!'>#</a>
**`ReverseDiff.hessian!`** &mdash; *Function*.



```
ReverseDiff.hessian!(result::AbstractArray, f, input::AbstractArray, cfg::HessianConfig = HessianConfig(input))

ReverseDiff.hessian!(result::DiffResult, f, input::AbstractArray, cfg::HessianConfig = HessianConfig(result, input))
```

Returns `result`. This method is exactly like `ReverseDiff.hessian(f, input, cfg)`, except it stores the resulting Hessian in `result` rather than allocating new memory.

If `result` is a `DiffBase.DiffResult`, the primal value `f(input)` and the gradient `∇f(input)` will be stored in it along with the Hessian `H(f)(input)`.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/hessians.jl#L28-L38' class='documenter-source'>source</a><br>


```
ReverseDiff.hessian!(tape::Union{HessianTape,CompiledHessian}, input)
```

Assuming `tape` represents a function of the form `f(::AbstractArray{Real})::Real`, return the Hessian `H(f)(input)`.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/hessians.jl#L63-L68' class='documenter-source'>source</a><br>


```
ReverseDiff.hessian!(result::AbstractArray, tape::Union{HessianTape,CompiledHessian}, input)

ReverseDiff.hessian!(result::DiffResult, tape::Union{HessianTape,CompiledHessian}, input)
```

Returns `result`. This method is exactly like `ReverseDiff.hessian!(tape, input)`, except it stores the resulting Hessian in `result` rather than allocating new memory.

If `result` is a `DiffBase.DiffResult`, the primal value `f(input)` and the gradient `∇f(input)` will be stored in it along with the Hessian `H(f)(input)`.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/hessians.jl#L75-L85' class='documenter-source'>source</a><br>


<a id='The-AbstractTape-API-1'></a>

## The `AbstractTape` API


ReverseDiff works by recording the target function's execution trace to a "tape", then running the tape forwards and backwards to propagate new input values and derivative information.


In many cases, it is the recording phase of this process that consumes the most time and memory, while the forward and reverse execution passes are often fast and non-allocating. Luckily, ReverseDiff provides the `AbstractTape` family of types, which enable the user to *pre-record* a reusable tape for a given function and differentiation operation.


**Note that pre-recording a tape can only capture the the execution trace of the target function with the given input values.** Therefore, re-running the tape (even with new input values) will only execute the paths that were recorded using the original input values. In other words, the tape cannot any re-enact branching behavior that depends on the input values. You can guarantee your own safety in this regard by never using the `AbstractTape` API with functions that contain control flow based on the input values.


Similarly to the branching issue, a tape is not guaranteed to capture any side-effects caused or depended on by the target function.

<a id='ReverseDiff.GradientTape' href='#ReverseDiff.GradientTape'>#</a>
**`ReverseDiff.GradientTape`** &mdash; *Type*.



```
ReverseDiff.GradientTape(f, input, cfg::GradientConfig = GradientConfig(input))
```

Return a `GradientTape` instance containing a pre-recorded execution trace of `f` at the given `input`.

This `GradientTape` can then be passed to `ReverseDiff.gradient!` to take gradients of the execution trace with new `input` values.

See `ReverseDiff.gradient` for a description of acceptable types for `input`.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/tape.jl#L87-L97' class='documenter-source'>source</a><br>

<a id='ReverseDiff.JacobianTape' href='#ReverseDiff.JacobianTape'>#</a>
**`ReverseDiff.JacobianTape`** &mdash; *Type*.



```
ReverseDiff.JacobianTape(f, input, cfg::JacobianConfig = JacobianConfig(input))
```

Return a `JacobianTape` instance containing a pre-recorded execution trace of `f` at the given `input`.

This `JacobianTape` can then be passed to `ReverseDiff.jacobian!` to take Jacobians of the execution trace with new `input` values.

See `ReverseDiff.jacobian` for a description of acceptable types for `input`.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/tape.jl#L116-L126' class='documenter-source'>source</a><br>


```
ReverseDiff.JacobianTape(f!, output, input, cfg::JacobianConfig = JacobianConfig(output, input))
```

Return a `JacobianTape` instance containing a pre-recorded execution trace of `f` at the given `output` and `input`.

This `JacobianTape` can then be passed to `ReverseDiff.jacobian!` to take Jacobians of the execution trace with new `input` values.

See `ReverseDiff.jacobian` for a description of acceptable types for `input`.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/tape.jl#L141-L151' class='documenter-source'>source</a><br>

<a id='ReverseDiff.HessianTape' href='#ReverseDiff.HessianTape'>#</a>
**`ReverseDiff.HessianTape`** &mdash; *Type*.



```
ReverseDiff.HessianTape(f, input, cfg::HessianConfig = HessianConfig(input))
```

Return a `HessianTape` instance containing a pre-recorded execution trace of `f` at the given `input`.

This `HessianTape` can then be passed to `ReverseDiff.hessian!` to take Hessians of the execution trace with new `input` values.

See `ReverseDiff.hessian` for a description of acceptable types for `input`.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/tape.jl#L172-L182' class='documenter-source'>source</a><br>

<a id='ReverseDiff.compile' href='#ReverseDiff.compile'>#</a>
**`ReverseDiff.compile`** &mdash; *Function*.



```
ReverseDiff.compile(t::AbstractTape)
```

Return a fully compiled representation of `t`. The type of this representation will be `CompiledGradient`/`CompiledJacobian`/`CompiledHessian`, depending on the type of `t`. This object can be passed to any API methods that accept `t`.

In many cases, compiling `t` can significantly speed up execution time. Note that the longer the tape, the more time compilation may take. Very long tapes (i.e. when `length(t)` is on the order of 10000 elements) can take a very long time to compile.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/tape.jl#L62-L72' class='documenter-source'>source</a><br>


<a id='The-AbstractConfig-API-1'></a>

## The `AbstractConfig` API


For the sake of convenience and performance, all "extra" information used by ReverseDiff's API methods is bundled up in the `ReverseDiff.AbstractConfig` family of types. These types allow the user to easily feed several different parameters to ReverseDiff's API methods, such as work buffers and tape configurations.


ReverseDiff's basic API methods will allocate these types automatically by default, but you can reduce memory usage and improve performance if you preallocate them yourself.

<a id='ReverseDiff.GradientConfig' href='#ReverseDiff.GradientConfig'>#</a>
**`ReverseDiff.GradientConfig`** &mdash; *Type*.



```
ReverseDiff.GradientConfig(input, tp::RawTape = RawTape())
```

Return a `GradientConfig` instance containing the preallocated tape and work buffers used by the `ReverseDiff.gradient`/`ReverseDiff.gradient!` methods.

Note that `input` is only used for type and shape information; it is not stored or modified in any way. It is assumed that the element type of `input` is same as the element type of the target function's output.

See `ReverseDiff.gradient` for a description of acceptable types for `input`.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/Config.jl#L23-L34' class='documenter-source'>source</a><br>


```
ReverseDiff.GradientConfig(input, ::Type{D}, tp::RawTape = RawTape())
```

Like `GradientConfig(input, tp)`, except the provided type `D` is assumed to be the element type of the target function's output.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/Config.jl#L39-L44' class='documenter-source'>source</a><br>

<a id='ReverseDiff.JacobianConfig' href='#ReverseDiff.JacobianConfig'>#</a>
**`ReverseDiff.JacobianConfig`** &mdash; *Type*.



```
ReverseDiff.JacobianConfig(input, tp::RawTape = RawTape())
```

Return a `JacobianConfig` instance containing the preallocated tape and work buffers used by the `ReverseDiff.jacobian`/`ReverseDiff.jacobian!` methods.

Note that `input` is only used for type and shape information; it is not stored or modified in any way. It is assumed that the element type of `input` is same as the element type of the target function's output.

See `ReverseDiff.jacobian` for a description of acceptable types for `input`.

```
ReverseDiff.JacobianConfig(input, ::Type{D}, tp::RawTape = RawTape())
```

Like `JacobianConfig(input, tp)`, except the provided type `D` is assumed to be the element type of the target function's output.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/Config.jl#L68-L84' class='documenter-source'>source</a><br>


```
ReverseDiff.JacobianConfig(output::AbstractArray, input, tp::RawTape = RawTape())
```

Return a `JacobianConfig` instance containing the preallocated tape and work buffers used by the `ReverseDiff.jacobian`/`ReverseDiff.jacobian!` methods. This method assumes the target function has the form `f!(output, input)`

Note that `input` and `output` are only used for type and shape information; they are not stored or modified in any way.

See `ReverseDiff.jacobian` for a description of acceptable types for `input`.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/Config.jl#L90-L101' class='documenter-source'>source</a><br>


```
ReverseDiff.JacobianConfig(result::DiffBase.DiffResult, input, tp::RawTape = RawTape())
```

A convenience method for `JacobianConfig(DiffBase.value(result), input, tp)`.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/Config.jl#L114-L118' class='documenter-source'>source</a><br>

<a id='ReverseDiff.HessianConfig' href='#ReverseDiff.HessianConfig'>#</a>
**`ReverseDiff.HessianConfig`** &mdash; *Type*.



```
ReverseDiff.HessianConfig(input::AbstractArray, gtp::RawTape = RawTape(), jtp::RawTape = RawTape())
```

Return a `HessianConfig` instance containing the preallocated tape and work buffers used by the `ReverseDiff.hessian`/`ReverseDiff.hessian!` methods. `gtp` is the tape used for the inner gradient calculation, while `jtp` is used for outer Jacobian calculation.

Note that `input` is only used for type and shape information; it is not stored or modified in any way. It is assumed that the element type of `input` is same as the element type of the target function's output.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/Config.jl#L130-L140' class='documenter-source'>source</a><br>


```
ReverseDiff.HessianConfig(input::AbstractArray, ::Type{D}, gtp::RawTape = RawTape(), jtp::RawTape = RawTape())
```

Like `HessianConfig(input, tp)`, except the provided type `D` is assumed to be the element type of the target function's output.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/Config.jl#L145-L150' class='documenter-source'>source</a><br>


```
ReverseDiff.HessianConfig(result::DiffBase.DiffResult, input::AbstractArray, gtp::RawTape = RawTape(), jtp::RawTape = RawTape())
```

Like `HessianConfig(input, tp)`, but utilize `result` along with `input` to construct work buffers.

Note that `result` and `input` are only used for type and shape information; they are not stored or modified in any way.


<a target='_blank' href='https://github.com/JuliaDiff/ReverseDiff.jl/tree/18691152c816da7041c1fd6255b9affb11681d84/src/api/Config.jl#L157-L165' class='documenter-source'>source</a><br>

