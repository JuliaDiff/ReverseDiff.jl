# Limitations of ReverseDiff

ReverseDiff works by injecting user code with new number types that record all operations
that occur on them, accumulating an execution trace of the target function which can be
re-run forwards and backwards to propagate new input values and derivative information.
Naturally, this technique has some limitations. Here's a list of all the roadblocks we've
seen users run into ("target function" here refers to the function being differentiated):

- **The target function can only be composed of generic Julia functions.** ReverseDiff cannot propagate derivative information through non-Julia code. Thus, your function may not work if it makes calls to external, non-Julia programs, e.g. uses explicit BLAS calls instead of `Ax_mul_Bx`-style functions.

- **The target function must be written generically enough to accept numbers of type `T<:Real` as input (or arrays of these numbers).** The function doesn't require a specific type signature, as long as the type signature is generic enough to avoid breaking this rule. This also means that any storage assigned used within the function must be generic as well.

- **Nested differentiation of closures is dangerous.** Differentiating closures is safe, and nested differentation is safe, but you might be vulnerable to a subtle bug if you try to do both. See [this ForwardDiff issue](https://github.com/JuliaDiff/ForwardDiff.jl/issues/83) for details. A fix is currently being planned for this problem.

- **The types of array inputs must be subtypes of `AbstractArray`.**

- **Array inputs that are being differentiated cannot be mutated**. This also applies to any "descendent" arrays that must be tracked (e.g. if `A` is an immutable input array, then `C = A * A` will also be immutable). If you try to perform `setindex!` on such arrays, an error will be thrown. In the future, this restriction might be lifted. Note that arrays explicitly constructed within the target function (e.g. via `ones`, `similar`, etc.) can be mutated, as well as output arrays used when taking the Jacobian of a function of the form `f!(output, input....).`
