# ReverseDiff

[![Build Status](https://travis-ci.org/JuliaDiff/ReverseDiff.jl.svg?branch=master)](https://travis-ci.org/JuliaDiff/ReverseDiff.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaDiff/ReverseDiff.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaDiff/ReverseDiff.jl?branch=master)

[**Go To ReverseDiff's Documentation**](http://www.juliadiff.org/ReverseDiff.jl/)

[**See ReverseDiff Usage Examples**](https://github.com/JuliaDiff/ReverseDiff.jl/tree/master/examples)

**Note: While ReverseDiff technically supports Julia v0.7/v1.0 and is somewhat maintained, it
is currently not actively developed. Instead, ForwardDiff/ReverseDiff's maintainers are
focused on the development of a new AD package built on top of [Cassette](https://github.com/jrevels/Cassette.jl).
In the meantime, it might be worth checking out other reverse-mode AD implementations in Nabla.jl,
AutoGrad.jl, Flux.jl, or XGrad.jl.**

ReverseDiff implements methods to take **gradients**, **Jacobians**, **Hessians**, and
higher-order derivatives of native Julia functions (or any callable object, really) using
**reverse mode automatic differentiation (AD)**.

While performance can vary depending on the functions you evaluate, the algorithms
implemented by ReverseDiff **generally outperform non-AD algorithms in both speed and
accuracy.**

[Wikipedia's entry on automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation)
is a useful resource for learning about the advantages of AD techniques over other common
differentiation methods (such as [finite differencing](https://en.wikipedia.org/wiki/Numerical_differentiation)).

## Installation

To install ReverseDiff, simply use Julia's package manager:

```julia
julia> Pkg.add("ReverseDiff")
```

The current version of ReverseDiff supports Julia v0.5 (and intends to support Julia v0.6 once it is released).

## Why use ReverseDiff?

Other Julia packages may provide some of these features, but only ReverseDiff provides all
of them (as far as I know at the time of this writing):

- supports a large subset of the Julia language, including loops, recursion, and control flow
- user-friendly API for reusing and compiling tapes
- user-friendly performance annotations such as `@forward` and `@skip` (with more to come!)
- compatible with ForwardDiff, enabling mixed-mode AD
- built-in definitions leverage the benefits of ForwardDiff's `Dual` numbers (e.g. SIMD, zero-overhead arithmetic)
- a familiar differentiation API for ForwardDiff users
- non-allocating linear algebra optimizations
- nested differentiation
- suitable as an execution backend for graphical machine learning libraries
- ReverseDiff doesn't need to record scalar indexing operations (a huge cost for many similar libraries)
- higher-order `map` and `broadcast` optimizations
- it's well tested

...and, simply put, it's fast (for gradients, at least). Using the code from `examples/gradient.jl`:

```julia
julia> using BenchmarkTools

# this script defines f and ∇f!
julia> include(joinpath(Pkg.dir("ReverseDiff"), "examples/gradient.jl"));

julia> a, b = rand(100, 100), rand(100, 100);

julia> inputs = (a, b);

julia> results = (similar(a), similar(b));

# Benchmark the original objective function, sum(a' * b + a * b')
julia> @benchmark f($a, $b)
BenchmarkTools.Trial:
  memory estimate:  234.61 kb
  allocs estimate:  6
  --------------
  minimum time:     110.000 μs (0.00% GC)
  median time:      137.416 μs (0.00% GC)
  mean time:        173.085 μs (11.63% GC)
  maximum time:     3.613 ms (91.47% GC)

# Benchmark ∇f! at the same inputs (this is executing the function,
# getting the gradient w.r.t. `a`, and getting the gradient w.r.t
# to `b` simultaneously). Notice that the whole thing is
# non-allocating.
julia> @benchmark ∇f!($results, $inputs)
BenchmarkTools.Trial:
  memory estimate:  0.00 bytes
  allocs estimate:  0
  --------------
  minimum time:     429.650 μs (0.00% GC)
  median time:      431.460 μs (0.00% GC)
  mean time:        469.916 μs (0.00% GC)
  maximum time:     937.512 μs (0.00% GC)
```

I've used this benchmark (and others) to pit ReverseDiff against every other native
Julia reverse-mode AD package that I know of (including source-to-source packages),
and have found ReverseDiff to be faster and use less memory in most cases.

## Should I use ReverseDiff or ForwardDiff?

ForwardDiff is algorithmically more efficient for differentiating functions where the input
dimension is less than the output dimension, while ReverseDiff is algorithmically more
efficient for differentiating functions where the output dimension is less than the input
dimension.

Thus, ReverseDiff is generally a better choice for gradients, but Jacobians and Hessians are
trickier to determine. For example, optimized methods for computing nested derivatives might
use a combination of forward-mode and reverse-mode AD.

ForwardDiff is often faster than ReverseDiff for lower dimensional gradients (`length(input)
< 100`), or gradients of functions where the number of input parameters is small compared
to the number of operations performed on them. ReverseDiff is often faster if your code
is expressed as a series of array operations, e.g. a composition of Julia's Base linear
algebra methods.

In general, your choice of algorithms will depend on the function being differentiated, and
you should benchmark different methods to see how they fare.
