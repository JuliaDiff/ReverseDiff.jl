# ReverseDiff

**This package and it's dependencies are not yet fully released. Tests should pass locally, but fail on Travis. The installation instructions will only work once the package is registered; until then, use `Pkg.clone`.**

[![Build Status](https://travis-ci.org/JuliaDiff/ReverseDiff.jl.svg?branch=master)](https://travis-ci.org/JuliaDiff/ReverseDiff.jl)

[**Go To ReverseDiff's Documentation**](http://www.juliadiff.org/DiffBase.jl/)

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

```
    julia> Pkg.add("ReverseDiff")
```

The current version of ReverseDiff supports Julia v0.5 and v0.6.

## Why use ReverseDiff?

Other Julia packages may provide some of these features, but only ReverseDiff provides all
of them (as far as I know at the time of this writing):

- supports most of the Julia language, including loops, recursion, and control flow
- user-friendly API for reusing and compiling tapes
- user-friendly performance annotations such as `@forward` and `@skip` (with more to come!)
- compatible with ForwardDiff, enabling mixed-mode AD
- built-in definitions leverage the benefits of ForwardDiff's `Dual` numbers (e.g. SIMD, zero-overhead arithmetic)
- a familiar differentiation API for ForwardDiff users
- non-allocating linear algebra optimizations
- suitable as an execution backend for graphical machine learning libraries
- ReverseDiff doesn't need to record scalar indexing operations (a huge cost for many similar libraries)
- higher-order `map` and `broadcast` optimizations

## Should I use ReverseDiff or ForwardDiff?

ForwardDiff is theoretically more efficient for differentiating functions where the input
dimension is less than the output dimension, while ReverseDiff is theoretically more
efficient for differentiating functions where the output dimension is less than the
input dimension.

Thus, ReverseDiff is generally a better choice for gradients, but Jacobians and Hessians are
trickier to determine. For example, optimized methods for computing nested derivatives might
use a combination of forward-mode and reverse-mode AD.

ForwardDiff is often faster than ReverseDiff for lower dimensional gradients (`length(input)
< 100`), or gradients of functions that are implemented as very large programs.

In general, your choice of algorithms will depend on the function being differentiated, and
you should benchmark different methods to see how they fare.
