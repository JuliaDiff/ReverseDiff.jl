using ReverseDiff: GradientTape, GradientConfig, gradient, gradient!, compile

#########
# setup #
#########

# some objective function to work with
f(a, b) = sum(a' * b + a * b')

# pre-record a GradientTape for `f` using inputs of shape 100x100 with Float64 elements
const f_tape = GradientTape(f, (rand(100, 100), rand(100, 100)))

# compile `f_tape` into a more optimized representation
const compiled_f_tape = compile(f_tape)

# some inputs and work buffers to play around with
a, b = rand(100, 100), rand(100, 100)
inputs = (a, b)
results = (similar(a), similar(b))
all_results = map(DiffBase.GradientResult, results)
cfg = GradientConfig(inputs)

####################
# taking gradients #
####################

# with pre-recorded/compiled tapes (generated in the setup above) #
#-----------------------------------------------------------------#

# this should be the fastest method, and non-allocating
gradient!(results, compiled_f_tape, inputs)

# the same as the above, but in addition to calculating the gradients, the value `f(a, b)`
# is loaded into the the provided `DiffResult` instances (see DiffBase.jl documentation).
gradient!(all_results, compiled_f_tape, inputs)

# this should be the second fastest method, and also non-allocating
gradient!(results, f_tape, inputs)

# you can also make your own function if you want to abstract away the tape
∇f!(results, inputs) = gradient!(results, compiled_f_tape, inputs)

# with a pre-allocated GradientConfig #
#-------------------------------------#
# these methods are more flexible than a pre-recorded tape, but can be
# wasteful since the tape will be re-recorded for every call.

gradient!(results, f, inputs, cfg)

gradient(f, inputs, cfg)

# without a pre-allocated GradientConfig #
#----------------------------------------#
# convenient, but pretty wasteful since it has to allocate the GradientConfig itself

gradient!(results, f, inputs)

gradient(f, inputs)

"""
    ReverseDiff.makeGradients(f, input :: Vector{<:Real})

Returns `(∇f!, f∇f!, g, yg)`.

`∇f!` takes a value similar to `input`, and returns the gradient at that point.
This gradient is also written to `g` to allow more efficient use that avoids memory allocation.

`f∇f!` takes a value similar to `input`, and returns both the function value and the gradient at that point.
This pair (function value, gradient) is also written to `yg` to allow more efficient use that avoids memory allocation.

Note: THIS USE CASE IS NOT THREAD SAFE.
"""
makeGradients(f, x0) = begin
  const f_tape = GradientTape(f, x0)
  const compiled_f_tape = compile(f_tape)
  g = similar(x0)
  yg = DiffBase.GradientResult(g)
  cfg = GradientConfig(x0)
  ∇f!(x)  = gradient!(g, compiled_f_tape, x)
  f∇f!(x) = gradient!(yg, compiled_f_tape, x)
  return(∇f!, f∇f!, g, yg)
end
