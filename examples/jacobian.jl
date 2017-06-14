using ReverseDiff: JacobianTape, JacobianConfig, jacobian, jacobian!, compile

#########
# setup #
#########

# some objective functions to work with
f(a, b) = (a + b) * (a * b)'
g!(out, a, b) = A_mul_Bc!(out, a + b, a * b)

# pre-record JacobianTapes for `f` and `g` using inputs of shape 10x10 with Float64 elements
const f_tape = JacobianTape(f, (rand(10, 10), rand(10, 10)))
const g_tape = JacobianTape(g!, rand(10, 10), (rand(10, 10), rand(10, 10)))

# compile `f_tape` and `g_tape` into more optimized representations
const compiled_f_tape = compile(f_tape)
const compiled_g_tape = compile(g_tape)

# some inputs and work buffers to play around with
a, b = rand(10, 10), rand(10, 10)
inputs = (a, b)
output = rand(10, 10)
results = (similar(a, 100, 100), similar(b, 100, 100))
fcfg = JacobianConfig(inputs)
gcfg = JacobianConfig(output, inputs)

####################
# taking Jacobians #
####################

# with pre-recorded/compiled tapes (generated in the setup above) #
#-----------------------------------------------------------------#

# these should be the fastest methods, and non-allocating
jacobian!(results, compiled_f_tape, inputs)
jacobian!(results, compiled_g_tape, inputs)

# these should be the second fastest methods, and also non-allocating
jacobian!(results, f_tape, inputs)
jacobian!(results, g_tape, inputs)

# with a pre-allocated JacobianConfig #
#-------------------------------------#
# this is more flexible than a pre-recorded tape, but can be wasteful since the tape
# will be re-recorded for every call.

jacobian!(results, f, inputs, fcfg)

jacobian(f, inputs, fcfg)

jacobian!(results, g!, output, inputs, gcfg)

jacobian(g!, output, inputs, gcfg)

# without a pre-allocated JacobianConfig #
#----------------------------------------#
# convenient, but pretty wasteful since it has to allocate the JacobianConfig itself

jacobian!(results, f, inputs)

jacobian(f, inputs)

jacobian!(results, g!, output, inputs)

jacobian(g!, output, inputs)
