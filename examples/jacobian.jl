using ReverseDiff: JacobianTape, JacobianConfig, compile, jacobian!, jacobian

#########
# setup #
#########

# generates a jacobin function for any recordable function at the given inputs
function generate_jacobian(f!, output, inputs)
    tape = compile(JacobianTape(f!, output, inputs))
    return (results, inputs) -> jacobian!(results, tape, inputs)
end

function generate_jacobian(f, inputs)
    tape = compile(JacobianTape(f, inputs))
    return (results, inputs) -> jacobian!(results, tape, inputs)
end

# some objective functions to work with
f(a, b) = (a + b) * (a * b)'
g!(out, a, b) = A_mul_Bc!(out, a + b, a * b)

# generate a jacobian function for `f` using inputs of shape 10x10 with Float64 elements
const Jf! = generate_jacobian(f, (rand(10, 10), rand(10, 10)))
const Jg! = generate_jacobian(g!, rand(10, 10), (rand(10, 10), rand(10, 10)))

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

# with a pre-recorded/comiled tape (generated in the setup above) #
#-----------------------------------------------------------------#
# this should be the fastest method, and non-allocating

Jf!(results, inputs)

Jg!(results, inputs)

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
