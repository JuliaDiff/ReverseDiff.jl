using ReverseDiff: GradientTape, GradientConfig, compile, gradient!, gradient

#########
# setup #
#########

# generates a gradient function for any recordable function at the given inputs
function generate_gradient(f, inputs)
    tape = compile(GradientTape(f, inputs))
    return (results, inputs) -> gradient!(results, tape, inputs)
end

# some objective function to work with
f(a, b) = sum(a' * b + a * b')

# generate a gradient function for `f` using inputs of shape 100x100 with Float64 elements
const ∇f! = generate_gradient(f, (rand(100, 100), rand(100, 100)))

# some inputs and work buffers to play around with
a, b = rand(100, 100), rand(100, 100)
inputs = (a, b)
results = (similar(a), similar(b))
cfg = GradientConfig(inputs)

####################
# taking gradients #
####################

# with a pre-recorded/comiled tape (generated in the setup above) #
#-----------------------------------------------------------------#
# this should be very fast and non-allocating

∇f!(results, inputs)

# with a pre-allocated GradientConfig #
#-------------------------------------#
# this should still be fairly fast, but will allocate memory for recording the tape

gradient!(results, f, inputs, cfg)

gradient(f, inputs, cfg)


# without a pre-allocated GradientConfig #
#----------------------------------------#
# this will be the slowest and allocate the most

gradient!(results, f, inputs)

gradient(f, inputs)
