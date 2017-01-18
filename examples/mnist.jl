using MNIST, ReverseDiff

const BATCH_SIZE = 100
const IMAGE_SIZE = 784
const CLASS_COUNT = 10

##################
# data wrangling #
##################

# loading MNIST data #
#--------------------#

function load_mnist(data)
    images, label_indices = data
    labels = zeros(CLASS_COUNT, length(label_indices))
    for i in eachindex(label_indices)
        labels[Int(label_indices[i]) + 1, i] = 1.0
    end
    return images, labels
end

const TRAIN_IMAGES, TRAIN_LABELS = load_mnist(MNIST.traindata())
const TEST_IMAGES, TEST_LABELS = load_mnist(MNIST.testdata())

# loading batches #
# -----------------#

immutable Batch{W,B,P,L}
    weights::W
    bias::B
    pixels::P
    labels::L
end

function Batch(images, labels, i)
    weights = zeros(CLASS_COUNT, IMAGE_SIZE)
    bias = zeros(CLASS_COUNT)
    range = i:(i + BATCH_SIZE - 1)
    return Batch(weights, bias, images[:, range], labels[:, range])
end

function load_batch!(batch, images, labels, i)
    offset = i - 1
    for batch_col in 1:BATCH_SIZE
        data_col = batch_col + offset
        for k in 1:size(images, 1)
            batch.pixels[k, batch_col] = images[k, data_col]
        end
        for k in 1:size(labels, 1)
            batch.labels[k, batch_col] = labels[k, data_col]
        end
    end
    return batch
end

####################
# model definition #
####################

# Here, we make a model out of simple `softmax` and `cross_entropy` functions. This could be
# improved by implementing something like Tensorflow's `softmax_cross_entropy_with_logits`,
# but my main goal is to show off ReverseDiff rather than implement the best possible model.

# Also note that our input's orientation is transposed compared to example implementations
# presented by row-major frameworks like Tensorflow. Julia is column-major, so I've set up
# the `Batch` code (see above) such that each column of `pixels` is an image and
# `size(pixels, 2) == BATCH_SIZE`.

# objective definitions #
#-----------------------#

# Here we use `@forward` to tell ReverseDiff to differentiate this scalar function in
# forward-mode. This allows us to call `minus_log.(y)` instead of `-(log.(y))`. By defining
# our own "fused" `minus_log` kernel using `@forward`, the operation `minus_log.(y)` becomes
# a single array instruction in the tape (instead of two separate ones,  as is the case with
# `-(log.(y))`), buying us a slight performance gain. Note that we didn't *need* to do this;
# it's simply a good place to show off the `@forward` feature.
ReverseDiff.@forward minus_log(x::Real) = -log(x)

# # Obsolete: compute cross_entropy and softmax in two separated functions
# # may introduce more numerical instability than putting them into one
# # single function.
# # Thus this function is no longer used.
# function cross_entropy(y′, y)
#     # add a floor and ceiling to the input data
#     y = max(y, eps(eltype(y)))
#     y = min(y, 1 - eps(eltype(y)))
#
#     entropy = mean(sum(y′ .* (minus_log.(y)), 1))
#     return entropy
# end
#
# # Obsolete: used only when with StatsFuns.logsumexp to
# # normalize columns
# function col_normalize(A::AbstractArray)
#     for (col,s) in enumerate(sum(A,1))
#         s == 0 && continue # What does a "normalized" column with a sum of zero look like?
#         A[:,col] = A[:,col]/s
#     end
#     A
# end

function dim_logsumexp(x, dim)
    # take the log(sum(exp)) of a 2-d array in a numerical stable way
    # excluding extreme cases
    # Note: we need something to check for NaNs and throw error message
    S = typeof(x)
    isempty(x) && return -S(Inf)
    u = max(0, maximum(x))
    abs(u) == Inf && return any(isnan, x) ? S(NaN) : u
    # the actual formula
    return log(sum(exp(x - u), dim)) + u
end

function softmax_cross_entropy(y′, y)
    # compte softmax and cross entropy together in a numerical stable way
    # since the outputing softmax is a ratio, rebase_x produces the same result as original x.
    max_y = max(0, maximum(y))
    rebase_y = y - max_y
    logsumexp = log(sum(exp(rebase_y), 1))
    softmax = exp(rebase_y - repmat(logsumexp, size(rebase_y, 1), 1))

    # # The following way is safer: it deals with the extreme cases
    # # with dim_logsumexp(), but don't know how to check NaNs with tape.
    # y = exp(rebase_x - repmat(dim_logsumexp(rebase_x, 1), size(rebase_x, 1), 1))

    # constrain on the extreme values of y
    softmax = min(softmax, 1 - eps(eltype(softmax)))
    softmax = max(softmax, eps(eltype(softmax)))

    # A more general definition of cross entropy
    # entropy = mean(sum(y′ .* (minus_log.(y)), 1))
    # The special case for binary y
    # Explanation here: https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_error_function_and_logistic_regression
    entropy = mean(sum(y′ .* minus_log.(softmax) + (1 - y′) .* minus_log.(1 - softmax), 1) ./ CLASS_COUNT)
    # Note: it is confirmed that the NaNs were because of the log() function.
    # There must be some y <= 0 that result in NaNs for large learning rates.
    return entropy
end

function model(weights, bias, pixels, labels)
    y = (weights * pixels) .+ bias
    #y = softmax(y)
    #return cross_entropy(labels, y)
    return softmax_cross_entropy(labels, y)
end

# gradient definitions #
#----------------------#

# generate the gradient function `∇model!(output, input)` from `model`
const ∇model! = begin
    # grab a sample batch as our seed data
    batch = Batch(TRAIN_IMAGES, TRAIN_LABELS, 1)

    # `compile_gradient` takes array arguments - it doesn't know anything about `Batch`
    input = (batch.weights, batch.bias, batch.pixels, batch.labels)

    # generate the gradient function (which is then returned from this `begin` block and
    # bound to `∇model!`). See the `ReverseDiff.compile_gradient` docs for details.
    ReverseDiff.compile_gradient(model, input)
end

# Add convenience method to `∇model!` that translates `Batch` args to `Tuple` args. Since
# `∇model!` is a binding to a function type generated by ReverseDiff, we can overload
# the type's calling behavior just like any other Julia type. For details, see
# http://docs.julialang.org/en/release-0.5/manual/methods/#function-like-objects.
function (::typeof(∇model!))(output::Batch, input::Batch)
    output_tuple = (output.weights, output.bias, output.pixels, output.labels)
    input_tuple = (input.weights, input.bias, input.pixels, input.labels)
    return ∇model!(output_tuple, input_tuple)
end

############
# training #
############

# This function is used to check if the fields in batch are valid/reasonable
# by summing the contents. Really hacky but quick way to do the work
function check_valid(batch::Batch)
    return (sum(batch.weights), sum(batch.bias), sum(batch.pixels), sum(batch.labels))
end

# Train each individual batch once for a time
function train_batch!(∇batch::Batch, batch::Batch, rate, labels)
    ∇model!(∇batch, batch)
    # Warning: this line is never excuted
    if any(isnan, check_valid(batch)) == NaN
        DomainError()
    end
    for i in eachindex(batch.weights)
        batch.weights[i] -= rate * ∇batch.weights[i]
    end
    for i in eachindex(batch.bias)
        batch.bias[i] -= rate * ∇batch.bias[i]
    end
end

# Note: for BATCH_SIZE = 100, rate ≈ 3.227e-4 is the edge of producing
# NaNs, i.e. rate = 3.227e-4 will produce NaNs within 5 iterations
# It is now confirmed that the NaNs are produced because of log() in
# the cross_entropy, so we need to throw an error/warning message when
# NaNs happen.
function train_all!(∇batch::Batch, batch::Batch, images, labels, rate = 4.2e-5, iters = 1000)
    # batch_count = floor(Int, size(images, 2) / BATCH_SIZE)
    for i in 1:iters
        # The usage of batch size is to train model on smaller sets
        # so as to use GPU (or distributed computation) later
        batch_ofs = (i - 1) * BATCH_SIZE + 1
        batch_ofs = batch_ofs > size(TRAIN_IMAGES, 2) ? batch_ofs % size(TRAIN_IMAGES, 2) : batch_ofs

        load_batch!(batch, images, labels, batch_ofs)
        train_batch!(∇batch, batch, rate, labels)
        # Warning: the following line is never excuted
        if any(isnan, check_valid(batch)) == NaN
            DomainError()
        end
    end
    return batch
end

#######################
# running the example #
#######################

#=

# load the code
include(joinpath(Pkg.dir("ReverseDiff"), "examples", "mnist.jl"))

# Construct the initial batch.
batch = Batch(TRAIN_IMAGES, TRAIN_LABELS, 1);

# Pre-allocate a reusable `Batch` for gradient storage.
# Note that this is essentially just a storage buffer;
# the initial values don't matter.
∇batch = Batch(TRAIN_IMAGES, TRAIN_LABELS, 1);

train_all!(∇batch, batch, TRAIN_IMAGES, TRAIN_LABELS)

=#


# test the model using TEST_IMAGES
# The functin is used to calculate column-wise matches
function accuracy(result::Array, label::Array)
    score = 0
    ncol = size(label, 2)
    for i in 1:ncol
        if indmax(result[:, i]) == indmax(label[:, i])
            score += 1
        end
    end
    score / ncol
end

# test the model with test images
# I did the test all at once without batches
# Can also test the model using batches
function test_model(batch::Batch)
    # load all test images
    #tbatch = Batch(TEST_IMAGES, TEST_LABELS, 1);
    tlabels = batch.weights * TEST_IMAGES .+ batch.bias
    # compute accuracy and return
    return accuracy(tlabels, TEST_LABELS)
end

# This function is used to find the best learning rate by running
# the model with given num_iters and print test accuracies
# Input: start_rate: the learning rate to start with
#        step_size: step size for learning rate
#        num_step: number of different learning rates you want to try
#        num_iters: the param to pass into train_all!()
function choose_params(start_rate = 3e-5, step_size = .5e-5, num_step = 10, num_iters = 1000)
    result = zeros(num_step)
    rate = start_rate
    for i in 1:num_step
        rate += step_size
        batch = Batch(TRAIN_IMAGES, TRAIN_LABELS, 1);
        ∇batch = Batch(TRAIN_IMAGES, TRAIN_LABELS, 1);

        train_all!(∇batch, batch, TRAIN_IMAGES, TRAIN_LABELS, rate, num_iters)
        result[i] = test_model(batch)
        println((i, rate, result[i]))
    end
    return result
end
