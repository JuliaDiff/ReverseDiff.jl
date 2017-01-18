@printf("Consolidating tape for gradients...\n")
include("mnist.jl")

@printf("Loading batch of size %d\n", BATCH_SIZE)
# Construct the initial batch.
batch = Batch(TRAIN_IMAGES, TRAIN_LABELS, 1);

# Pre-allocate a reusable `Batch` for gradient storage.
# Note that this is essentially just a storage buffer;
# the initial values don't matter.
∇batch = Batch(TRAIN_IMAGES, TRAIN_LABELS, 1);

rate = 4.2e-5

@printf("Training model with learning rate %f ...\n", rate)

train_all!(∇batch, batch, TRAIN_IMAGES, TRAIN_LABELS, rate, 1000)

@printf("The model accuracy is: %7.4f\n", test_model(batch))
